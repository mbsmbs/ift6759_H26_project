import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

# Calcule le IoU entre 2 bounding boxes a et b
def iou_xyxy(a: Dict[str, float], b: Dict[str, float]) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    denom = area_a + area_b - inter # Union
    return 0.0 if denom <= 0 else inter / denom

# Charger GT depuis MoCA -> Dict de bounding boxes
def load_gt_from_moca(csv_path: Path, video: str = None) -> Dict[str, Dict[str, float]]:
    '''
        - lit les annotaions
        - filtre les bonnes lignes
        - convertit les boxes
        - retourne un dictionnaire pret pour l'evaluation
    '''
    gt = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    for row in rows:
        file_key = row[1].lstrip("/") # nom de l'image
        if video and not file_key.startswith(f"{video}/"):
            continue
        spatial = json.loads(row[4])  # [2, x, y, w, h]
        if int(spatial[0]) != 2:
            continue
        x, y, w, h = float(spatial[1]), float(spatial[2]), float(spatial[3]), float(spatial[4])
        gt[file_key] = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
    return gt

# Charger les detections depuis un JSON -> prepare les donnees pour l'evaluation
def load_dets(dets_json: Path, video: str = None, max_det_per_frame: int = 1) -> Dict[str, List[Dict[str, float]]]:
    with dets_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    dets = payload.get("detections", payload)

    out = {}
    for frame_key, lst in dets.items():
        if video and not frame_key.startswith(f"{video}/"):
            continue
        ranked = sorted(lst, key=lambda d: float(d.get("score", 0.0)), reverse=True)
        if max_det_per_frame is not None and max_det_per_frame >= 0:
            ranked = ranked[:max_det_per_frame]
        out[frame_key] = ranked
    return out

# Mettre les predictions dans une seule liste triee par score
def flatten_predictions(
    dets_by_frame: Dict[str, List[Dict[str, float]]],
    gt_by_frame: Dict[str, Dict[str, float]],
) -> List[Tuple[str, Dict[str, float], float]]:
    '''
        - Fusionne toutes les predictions
        - filtre celles sans GT
        - trie par score
    '''
    preds = []
    for frame_key, dets in dets_by_frame.items():
        if frame_key not in gt_by_frame:
            continue
        for d in dets:
            preds.append((frame_key, d, float(d.get("score", 0.0))))
    preds.sort(key=lambda x: x[2], reverse=True)
    return preds

# Calcule AP : Aire sous la courbe Precision-Recall
def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    # Construire les bornes
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    # Rendre la courbe de precision monotone decroissante
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Calculer l'aire sous la courbe
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_one(
    gt_by_frame: Dict[str, Dict[str, float]],
    dets_by_frame: Dict[str, List[Dict[str, float]]],
    iou_thr: float,
    score_thr: float,
) -> Dict[str, float]:
    
    preds = flatten_predictions(dets_by_frame, gt_by_frame)
    num_gt = len(gt_by_frame)
    matched = set()
    tps, fps, scores = [], [], []

    # Distinguer TP/FP pour chaque prediction
    for frame_key, det, score in preds:
        gt = gt_by_frame[frame_key]
        iou = iou_xyxy(gt, det)
        is_tp = iou >= iou_thr and frame_key not in matched
        if is_tp:
            matched.add(frame_key)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
        scores.append(score)

    cum_tp, cum_fp = 0, 0
    recalls, precisions = [], []
    # Parcourir les predictions -> calculer les courbes Precision-Recall
    for tp, fp in zip(tps, fps):
        cum_tp += tp
        cum_fp += fp
        rec = cum_tp / num_gt if num_gt > 0 else 0.0
        prec = cum_tp / max(1, (cum_tp + cum_fp))
        recalls.append(rec)
        precisions.append(prec)

    # Calculer AP
    ap50 = compute_ap(recalls, precisions) if scores else 0.0

    matched_thr = set()
    tp_thr, fp_thr = 0, 0
    # Re-evaluer TP/FP avec les seuils fixes (score_thr, iou_thr)
    for frame_key, det, score in preds:
        if score < score_thr:
            continue
        gt = gt_by_frame[frame_key]
        iou = iou_xyxy(gt, det)
        is_tp = iou >= iou_thr and frame_key not in matched_thr
        if is_tp:
            matched_thr.add(frame_key)
            tp_thr += 1
        else:
            fp_thr += 1

    # Calculer FN, Precision, Recall, F1 pour les seuils fixes
    fn_thr = num_gt - tp_thr
    precision_thr = tp_thr / max(1, (tp_thr + fp_thr))
    recall_thr = tp_thr / num_gt if num_gt > 0 else 0.0
    f1_thr = (2 * precision_thr * recall_thr / (precision_thr + recall_thr)) if (precision_thr + recall_thr) > 0 else 0.0

    # Retourner tous les metrics
    return {
        "num_gt_frames": num_gt,
        "num_predictions_considered": len(scores),
        "ap50": ap50,
        "op_score_threshold": score_thr,
        "op_iou_threshold": iou_thr,
        "op_tp": tp_thr,
        "op_fp": fp_thr,
        "op_fn": fn_thr,
        "op_precision": precision_thr,
        "op_recall": recall_thr,
        "op_f1": f1_thr,
    }

# Trouver les videos uniques dans les annotations pour evaluation
def discover_videos(annotations_csv: Path) -> List[str]:
    gt_all = load_gt_from_moca(annotations_csv, video=None)
    return sorted({k.split("/", 1)[0] for k in gt_all.keys()})

# Evaluer une detections json sur plusieurs videos -> retourner les metrics par video + macro
def evaluate_batch(
    dets_json: Path,
    annotations_csv: Path,
    videos: List[str],
    iou_thr: float,
    score_thr: float,
    max_det_per_frame: int,
) -> Tuple[List[Dict], Dict]:
    rows = []
    # Evaluer chaque video independamment -> calculer les metrics par video
    for video in videos:
        gt = load_gt_from_moca(annotations_csv, video=video)
        dets = load_dets(dets_json, video=video, max_det_per_frame=max_det_per_frame)
        r = evaluate_one(gt, dets, iou_thr=iou_thr, score_thr=score_thr)
        rows.append({"video": video, **r})
    # Calculer les metrics macro (moyenne sur les videos)
    macro = {
        "video": "MACRO",
        "num_videos": len(rows),
        "ap50": mean(r["ap50"] for r in rows) if rows else 0.0,
        "precision": mean(r["op_precision"] for r in rows) if rows else 0.0,
        "recall": mean(r["op_recall"] for r in rows) if rows else 0.0,
        "f1": mean(r["op_f1"] for r in rows) if rows else 0.0,
    }
    return rows, macro

# Tester plusieurs parametres (IoU, score threshold, max det per frame) -> ecrire les resultats par video + macro dans un CSV + trouver la meilleure combinaison par F1 et AP50
def run_sweep(args):
    # 1. Read inputs
    dets_json = Path(args.dets_json)
    annotations_csv = Path(args.annotations_csv)
    videos = args.videos if args.videos else discover_videos(annotations_csv)
    # 2. List of parameters
    ious = parse_float_list(args.iou_thresholds)
    scores = parse_float_list(args.score_thresholds)
    ks = parse_int_list(args.max_det_per_frame_list)

    # 3. Try multiple combination
    grid_rows = []
    macro_rows = []
    for iou_thr in ious: # higher IoU -> stricter, lower IoU -> easier
        for score_thr in scores: # higher -> fewer, more confident detections, lower -> more detections, possibly more false positives
            for k in ks:
                rows, macro = evaluate_batch(
                    dets_json=dets_json,
                    annotations_csv=annotations_csv,
                    videos=videos,
                    iou_thr=iou_thr,
                    score_thr=score_thr,
                    max_det_per_frame=k,
                )
                
                for r in rows:
                    # Save all per-video results
                    grid_rows.append(
                        {
                            "video": r["video"],
                            "iou_threshold": iou_thr,
                            "score_threshold": score_thr,
                            "max_det_per_frame": k,
                            "num_gt_frames": r["num_gt_frames"],
                            "num_predictions_considered": r["num_predictions_considered"],
                            "ap50": r["ap50"],
                            "op_precision": r["op_precision"],
                            "op_recall": r["op_recall"],
                            "op_f1": r["op_f1"],
                        }
                    )
                macro_rows.append(
                    {
                        "video": "MACRO",
                        "iou_threshold": iou_thr,
                        "score_threshold": score_thr,
                        "max_det_per_frame": k,
                        "num_videos": macro["num_videos"],
                        "ap50": macro["ap50"],
                        "precision": macro["precision"],
                        "recall": macro["recall"],
                        "f1": macro["f1"],
                    }
                )
                print(
                    f"IoU={iou_thr:.2f} score={score_thr:.2f} k={k} -> "
                    f"AP50={macro['ap50']:.4f} P={macro['precision']:.4f} "
                    f"R={macro['recall']:.4f} F1={macro['f1']:.4f}"
                )

    out_grid = Path(args.out_grid_csv)
    out_best = Path(args.out_best_csv)
    out_grid.parent.mkdir(parents=True, exist_ok=True)

    grid_fields = [
        "video",
        "iou_threshold",
        "score_threshold",
        "max_det_per_frame",
        "num_gt_frames",
        "num_predictions_considered",
        "ap50",
        "op_precision",
        "op_recall",
        "op_f1",
    ]
    with out_grid.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=grid_fields)
        w.writeheader()
        w.writerows(grid_rows)

    best_f1 = max(macro_rows, key=lambda x: x["f1"])
    best_ap = max(macro_rows, key=lambda x: x["ap50"])
    best_fields = [
        "criterion",
        "video",
        "iou_threshold",
        "score_threshold",
        "max_det_per_frame",
        "num_videos",
        "ap50",
        "precision",
        "recall",
        "f1",
    ]
    with out_best.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=best_fields)
        w.writeheader()
        w.writerow({"criterion": "best_macro_f1", **best_f1})
        w.writerow({"criterion": "best_macro_ap50", **best_ap})

    print(f"Wrote grid CSV: {out_grid}")
    print(f"Wrote best CSV: {out_best}")

# Parse the version specification str for the master mode
def parse_version_spec(raw: str) -> Tuple[str, Path, float, str]:
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) < 3:
        raise ValueError(
            "Each --version must be: name|dets_json|score_threshold|note(optional)"
        )
    name = parts[0]
    dets_json = Path(parts[1])
    score_thr = float(parts[2])
    note = parts[3] if len(parts) > 3 else ""
    return name, dets_json, score_thr, note

# Point d'entree pour evaluation de plusieurs versions -> ecrire les resultats par video + macro dans un CSV
def run_master(args):
    annotations_csv = Path(args.annotations_csv)
    videos = args.videos if args.videos else discover_videos(annotations_csv)
    out = Path(args.out_master_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw in args.version:
        name, dets_json, score_thr, note = parse_version_spec(raw)
        _, macro = evaluate_batch(
            dets_json=dets_json,
            annotations_csv=annotations_csv,
            videos=videos,
            iou_thr=args.iou_threshold,
            score_thr=score_thr,
            max_det_per_frame=args.max_det_per_frame,
        )
        rows.append(
            {
                "version": name,
                "score_threshold": score_thr,
                "ap50_macro": macro["ap50"],
                "precision_macro": macro["precision"],
                "recall_macro": macro["recall"],
                "f1_macro": macro["f1"],
                "note": note,
                "dets_json": str(dets_json),
            }
        )
        print(
            f"{name}: AP50={macro['ap50']:.4f} P={macro['precision']:.4f} "
            f"R={macro['recall']:.4f} F1={macro['f1']:.4f} (score={score_thr})"
        )

    fields = [
        "version",
        "score_threshold",
        "ap50_macro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "note",
        "dets_json",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote master CSV: {out}")


def build_parser():
    p = argparse.ArgumentParser(description="Single-file evaluator for V4 (sweep/master).")
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("sweep", help="Sweep score/IoU thresholds for one detections json.")
    ps.add_argument("--dets-json", type=str, required=True)
    ps.add_argument("--annotations-csv", type=str, default="data/MoCA/Annotations/annotations.csv")
    ps.add_argument("--videos", nargs="+", default=None)
    ps.add_argument("--iou-thresholds", type=str, default="0.5")
    ps.add_argument("--score-thresholds", type=str, default="0.01,0.03,0.05,0.08,0.1")
    ps.add_argument("--max-det-per-frame-list", type=str, default="1")
    ps.add_argument("--out-grid-csv", type=str, default="outputs/owlvit/final/eval_sweep_grid.csv")
    ps.add_argument("--out-best-csv", type=str, default="outputs/owlvit/final/eval_sweep_best.csv")

    pm = sub.add_parser("master", help="Evaluate multiple versions and write one comparison CSV.")
    pm.add_argument(
        "--version",
        action="append",
        required=True,
        help="name|dets_json|score_threshold|note(optional). Repeat for each version.",
    )
    pm.add_argument("--annotations-csv", type=str, default="data/MoCA/Annotations/annotations.csv")
    pm.add_argument("--videos", nargs="+", default=None)
    pm.add_argument("--iou-threshold", type=float, default=0.5)
    pm.add_argument("--max-det-per-frame", type=int, default=1)
    pm.add_argument("--out-master-csv", type=str, default="outputs/owlvit/final/v4_eval_master.csv")
    return p


def main():
    # 1. Parse args
    args = build_parser().parse_args()
    # 2. Dispatch to mode
    if args.mode == "sweep": # quel est le meilleur seuil pour ce modele?
        run_sweep(args)
    elif args.mode == "master": # quelle version est la meilleure?
        run_master(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
