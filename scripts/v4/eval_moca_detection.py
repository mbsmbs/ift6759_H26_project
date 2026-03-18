import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    # Évaluation détection (frame-level) entre prédictions V4 et annotations MoCA.
    parser = argparse.ArgumentParser(description="Evaluate OWL-ViT detections against MoCA GT annotations.")
    parser.add_argument("--dets-json", type=str, required=True, help="Path to detections json.")
    parser.add_argument(
        "--annotations-csv",
        type=str,
        default="data/MoCA/Annotations/annotations.csv",
        help="Path to MoCA annotations.csv",
    )
    parser.add_argument("--video", type=str, default=None, help="Optional video filter (e.g., arabian_horn_viper).")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for TP matching.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Score threshold used for precision/recall summary point.",
    )
    parser.add_argument(
        "--max-det-per-frame",
        type=int,
        default=1,
        help="Use at most K highest-score detections per frame for evaluation.",
    )
    return parser.parse_args()


def iou_xyxy(a: Dict[str, float], b: Dict[str, float]) -> float:
    # Calcule l'IoU entre deux boîtes au format (x1, y1, x2, y2).
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
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def load_gt_from_moca(csv_path: Path, video: str = None) -> Dict[str, Dict[str, float]]:
    # Charge une GT par frame (MoCA contient une boîte principale par image annotée).
    gt = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]

    for row in rows:
        # VIA export columns:
        # 0: metadata_id, 1: file_list, 2: flags, 3: temporal_coordinates
        # 4: spatial_coordinates, 5: metadata, ...
        file_list = row[1].lstrip("/")  # e.g., arabian_horn_viper/00000.jpg
        if video and not file_list.startswith(f"{video}/"):
            continue

        spatial = json.loads(row[4])  # [2, x, y, w, h]
        if int(spatial[0]) != 2:
            continue
        x, y, w, h = float(spatial[1]), float(spatial[2]), float(spatial[3]), float(spatial[4])
        gt[file_list] = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
    return gt


def load_dets(dets_json: Path, video: str = None, max_det_per_frame: int = 1) -> Dict[str, List[Dict[str, float]]]:
    # Lit le JSON de détections et garde au plus K boîtes par frame (triées par score).
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


def flatten_predictions(
    dets_by_frame: Dict[str, List[Dict[str, float]]],
    gt_by_frame: Dict[str, Dict[str, float]],
) -> List[Tuple[str, Dict[str, float], float]]:
    # Aplatit toutes les prédictions valides en une liste triée par score décroissant.
    preds = []
    for frame_key, dets in dets_by_frame.items():
        if frame_key not in gt_by_frame:
            continue
        for d in dets:
            preds.append((frame_key, d, float(d.get("score", 0.0))))
    preds.sort(key=lambda x: x[2], reverse=True)
    return preds


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    # AP par interpolation de type VOC sur la courbe précision-rappel.
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate(
    gt_by_frame: Dict[str, Dict[str, float]],
    dets_by_frame: Dict[str, List[Dict[str, float]]],
    iou_thr: float,
    score_thr: float,
):
    # 1) Calcule la courbe PR globale (AP@IoU_thr).
    # 2) Calcule un point opératoire au seuil de score fixé.
    preds = flatten_predictions(dets_by_frame, gt_by_frame)
    num_gt = len(gt_by_frame)
    matched = set()
    tps, fps, scores = [], [], []

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
    for tp, fp in zip(tps, fps):
        cum_tp += tp
        cum_fp += fp
        rec = cum_tp / num_gt if num_gt > 0 else 0.0
        prec = cum_tp / max(1, (cum_tp + cum_fp))
        recalls.append(rec)
        precisions.append(prec)

    ap50 = compute_ap(recalls, precisions) if scores else 0.0

    # Point opératoire unique au seuil de score demandé.
    matched_thr = set()
    tp_thr, fp_thr = 0, 0
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

    fn_thr = num_gt - tp_thr
    precision_thr = tp_thr / max(1, (tp_thr + fp_thr))
    recall_thr = tp_thr / num_gt if num_gt > 0 else 0.0
    f1_thr = (2 * precision_thr * recall_thr / (precision_thr + recall_thr)) if (precision_thr + recall_thr) > 0 else 0.0

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


def main():
    args = parse_args()
    gt_by_frame = load_gt_from_moca(Path(args.annotations_csv), video=args.video)
    dets_by_frame = load_dets(
        Path(args.dets_json),
        video=args.video,
        max_det_per_frame=args.max_det_per_frame,
    )
    result = evaluate(
        gt_by_frame=gt_by_frame,
        dets_by_frame=dets_by_frame,
        iou_thr=args.iou_threshold,
        score_thr=args.score_threshold,
    )

    tag = args.video if args.video else "ALL"
    print(f"MoCA Detection Eval [{tag}]")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
