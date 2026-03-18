import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import List

from eval_moca_detection import evaluate, load_dets, load_gt_from_moca


def parse_float_list(raw: str) -> List[float]:
    # Accepte "0.3,0.5,0.7" et renvoie [0.3, 0.5, 0.7].
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    # Accepte "1,3,5" et renvoie [1, 3, 5].
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep d'evaluation MoCA pour V4 sur plusieurs seuils IoU/score."
    )
    parser.add_argument("--dets-json", type=str, required=True)
    parser.add_argument("--annotations-csv", type=str, default="data/MoCA/Annotations/annotations.csv")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--iou-thresholds", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--score-thresholds", type=str, default="0.0,0.1,0.2,0.3")
    parser.add_argument("--max-det-per-frame-list", type=str, default="1")
    parser.add_argument(
        "--out-grid-csv",
        type=str,
        default="outputs/owlvit/final/eval_moca_sweep_grid.csv",
    )
    parser.add_argument(
        "--out-best-csv",
        type=str,
        default="outputs/owlvit/final/eval_moca_sweep_best.csv",
    )
    return parser.parse_args()


def evaluate_one_setting(
    dets_json: Path,
    annotations_csv: Path,
    videos: List[str],
    iou_thr: float,
    score_thr: float,
    max_det_per_frame: int,
):
    rows = []
    for video in videos:
        gt_by_frame = load_gt_from_moca(annotations_csv, video=video)
        dets_by_frame = load_dets(dets_json, video=video, max_det_per_frame=max_det_per_frame)
        result = evaluate(
            gt_by_frame=gt_by_frame,
            dets_by_frame=dets_by_frame,
            iou_thr=iou_thr,
            score_thr=score_thr,
        )
        rows.append({"video": video, **result})
    return rows


def macro_row(rows: List[dict], iou_thr: float, score_thr: float, max_det_per_frame: int):
    return {
        "video": "MACRO",
        "iou_threshold": iou_thr,
        "score_threshold": score_thr,
        "max_det_per_frame": max_det_per_frame,
        "num_videos": len(rows),
        "ap50": mean(r["ap50"] for r in rows),
        "precision": mean(r["op_precision"] for r in rows),
        "recall": mean(r["op_recall"] for r in rows),
        "f1": mean(r["op_f1"] for r in rows),
    }


def main():
    args = parse_args()
    dets_json = Path(args.dets_json)
    annotations_csv = Path(args.annotations_csv)
    out_grid = Path(args.out_grid_csv)
    out_best = Path(args.out_best_csv)
    out_grid.parent.mkdir(parents=True, exist_ok=True)

    iou_thresholds = parse_float_list(args.iou_thresholds)
    score_thresholds = parse_float_list(args.score_thresholds)
    max_det_values = parse_int_list(args.max_det_per_frame_list)

    grid_rows = []
    macro_rows = []

    for iou_thr in iou_thresholds:
        for score_thr in score_thresholds:
            for max_det in max_det_values:
                setting_rows = evaluate_one_setting(
                    dets_json=dets_json,
                    annotations_csv=annotations_csv,
                    videos=args.videos,
                    iou_thr=iou_thr,
                    score_thr=score_thr,
                    max_det_per_frame=max_det,
                )

                for r in setting_rows:
                    grid_rows.append(
                        {
                            "video": r["video"],
                            "iou_threshold": iou_thr,
                            "score_threshold": score_thr,
                            "max_det_per_frame": max_det,
                            "num_gt_frames": r["num_gt_frames"],
                            "num_predictions_considered": r["num_predictions_considered"],
                            "ap50": r["ap50"],
                            "op_precision": r["op_precision"],
                            "op_recall": r["op_recall"],
                            "op_f1": r["op_f1"],
                        }
                    )

                macro = macro_row(setting_rows, iou_thr, score_thr, max_det)
                macro_rows.append(macro)
                print(
                    f"IoU={iou_thr:.2f} score={score_thr:.2f} k={max_det} "
                    f"-> AP50={macro['ap50']:.4f} P={macro['precision']:.4f} "
                    f"R={macro['recall']:.4f} F1={macro['f1']:.4f}"
                )

    # Écrit la grille complète (par vidéo + paramétrage).
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
        writer = csv.DictWriter(f, fieldnames=grid_fields)
        writer.writeheader()
        writer.writerows(grid_rows)

    # Conserve les meilleurs réglages macro selon F1 et AP.
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
        writer = csv.DictWriter(f, fieldnames=best_fields)
        writer.writeheader()
        writer.writerow({"criterion": "best_macro_f1", **best_f1})
        writer.writerow({"criterion": "best_macro_ap50", **best_ap})

    print(f"Wrote grid CSV: {out_grid}")
    print(f"Wrote best CSV: {out_best}")


if __name__ == "__main__":
    main()
