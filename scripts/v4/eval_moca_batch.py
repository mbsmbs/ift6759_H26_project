import argparse
import csv
from pathlib import Path
from statistics import mean

from eval_moca_detection import evaluate, load_dets, load_gt_from_moca


def parse_args():
    # Paramètres batch pour évaluer plusieurs vidéos MoCA en une seule commande.
    parser = argparse.ArgumentParser(description="Batch evaluation of MoCA videos for V4 detections.")
    parser.add_argument("--dets-json", type=str, required=True)
    parser.add_argument("--annotations-csv", type=str, default="data/MoCA/Annotations/annotations.csv")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-det-per-frame", type=int, default=1)
    parser.add_argument("--videos", nargs="+", default=None, help="Optional explicit list of videos.")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--output-csv", type=str, default="outputs/owlvit/final/eval_moca_batch.csv")
    return parser.parse_args()


def discover_videos(annotations_csv: Path):
    # Déduit la liste des vidéos à partir des clés frame "video/frame.jpg".
    gt_all = load_gt_from_moca(annotations_csv, video=None)
    videos = sorted({k.split("/", 1)[0] for k in gt_all.keys()})
    return videos


def main():
    args = parse_args()
    annotations_csv = Path(args.annotations_csv)
    dets_json = Path(args.dets_json)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.videos:
        videos = args.videos
    else:
        videos = discover_videos(annotations_csv)

    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    rows = []
    for video in videos:
        # Évaluation indépendante par vidéo pour faciliter l’analyse comparative.
        gt_by_frame = load_gt_from_moca(annotations_csv, video=video)
        dets_by_frame = load_dets(dets_json, video=video, max_det_per_frame=args.max_det_per_frame)
        result = evaluate(
            gt_by_frame=gt_by_frame,
            dets_by_frame=dets_by_frame,
            iou_thr=args.iou_threshold,
            score_thr=args.score_threshold,
        )
        row = {"video": video, **result}
        rows.append(row)
        print(
            f"{video}: AP50={row['ap50']:.4f} P={row['op_precision']:.4f} "
            f"R={row['op_recall']:.4f} F1={row['op_f1']:.4f} GT={row['num_gt_frames']}"
        )

    if not rows:
        print("No videos to evaluate.")
        return

    fieldnames = [
        "video",
        "num_gt_frames",
        "num_predictions_considered",
        "ap50",
        "op_score_threshold",
        "op_iou_threshold",
        "op_tp",
        "op_fp",
        "op_fn",
        "op_precision",
        "op_recall",
        "op_f1",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Moyennes macro: chaque vidéo contribue de manière égale.
    macro = {
        "videos": len(rows),
        "ap50": mean(r["ap50"] for r in rows),
        "precision": mean(r["op_precision"] for r in rows),
        "recall": mean(r["op_recall"] for r in rows),
        "f1": mean(r["op_f1"] for r in rows),
    }
    print(
        f"Macro over {macro['videos']} videos: AP50={macro['ap50']:.4f} "
        f"P={macro['precision']:.4f} R={macro['recall']:.4f} F1={macro['f1']:.4f}"
    )
    print(f"Wrote CSV to: {out_csv}")


if __name__ == "__main__":
    main()
