import argparse
import json
from pathlib import Path

from track_iou import build_tracks, tracks_to_jsonable


def parse_args():
    parser = argparse.ArgumentParser(description="Run IoU temporal tracking on OWL-ViT detections.")
    parser.add_argument("--dets-json", type=str, default="outputs/owlvit/dets_top1.json")
    parser.add_argument("--output-json", type=str, default="outputs/owlvit/tracks.json")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--min-track-len", type=int, default=1)
    parser.add_argument("--agg", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    dets_path = Path(args.dets_json)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with dets_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    detections_by_frame = payload.get("detections", payload)

    tracks = build_tracks(
        detections_by_frame=detections_by_frame,
        iou_threshold=args.iou_threshold,
        max_gap=args.max_gap,
        score_threshold=args.score_threshold,
        min_track_len=args.min_track_len,
    )
    tracks_json = tracks_to_jsonable(tracks, agg=args.agg, window=args.window)

    out = {
        "meta": {
            "source_dets_json": str(dets_path),
            "iou_threshold": args.iou_threshold,
            "score_threshold": args.score_threshold,
            "max_gap": args.max_gap,
            "min_track_len": args.min_track_len,
            "agg": args.agg,
            "window": args.window,
            "num_tracks": len(tracks_json),
        },
        "tracks": tracks_json,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote tracks to: {out_path}")
    print(f"Tracks: {len(tracks_json)}")


if __name__ == "__main__":
    main()
