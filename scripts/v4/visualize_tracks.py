import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize temporal tracks on frames.")
    parser.add_argument("--tracks-json", type=str, default="outputs/owlvit/tracks_top1.json")
    parser.add_argument("--images-root", type=str, default="data/MoCA/JPEGImages")
    parser.add_argument("--output-dir", type=str, default="outputs/owlvit/vis_tracks")
    parser.add_argument("--video", type=str, default=None, help="Optional video filter.")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    tracks_path = Path(args.tracks_json)
    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tracks_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frame_to_items = defaultdict(list)
    for track in payload.get("tracks", []):
        track_id = track.get("track_id", -1)
        for det in track.get("detections", []):
            if float(det.get("score", 0.0)) < args.min_score:
                continue
            frame = det["frame"]
            if args.video and not frame.startswith(f"{args.video}/"):
                continue
            frame_to_items[frame].append(
                {
                    "track_id": int(track_id),
                    "x1": float(det["x1"]),
                    "y1": float(det["y1"]),
                    "x2": float(det["x2"]),
                    "y2": float(det["y2"]),
                    "score": float(det["score"]),
                }
            )

    frame_keys = sorted(frame_to_items.keys())
    if args.max_frames is not None:
        frame_keys = frame_keys[: args.max_frames]

    for idx, frame_key in enumerate(frame_keys, start=1):
        image_path = images_root / frame_key
        if not image_path.exists():
            print(f"[skip] image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item in frame_to_items[frame_key]:
            draw.rectangle(
                [(item["x1"], item["y1"]), (item["x2"], item["y2"])],
                outline="lime",
                width=3,
            )
            label = f"track {item['track_id']} | s={item['score']:.2f}"
            draw.rectangle(
                (item["x1"], max(0, item["y1"] - 16), min(image.width, item["x1"] + 180), item["y1"]),
                fill="lime",
            )
            draw.text((item["x1"] + 2, max(0, item["y1"] - 15)), label, fill="black")

        out_name = frame_key.replace("/", "__")
        out_path = out_dir / out_name
        image.save(out_path)
        print(f"[{idx}/{len(frame_keys)}] wrote {out_path}")


if __name__ == "__main__":
    main()
