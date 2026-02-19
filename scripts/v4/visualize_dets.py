import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize detections from dets.json on images.")
    parser.add_argument("--dets-json", type=str, default="outputs/owlvit/dets.json")
    parser.add_argument("--images-root", type=str, default="data/MoCA/JPEGImages")
    parser.add_argument("--output-dir", type=str, default="outputs/owlvit/vis")
    parser.add_argument("--video", type=str, default=None, help="Optional video filter.")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames to render.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score to draw a box.")
    return parser.parse_args()


def main():
    args = parse_args()
    dets_path = Path(args.dets_json)
    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with dets_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    prompts = payload.get("meta", {}).get("prompts", [])
    detections = payload.get("detections", payload)

    frame_keys = sorted(detections.keys())
    if args.video:
        prefix = f"{args.video}/"
        frame_keys = [k for k in frame_keys if k.startswith(prefix)]
    if args.max_frames is not None:
        frame_keys = frame_keys[: args.max_frames]

    for idx, frame_key in enumerate(frame_keys, start=1):
        image_path = images_root / frame_key
        if not image_path.exists():
            print(f"[skip] image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for det in detections.get(frame_key, []):
            score = float(det.get("score", 0.0))
            if score < args.min_score:
                continue

            x1 = float(det["x1"])
            y1 = float(det["y1"])
            x2 = float(det["x2"])
            y2 = float(det["y2"])
            prompt_id = int(det.get("prompt_id", -1))
            prompt_text = prompts[prompt_id] if 0 <= prompt_id < len(prompts) else f"prompt_{prompt_id}"
            label = f"{prompt_text} | s={score:.2f}"

            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            text_bg = (x1, max(0, y1 - 16), min(image.width, x1 + 420), y1)
            draw.rectangle(text_bg, fill="red")
            draw.text((x1 + 2, max(0, y1 - 15)), label, fill="white")

        out_name = frame_key.replace("/", "__")
        out_path = out_dir / out_name
        image.save(out_path)
        print(f"[{idx}/{len(frame_keys)}] wrote {out_path}")


if __name__ == "__main__":
    main()
