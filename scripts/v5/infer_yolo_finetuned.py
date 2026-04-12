import argparse
import json
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO


DEFAULT_VIDEOS = [
    "arabian_horn_viper",
    "arctic_fox",
    "arctic_fox_1",
    "arctic_fox_2",
    "arctic_fox_3",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference YOLO fine-tune MoCA -> JSON compatible evaluate_v4.py"
    )
    parser.add_argument("--input-root", type=str, default="data/MoCA/JPEGImages")
    parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/yolo/models/yolo_finetune_moca_run1/weights/best.pt",
        help="Checkpoint YOLO fine-tuné (best.pt).",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu, mps, cuda:0")
    parser.add_argument("--conf", type=float, default=0.001, help="Seuil confiance YOLO avant top-k.")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="IoU NMS YOLO.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--top-k-per-frame", type=int, default=1)
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/yolo/final/yolo_finetuned_moca_dets_top1_5videos.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    model_path = Path(args.model)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")

    model = YOLO(str(model_path))

    detections: Dict[str, List[Dict[str, float]]] = {}
    total_frames = 0
    kept_dets = 0

    for video in args.videos:
        video_dir = input_root / video
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Video folder not found: {video_dir}")

        frame_paths = sorted(video_dir.glob("*.jpg"))
        total_frames += len(frame_paths)
        print(f"[video={video}] num_frames={len(frame_paths)}")

        for idx, frame_path in enumerate(frame_paths, start=1):
            result = model.predict(
                source=str(frame_path),
                conf=args.conf,
                iou=args.nms_iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            frame_key = frame_path.relative_to(input_root).as_posix()
            frame_dets: List[Dict[str, float]] = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().tolist()
                scores = result.boxes.conf.cpu().tolist()

                for box, score in zip(boxes_xyxy, scores):
                    x1, y1, x2, y2 = box
                    frame_dets.append(
                        {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "score": float(score),
                            "class_id": 0,
                        }
                    )

            frame_dets = sorted(frame_dets, key=lambda d: d["score"], reverse=True)
            if args.top_k_per_frame is not None and args.top_k_per_frame >= 0:
                frame_dets = frame_dets[: args.top_k_per_frame]

            kept_dets += len(frame_dets)
            detections[frame_key] = frame_dets

            if idx % 50 == 0 or idx == len(frame_paths):
                print(f"[{video}] {idx}/{len(frame_paths)}")

    payload = {
        "meta": {
            "model": str(model_path),
            "device": args.device,
            "input_root": str(input_root),
            "videos": args.videos,
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "imgsz": args.imgsz,
            "top_k_per_frame": args.top_k_per_frame,
            "num_frames": total_frames,
            "num_detections_kept": kept_dets,
        },
        "detections": detections,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote detections: {out_path}")
    print(f"Total frames: {total_frames}")
    print(f"Total detections kept: {kept_dets}")


if __name__ == "__main__":
    main()
