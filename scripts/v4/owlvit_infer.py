import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

# Forcer Transformers en mode PyTorch uniquement pour éviter
# des imports TensorFlow/Keras instables sur certains environnements.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


DEFAULT_PROMPTS = [
    "a camouflaged animal",
    "a snake",
    "an animal hidden in sand",
]


def load_model(model_name: str, device: str):
    # Charge OWL-ViT une seule fois pour éviter le surcoût par frame.
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def predict_one(
    image_path: Path,
    prompts: List[str],
    threshold: float,
    processor: OwlViTProcessor,
    model: OwlViTForObjectDetection,
    device: str,
) -> List[Dict[str, float]]:
    # Inférence sur une image: boîtes + score + prompt_id + ratio de surface.
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    image_area = float(max(1, img_w * img_h))
    inputs = processor(text=[prompts], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        box_w = max(0.0, float(x2 - x1))
        box_h = max(0.0, float(y2 - y1))
        box_area_ratio = (box_w * box_h) / image_area
        prompt_id = int(label.item())
        detections.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "score": float(score.item()),
                "prompt_id": prompt_id,
                "class_id": 0,
                "box_area_ratio": float(box_area_ratio),
            }
        )
    return detections


def parse_args():
    # Paramètres d'inférence batch et de filtrage simple des boîtes.
    parser = argparse.ArgumentParser(description="OWL-ViT batch inference on MoCA frames.")
    parser.add_argument("--input-root", default="data/MoCA/JPEGImages", type=str)
    parser.add_argument("--video", required=True, type=str, help="Video folder name inside input root.")
    parser.add_argument("--output-json", default="outputs/owlvit/dets.json", type=str)
    parser.add_argument("--model-name", default="google/owlvit-base-patch32", type=str)
    parser.add_argument("--threshold", default=0.15, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument(
        "--top-k-per-frame",
        default=None,
        type=int,
        help="Keep only top-K detections by score per frame (e.g., 1).",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        default=None,
        type=float,
        help="Drop detections with area ratio above this value (e.g., 0.35).",
    )
    parser.add_argument(
        "--keep-top1-if-empty",
        action="store_true",
        help="If filtering removes all detections in a frame, keep the best original detection.",
    )
    parser.add_argument(
        "--area-penalty-lambda",
        default=0.0,
        type=float,
        help="Compactness penalty: final_score = score - lambda * box_area_ratio.",
    )
    parser.add_argument("--prompts", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    input_root = Path(args.input_root)
    video_dir = input_root / args.video
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    frame_paths = sorted(video_dir.glob("*.jpg"))
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    processor, model = load_model(args.model_name, device)

    detections = {}
    total = len(frame_paths)
    for idx, frame_path in enumerate(frame_paths, start=1):
        # Clé normalisée attendue par les autres scripts: "video/frame.jpg".
        key = frame_path.relative_to(input_root).as_posix()
        frame_dets = predict_one(
            image_path=frame_path,
            prompts=prompts,
            threshold=args.threshold,
            processor=processor,
            model=model,
            device=device,
        )
        for det in frame_dets:
            # Score pénalisé par la taille de boîte pour limiter les boîtes trop larges.
            det["score_compact"] = float(det["score"] - args.area_penalty_lambda * det.get("box_area_ratio", 0.0))

        frame_dets = sorted(frame_dets, key=lambda d: d["score_compact"], reverse=True)
        original_frame_dets = list(frame_dets)
        if args.max_box_area_ratio is not None:
            # Supprime les boîtes dépassant un ratio de surface maximal.
            frame_dets = [d for d in frame_dets if d.get("box_area_ratio", 1.0) <= args.max_box_area_ratio]
        if args.keep_top1_if_empty and not frame_dets and original_frame_dets:
            # Mécanisme de secours: évite les frames totalement vides.
            frame_dets = [original_frame_dets[0]]
        if args.top_k_per_frame is not None and args.top_k_per_frame >= 0:
            frame_dets = frame_dets[: args.top_k_per_frame]
        detections[key] = frame_dets
        if idx % 20 == 0 or idx == total:
            print(f"[{idx}/{total}] processed {key}")

    payload = {
        "meta": {
            "model_name": args.model_name,
            "device": device,
            "threshold": args.threshold,
            "top_k_per_frame": args.top_k_per_frame,
            "max_box_area_ratio": args.max_box_area_ratio,
            "keep_top1_if_empty": args.keep_top1_if_empty,
            "area_penalty_lambda": args.area_penalty_lambda,
            "prompts": prompts,
            "input_root": str(input_root),
            "video": args.video,
            "num_frames": total,
        },
        "detections": detections,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote detections to: {out_path}")


if __name__ == "__main__":
    main()
