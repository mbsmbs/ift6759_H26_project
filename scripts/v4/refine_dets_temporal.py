import argparse
import json
from pathlib import Path
from typing import Dict, List


def iou_xyxy(a: Dict[str, float], b: Dict[str, float]) -> float:
    # IoU entre deux boîtes (x1, y1, x2, y2).
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


def parse_args():
    # Raffinement temporel léger: sélection d'une boîte cohérente d'une frame à l'autre.
    parser = argparse.ArgumentParser(description="Temporal refinement for per-frame OWL-ViT detections.")
    parser.add_argument("--input-dets-json", type=str, required=True)
    parser.add_argument("--output-dets-json", type=str, required=True)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Use only top-k candidates per frame before refinement.")
    parser.add_argument("--iou-weight", type=float, default=0.35, help="Temporal consistency weight.")
    parser.add_argument("--area-penalty-lambda", type=float, default=0.0, help="Penalty on bbox area ratio.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop very low-score candidates before refinement.")
    parser.add_argument("--keep-empty", action="store_true", help="Keep empty frames; otherwise fallback to best raw candidate.")
    return parser.parse_args()


def det_compact_score(det: Dict[str, float], area_lambda: float) -> float:
    # Score ajusté pour pénaliser les boîtes trop grandes.
    score = float(det.get("score", 0.0))
    area_ratio = float(det.get("box_area_ratio", 0.0))
    return score - area_lambda * area_ratio


def refine_video(
    frame_keys: List[str],
    detections: Dict[str, List[Dict[str, float]]],
    top_k: int,
    iou_weight: float,
    area_lambda: float,
    min_score: float,
    keep_empty: bool,
) -> Dict[str, List[Dict[str, float]]]:
    # Sélectionne 1 détection/frame en combinant score et cohérence IoU avec la frame précédente.
    out = {}
    prev_selected = None

    for fk in frame_keys:
        raw = [d for d in detections.get(fk, []) if float(d.get("score", 0.0)) >= min_score]
        if not raw:
            out[fk] = []
            continue

        ranked = sorted(raw, key=lambda d: det_compact_score(d, area_lambda), reverse=True)
        candidates = ranked[: max(1, top_k)]

        if prev_selected is None:
            chosen = candidates[0]
        else:
            # Critère temporel: score compact + bonus d'IoU avec la boîte précédente retenue.
            chosen = max(
                candidates,
                key=lambda d: det_compact_score(d, area_lambda) + iou_weight * iou_xyxy(prev_selected, d),
            )

        out[fk] = [chosen]
        prev_selected = chosen

    if not keep_empty:
        # Fallback optionnel: si une frame est vide après filtrage, garder le meilleur brut.
        for fk in frame_keys:
            if out.get(fk):
                continue
            raw = detections.get(fk, [])
            if raw:
                out[fk] = [max(raw, key=lambda d: float(d.get("score", 0.0)))]
            else:
                out[fk] = []

    return out


def main():
    args = parse_args()
    in_path = Path(args.input_dets_json)
    out_path = Path(args.output_dets_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    detections = payload.get("detections", payload)
    all_keys = sorted(detections.keys())
    if args.video:
        prefix = f"{args.video}/"
        frame_keys = [k for k in all_keys if k.startswith(prefix)]
    else:
        frame_keys = all_keys

    refined_subset = refine_video(
        frame_keys=frame_keys,
        detections=detections,
        top_k=args.top_k,
        iou_weight=args.iou_weight,
        area_lambda=args.area_penalty_lambda,
        min_score=args.min_score,
        keep_empty=args.keep_empty,
    )

    refined_all = dict(detections)
    for k, v in refined_subset.items():
        refined_all[k] = v

    meta = payload.get("meta", {})
    meta["temporal_refine"] = {
        "input": str(in_path),
        "video": args.video,
        "top_k": args.top_k,
        "iou_weight": args.iou_weight,
        "area_penalty_lambda": args.area_penalty_lambda,
        "min_score": args.min_score,
        "keep_empty": args.keep_empty,
    }

    out_payload = {"meta": meta, "detections": refined_all}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Wrote refined detections to: {out_path}")


if __name__ == "__main__":
    main()
