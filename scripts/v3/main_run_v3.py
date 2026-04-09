from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import configF_dev as config

from motion_proposals import Proposal, add_full_frame_fallback, generate_motion_proposals
from clip_rerank import CLIPReranker, crop_xyxy_from_bgr


DEFAULT_PROMPTS = [
    "a camouflaged animal",
    "an animal hidden in nature",
    "a hidden animal",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config_value(name: str, default=None):
    return getattr(config, name, default)


def resolve_config_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    return REPO_ROOT / path


def load_prompts() -> List[str]:
    prompts = get_config_value("PROMPTS", None)
    prompts_json = get_config_value("PROMPTS_JSON", None)

    if prompts is not None:
        if not prompts:
            raise ValueError("PROMPTS in config.py is empty.")
        return [str(p) for p in prompts]

    if prompts_json is not None:
        path = resolve_config_path(prompts_json)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            prompts = payload
        elif isinstance(payload, dict) and "prompts" in payload:
            prompts = payload["prompts"]
        else:
            raise ValueError("PROMPTS_JSON must point to a JSON list or a dict with key 'prompts'.")

        if not prompts:
            raise ValueError("Prompt list loaded from PROMPTS_JSON is empty.")
        return [str(p) for p in prompts]

    return list(DEFAULT_PROMPTS)


def read_video_names(images_root: Path, single_video: str | None, video_list: str | None) -> List[str]:
    if single_video:
        return [single_video]

    if video_list:
        path = resolve_config_path(video_list)
        with path.open("r", encoding="utf-8") as f:
            videos = [line.strip() for line in f if line.strip()]
        return videos

    return sorted([p.name for p in images_root.iterdir() if p.is_dir()])


def list_frames(video_dir: Path) -> List[Path]:
    return sorted([p for p in video_dir.iterdir() if p.is_file()])


def normalize_motion_scores(proposals: Sequence[Proposal]) -> List[float]:
    if not proposals:
        return []

    return [min(float(p.motion_score) / 50.0, 1.0) for p in proposals]


def build_detection_record(
    proposal: Proposal,
    score: float,
    clip_score: float,
    motion_score_norm: float,
    prompt_id: int,
    class_id: int = 0,
) -> Dict[str, float]:
    return {
        "x1": float(proposal.x1),
        "y1": float(proposal.y1),
        "x2": float(proposal.x2),
        "y2": float(proposal.y2),
        "score": float(score),
        "clip_score": float(clip_score),
        "motion_score": float(proposal.motion_score),
        "motion_score_norm": float(motion_score_norm),
        "prompt_id": int(prompt_id),
        "class_id": int(class_id),
    }


def process_video(
    video_name: str,
    video_dir: Path,
    reranker: CLIPReranker,
    prompts: Sequence[str],
    score_alpha: float,
    score_beta: float,
    diff_threshold: int,
    blur_ksize: int,
    morph_kernel: int,
    morph_iterations: int,
    min_area: int,
    max_area_ratio: float,
    box_expand: float,
    proposal_top_k: int,
    proposal_nms_iou: float,
    top_k_per_frame: int,
    keep_full_frame_fallback: bool,
    max_frames_per_video: int | None,
    frame_k: int,
) -> Dict[str, List[Dict]]:
    frame_paths = list_frames(video_dir)

    if max_frames_per_video is not None:
        frame_paths = frame_paths[:max_frames_per_video]

    if len(frame_paths) == 0:
        print(f"[warn] No frames found in {video_dir}")
        return {}

    if frame_k < 1:
        raise ValueError(f"FRAME_K must be >= 1, got {frame_k}")

    detections_by_frame: Dict[str, List[Dict]] = {}

    for frame_idx in range(len(frame_paths)):
        curr_path = frame_paths[frame_idx]
        curr_bgr = cv2.imread(str(curr_path), cv2.IMREAD_COLOR)

        if curr_bgr is None:
            print(f"[warn] Could not read frame: {curr_path}")
            continue

        future_idx = frame_idx + frame_k
        frame_key = f"{video_name}/{curr_path.name}"

        # No future frame available for t+k
        if future_idx >= len(frame_paths):
            detections_by_frame[frame_key] = []
            continue

        future_path = frame_paths[future_idx]
        future_bgr = cv2.imread(str(future_path), cv2.IMREAD_COLOR)

        if future_bgr is None:
            print(f"[warn] Could not read future frame: {future_path}")
            detections_by_frame[frame_key] = []
            continue

        proposals, _ = generate_motion_proposals(
            prev_bgr=curr_bgr,
            curr_bgr=future_bgr,
            diff_threshold=diff_threshold,
            blur_ksize=blur_ksize,
            morph_kernel=morph_kernel,
            morph_iterations=morph_iterations,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            box_expand=box_expand,
            top_k=proposal_top_k,
            nms_iou=proposal_nms_iou,
        )

        if keep_full_frame_fallback:
            proposals = add_full_frame_fallback(
                proposals,
                image_shape=future_bgr.shape,
                fallback_score=1.0,
            )

        if len(proposals) == 0:
            detections_by_frame[frame_key] = []
            continue

        # Use crops from the future frame t+k
        crops = [
            crop_xyxy_from_bgr(future_bgr, p.x1, p.y1, p.x2, p.y2)
            for p in proposals
        ]

        ranked = reranker.rank_candidates(crops=crops, prompts=prompts)
        motion_score_norms = normalize_motion_scores(proposals)

        scored_items = []
        for item in ranked:
            idx = item.candidate_index
            clip_score = float(item.clip_score)
            motion_norm = float(motion_score_norms[idx])
            final_score = float(score_alpha * motion_norm + score_beta * clip_score)

            scored_items.append(
                (
                    final_score,
                    build_detection_record(
                        proposal=proposals[idx],
                        score=final_score,
                        clip_score=clip_score,
                        motion_score_norm=motion_norm,
                        prompt_id=item.prompt_index,
                        class_id=0,
                    ),
                )
            )

        scored_items.sort(key=lambda x: x[0], reverse=True)
        final_dets = [det for _, det in scored_items[:top_k_per_frame]]
        detections_by_frame[frame_key] = final_dets

        print(
            f"[{video_name}] frame {frame_idx + 1}/{len(frame_paths)} "
            f"{curr_path.name} -> {future_path.name}: "
            f"proposals={len(proposals)} kept={len(final_dets)}"
        )

    return detections_by_frame


def main():
    images_root = resolve_config_path(
        get_config_value("IMAGES_ROOT", Path("data") / "MoCA" / "JPEGImages")
    )
    output_json = resolve_config_path(
        get_config_value("OUTPUT_JSON", Path("outputs") / "v3" / "dev_predictions.json")
    )

    single_video = get_config_value("VIDEO", None)
    video_list = get_config_value("VIDEO_LIST", None)

    diff_threshold = get_config_value("DIFF_THRESHOLD", 25)
    blur_ksize = get_config_value("BLUR_KSIZE", 5)
    morph_kernel = get_config_value("MORPH_KERNEL", 5)
    morph_iterations = get_config_value("MORPH_ITERATIONS", 2)
    min_area = get_config_value("MIN_AREA", 400)
    max_area_ratio = get_config_value("MAX_AREA_RATIO", 0.60)
    box_expand = get_config_value("BOX_EXPAND", 0.15)
    proposal_top_k = get_config_value("PROPOSAL_TOP_K", 10)
    proposal_nms_iou = get_config_value("PROPOSAL_NMS_IOU", 0.5)

    clip_model_name = get_config_value("CLIP_MODEL_NAME", "ViT-B-32")
    clip_pretrained = get_config_value("CLIP_PRETRAINED", "openai")
    clip_hf_model_name = get_config_value("CLIP_HF_MODEL_NAME", "openai/clip-vit-base-patch32")

    top_k_per_frame = get_config_value("TOP_K_PER_FRAME", 1)
    score_alpha = get_config_value("SCORE_ALPHA", 0.35)
    score_beta = get_config_value("SCORE_BETA", 0.65)
    keep_full_frame_fallback = get_config_value("KEEP_FULL_FRAME_FALLBACK", False)

    max_frames_per_video = get_config_value("MAX_FRAMES_PER_VIDEO", None)
    frame_k = get_config_value("FRAME_K", 1)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    if not images_root.exists():
        raise FileNotFoundError(f"IMAGES_ROOT does not exist: {images_root}")

    prompts = load_prompts()
    videos = read_video_names(
        images_root=images_root,
        single_video=single_video,
        video_list=video_list,
    )

    reranker = CLIPReranker(
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        hf_model_name=clip_hf_model_name,
    )

    all_detections: Dict[str, List[Dict]] = {}

    for video_name in videos:
        video_dir = images_root / video_name
        if not video_dir.exists():
            print(f"[warn] video dir not found, skipping: {video_dir}")
            continue

        print(f"\nProcessing video: {video_name}")
        dets_video = process_video(
            video_name=video_name,
            video_dir=video_dir,
            reranker=reranker,
            prompts=prompts,
            score_alpha=score_alpha,
            score_beta=score_beta,
            diff_threshold=diff_threshold,
            blur_ksize=blur_ksize,
            morph_kernel=morph_kernel,
            morph_iterations=morph_iterations,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            box_expand=box_expand,
            proposal_top_k=proposal_top_k,
            proposal_nms_iou=proposal_nms_iou,
            top_k_per_frame=top_k_per_frame,
            keep_full_frame_fallback=keep_full_frame_fallback,
            max_frames_per_video=max_frames_per_video,
            frame_k=frame_k,
        )
        all_detections.update(dets_video)

    payload = {
        "meta": {
            "method": "v3_motion_clip_rerank_t_to_t_plus_k",
            "images_root": str(images_root),
            "prompts": list(prompts),
            "params": {
                "frame_k": frame_k,
                "diff_threshold": diff_threshold,
                "blur_ksize": blur_ksize,
                "morph_kernel": morph_kernel,
                "morph_iterations": morph_iterations,
                "min_area": min_area,
                "max_area_ratio": max_area_ratio,
                "box_expand": box_expand,
                "proposal_top_k": proposal_top_k,
                "proposal_nms_iou": proposal_nms_iou,
                "top_k_per_frame": top_k_per_frame,
                "score_alpha": score_alpha,
                "score_beta": score_beta,
                "keep_full_frame_fallback": bool(keep_full_frame_fallback),
                "clip_model_name": clip_model_name,
                "clip_pretrained": clip_pretrained,
                "clip_hf_model_name": clip_hf_model_name,
            },
            "videos": videos,
        },
        "detections": all_detections,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote detections JSON to: {output_json}")
    print(f"Total frames with entries: {len(all_detections)}")


if __name__ == "__main__":
    main()
