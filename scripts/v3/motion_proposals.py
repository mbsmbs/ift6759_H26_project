from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Proposal:
    x1: float
    y1: float
    x2: float
    y2: float
    motion_score: float
    area: float


def clip_box_to_image(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def iou_xyxy(a: Proposal, b: Proposal) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def nms_proposals(proposals: List[Proposal], iou_threshold: float = 0.5) -> List[Proposal]:
    if not proposals:
        return []

    proposals = sorted(proposals, key=lambda p: p.motion_score, reverse=True)
    kept: List[Proposal] = []

    for prop in proposals:
        should_keep = True
        for prev in kept:
            if iou_xyxy(prop, prev) >= iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(prop)
    return kept


def preprocess_gray(image_bgr: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def build_motion_mask(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    diff_threshold: int = 25,
    blur_ksize: int = 5,
    morph_kernel: int = 5,
    morph_iterations: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:

    prev_gray = preprocess_gray(prev_bgr, blur_ksize=blur_ksize)
    curr_gray = preprocess_gray(curr_bgr, blur_ksize=blur_ksize)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    if morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=morph_iterations)

    return diff, mask


def mask_to_proposals(
    mask: np.ndarray,
    diff_map: np.ndarray,
    min_area: int = 400,
    max_area_ratio: float = 0.60,
    box_expand: float = 0.15,
    top_k: int = 10,
    nms_iou: float = 0.5,
) -> List[Proposal]:

    height, width = mask.shape[:2]
    image_area = float(height * width)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    proposals: List[Proposal] = []

    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        box_area = float(w * h)
        if box_area / image_area > max_area_ratio:
            continue

        pad_x = int(round(w * box_expand))
        pad_y = int(round(h * box_expand))

        x1 = x - pad_x
        y1 = y - pad_y
        x2 = x + w + pad_x
        y2 = y + h + pad_y
        x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, width=width, height=height)

        component_mask = (labels == label_idx)
        motion_pixels = diff_map[component_mask]
        motion_score = float(np.mean(motion_pixels)) if motion_pixels.size > 0 else 0.0

        proposals.append(
            Proposal(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                motion_score=motion_score,
                area=float((x2 - x1) * (y2 - y1)),
            )
        )

    proposals = nms_proposals(proposals, iou_threshold=nms_iou)
    proposals = sorted(proposals, key=lambda p: p.motion_score, reverse=True)[:top_k]
    return proposals


def generate_motion_proposals(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    diff_threshold: int = 25,
    blur_ksize: int = 5,
    morph_kernel: int = 5,
    morph_iterations: int = 2,
    min_area: int = 400,
    max_area_ratio: float = 0.60,
    box_expand: float = 0.15,
    top_k: int = 10,
    nms_iou: float = 0.5,
) -> Tuple[List[Proposal], np.ndarray]:

    diff_map, mask = build_motion_mask(
        prev_bgr=prev_bgr,
        curr_bgr=curr_bgr,
        diff_threshold=diff_threshold,
        blur_ksize=blur_ksize,
        morph_kernel=morph_kernel,
        morph_iterations=morph_iterations,
    )

    proposals = mask_to_proposals(
        mask=mask,
        diff_map=diff_map,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        box_expand=box_expand,
        top_k=top_k,
        nms_iou=nms_iou,
    )
    return proposals, mask


def add_full_frame_fallback(
    proposals: List[Proposal],
    image_shape: Tuple[int, int, int],
    fallback_score: float = 1.0,
) -> List[Proposal]:

    if proposals:
        return proposals

    height, width = image_shape[:2]
    full = Proposal(
        x1=0.0,
        y1=0.0,
        x2=float(width - 1),
        y2=float(height - 1),
        motion_score=float(fallback_score),
        area=float(width * height),
    )
    return [full]