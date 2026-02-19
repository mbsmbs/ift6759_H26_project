from dataclasses import dataclass, field
from typing import Dict, List


def iou_xyxy(a: Dict[str, float], b: Dict[str, float]) -> float:
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


@dataclass
class Track:
    track_id: int
    frame_keys: List[str] = field(default_factory=list)
    detections: List[Dict[str, float]] = field(default_factory=list)
    last_frame_index: int = -1

    def add(self, frame_key: str, det: Dict[str, float], frame_index: int) -> None:
        self.frame_keys.append(frame_key)
        self.detections.append(det)
        self.last_frame_index = frame_index

    @property
    def last_det(self) -> Dict[str, float]:
        return self.detections[-1]


def build_tracks(
    detections_by_frame: Dict[str, List[Dict[str, float]]],
    iou_threshold: float = 0.3,
    max_gap: int = 1,
    score_threshold: float = 0.0,
    min_track_len: int = 1,
) -> List[Track]:
    frame_keys = sorted(detections_by_frame.keys())
    active_tracks: List[Track] = []
    all_tracks: List[Track] = []
    next_track_id = 0

    for frame_index, frame_key in enumerate(frame_keys):
        current = [
            d
            for d in detections_by_frame.get(frame_key, [])
            if float(d.get("score", 0.0)) >= score_threshold
        ]

        # Keep only tracks still eligible for matching.
        active_tracks = [t for t in active_tracks if frame_index - t.last_frame_index <= max_gap]

        matched_track_ids = set()
        unmatched_det_idx = set(range(len(current)))
        candidates = []

        for ti, tr in enumerate(active_tracks):
            for di, det in enumerate(current):
                overlap = iou_xyxy(tr.last_det, det)
                if overlap >= iou_threshold:
                    candidates.append((overlap, ti, di))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for _, ti, di in candidates:
            if di not in unmatched_det_idx:
                continue
            tr = active_tracks[ti]
            if tr.track_id in matched_track_ids:
                continue
            tr.add(frame_key, current[di], frame_index)
            matched_track_ids.add(tr.track_id)
            unmatched_det_idx.remove(di)

        for di in sorted(unmatched_det_idx):
            tr = Track(track_id=next_track_id)
            tr.add(frame_key, current[di], frame_index)
            active_tracks.append(tr)
            all_tracks.append(tr)
            next_track_id += 1

    return [tr for tr in all_tracks if len(tr.detections) >= min_track_len]


def aggregate_track(track: Track, agg: str = "max", window: int = 5) -> Dict:
    if window > 0:
        score_slice = track.detections[-window:]
    else:
        score_slice = track.detections

    scores = [float(d["score"]) for d in score_slice]
    if agg == "mean":
        score_agg = sum(scores) / max(len(scores), 1)
    else:
        score_agg = max(scores) if scores else 0.0

    best = max(track.detections, key=lambda d: float(d["score"]))
    return {
        "track_id": track.track_id,
        "length": len(track.detections),
        "start_frame": track.frame_keys[0],
        "end_frame": track.frame_keys[-1],
        "score_agg": float(score_agg),
        "bbox": {
            "x1": float(best["x1"]),
            "y1": float(best["y1"]),
            "x2": float(best["x2"]),
            "y2": float(best["y2"]),
        },
        "class_id": int(best.get("class_id", 0)),
        "detections": [
            {
                "frame": fk,
                "x1": float(det["x1"]),
                "y1": float(det["y1"]),
                "x2": float(det["x2"]),
                "y2": float(det["y2"]),
                "score": float(det["score"]),
                "prompt_id": int(det.get("prompt_id", -1)),
                "class_id": int(det.get("class_id", 0)),
            }
            for fk, det in zip(track.frame_keys, track.detections)
        ],
    }


def tracks_to_jsonable(tracks: List[Track], agg: str = "max", window: int = 5) -> List[Dict]:
    return [aggregate_track(track=t, agg=agg, window=window) for t in tracks]
