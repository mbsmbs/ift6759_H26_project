import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize V4 detections and tracks.")
    parser.add_argument("--dets-json", type=str, default="outputs/owlvit/dets_top1.json")
    parser.add_argument("--tracks-json", type=str, default="outputs/owlvit/tracks_top1.json")
    parser.add_argument("--video", type=str, default=None, help="Optional video filter.")
    return parser.parse_args()


def load_json(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_detections(payload, video=None):
    detections = payload.get("detections", payload)
    frame_keys = sorted(detections.keys())
    if video:
        frame_keys = [k for k in frame_keys if k.startswith(f"{video}/")]

    total_frames = len(frame_keys)
    frames_with_det = 0
    total_dets = 0
    scores = []

    for k in frame_keys:
        dets = detections.get(k, [])
        if dets:
            frames_with_det += 1
        total_dets += len(dets)
        scores.extend(float(d.get("score", 0.0)) for d in dets)

    det_frame_ratio = (frames_with_det / total_frames) if total_frames > 0 else 0.0
    avg_dets_per_frame = (total_dets / total_frames) if total_frames > 0 else 0.0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    return {
        "total_frames": total_frames,
        "frames_with_detection": frames_with_det,
        "detection_frame_ratio": det_frame_ratio,
        "total_detections": total_dets,
        "avg_detections_per_frame": avg_dets_per_frame,
        "avg_detection_score": avg_score,
    }


def summarize_tracks(payload, video=None):
    tracks = payload.get("tracks", [])
    if video:
        prefix = f"{video}/"
        filtered = []
        for t in tracks:
            dets = [d for d in t.get("detections", []) if d.get("frame", "").startswith(prefix)]
            if dets:
                t2 = dict(t)
                t2["detections"] = dets
                t2["length"] = len(dets)
                t2["start_frame"] = dets[0]["frame"]
                t2["end_frame"] = dets[-1]["frame"]
                filtered.append(t2)
        tracks = filtered

    num_tracks = len(tracks)
    lengths = [int(t.get("length", len(t.get("detections", [])))) for t in tracks]
    scores = [float(t.get("score_agg", 0.0)) for t in tracks]

    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
    max_len = max(lengths) if lengths else 0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    return {
        "num_tracks": num_tracks,
        "avg_track_length": avg_len,
        "max_track_length": max_len,
        "avg_track_score": avg_score,
    }


def main():
    args = parse_args()
    dets_payload = load_json(args.dets_json)
    tracks_payload = load_json(args.tracks_json)

    dsum = summarize_detections(dets_payload, video=args.video)
    tsum = summarize_tracks(tracks_payload, video=args.video)

    video_name = args.video if args.video else "ALL"
    print(f"V4 Summary [{video_name}]")
    print("Detections:")
    print(f"  total_frames: {dsum['total_frames']}")
    print(f"  frames_with_detection: {dsum['frames_with_detection']}")
    print(f"  detection_frame_ratio: {dsum['detection_frame_ratio']:.3f}")
    print(f"  total_detections: {dsum['total_detections']}")
    print(f"  avg_detections_per_frame: {dsum['avg_detections_per_frame']:.3f}")
    print(f"  avg_detection_score: {dsum['avg_detection_score']:.3f}")
    print("Tracks:")
    print(f"  num_tracks: {tsum['num_tracks']}")
    print(f"  avg_track_length: {tsum['avg_track_length']:.3f}")
    print(f"  max_track_length: {tsum['max_track_length']}")
    print(f"  avg_track_score: {tsum['avg_track_score']:.3f}")


if __name__ == "__main__":
    main()
