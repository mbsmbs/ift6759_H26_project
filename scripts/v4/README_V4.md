# V4 Pipeline (Reproducible)

This folder contains the V4 implementation:
- OWL-ViT frame-level inference
- temporal refinement
- IoU tracking
- MoCA evaluation
- visualizations

## Final frozen config
Use `scripts/v4/V4_FINAL_CONFIG.json`.

## 5-video set
- `arabian_horn_viper`
- `arctic_fox`
- `arctic_fox_1`
- `arctic_fox_2`
- `arctic_fox_3`

## Run

### 1) Inference (top-5 candidates per frame)
```bash
for v in arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3; do
  python scripts/v4/owlvit_infer.py \
    --video "$v" \
    --threshold 0.10 \
    --top-k-per-frame 5 \
    --output-json "outputs/owlvit/final/dets_top5_${v}.json"
done
```

### 2) Temporal refinement (1 detection/frame)
```bash
for v in arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3; do
  python scripts/v4/refine_dets_temporal.py \
    --input-dets-json "outputs/owlvit/final/dets_top5_${v}.json" \
    --output-dets-json "outputs/owlvit/final/dets_top1_refined_${v}.json" \
    --video "$v" \
    --top-k 5 \
    --iou-weight 0.35 \
    --area-penalty-lambda 0.4 \
    --min-score 0.0
done
```

### 3) Merge refined detections
```bash
python - << 'PY'
import json
files = [
 'outputs/owlvit/final/dets_top1_refined_full.json',
 'outputs/owlvit/final/dets_top1_refined_arctic_fox.json',
 'outputs/owlvit/final/dets_top1_refined_arctic_fox_1.json',
 'outputs/owlvit/final/dets_top1_refined_arctic_fox_2.json',
 'outputs/owlvit/final/dets_top1_refined_arctic_fox_3.json',
]
out = 'outputs/owlvit/final/dets_top1_refined_5videos.json'
merged = {'meta': {'sources': files}, 'detections': {}}
for p in files:
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    merged['detections'].update(data.get('detections', data))
with open(out, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2)
print('wrote', out, 'frames', len(merged['detections']))
PY
```

### 4) Tracking
```bash
python scripts/v4/run_v4_temporal.py \
  --dets-json outputs/owlvit/final/dets_top1_refined_5videos.json \
  --output-json outputs/owlvit/final/tracks_top1_refined_5videos.json \
  --iou-threshold 0.3 \
  --score-threshold 0.0 \
  --max-gap 1 \
  --min-track-len 1 \
  --agg max \
  --window 5
```

### 5) Evaluation
```bash
python scripts/v4/eval_moca_batch.py \
  --dets-json outputs/owlvit/final/dets_top1_refined_5videos.json \
  --videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3 \
  --iou-threshold 0.5 \
  --score-threshold 0.2 \
  --max-det-per-frame 1 \
  --output-csv outputs/owlvit/final/eval_moca_batch_5videos.csv
```

### 6) Visualizations
```bash
for v in arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3; do
  python scripts/v4/visualize_dets.py \
    --dets-json outputs/owlvit/final/dets_top1_refined_5videos.json \
    --video "$v" \
    --output-dir outputs/owlvit/final/vis_dets_5videos \
    --max-frames 100000 \
    --min-score 0.0

  python scripts/v4/visualize_tracks.py \
    --tracks-json outputs/owlvit/final/tracks_top1_refined_5videos.json \
    --video "$v" \
    --output-dir outputs/owlvit/final/vis_tracks_5videos \
    --max-frames 100000 \
    --min-score 0.0
done
```

## Final artifacts
- `outputs/owlvit/final/dets_top1_refined_5videos.json`
- `outputs/owlvit/final/tracks_top1_refined_5videos.json`
- `outputs/owlvit/final/eval_moca_batch_5videos.csv`
- `outputs/owlvit/final/eval_moca_batch_5videos_report.csv`
- `outputs/owlvit/final/vis_dets_5videos/`
- `outputs/owlvit/final/vis_tracks_5videos/`
