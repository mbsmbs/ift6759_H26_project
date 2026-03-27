# V4 Pipeline (Baseline vs Fine-tuning)

## Fichiers de code V4

- `scripts/v4/owlvit_infer.py`
- `scripts/v4/finetune_detector.py`
- `scripts/v4/evaluate_v4.py`
- `scripts/v4/V4_FINAL_CONFIG.json`

## 5 videos MoCA

- `arabian_horn_viper`
- `arctic_fox`
- `arctic_fox_1`
- `arctic_fox_2`
- `arctic_fox_3`

## Workflow minimum

### 1) Baseline (zero-shot)

```bash
for v in arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3; do
  python scripts/v4/owlvit_infer.py \
    --video "$v" \
    --threshold 0.10 \
    --top-k-per-frame 1 \
    --prompts "a camouflaged animal" "an animal hidden in background" \
    --output-json "outputs/owlvit/final/baseline_dets_${v}.json"
done
```

### 2) Fine-tuning

```bash
python scripts/v4/finetune_detector.py \
  --output-dir outputs/owlvit/finetune_detector_main \
  --prompts "a camouflaged animal" "an animal hidden in background" \
  --train-videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 \
  --val-videos arctic_fox_3 \
  --epochs 2 \
  --batch-size 2 \
  --lr 2e-6 \
  --loss-cls-weight 1.0 \
  --loss-box-weight 2.0 \
  --loss-iou-weight 1.0 \
  --freeze-text \
  --best-metric f1
```

### 3) Inference fine-tuned + calibration threshold

```bash
for v in arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3; do
  python scripts/v4/owlvit_infer.py \
    --video "$v" \
    --threshold 0.01 \
    --top-k-per-frame 1 \
    --checkpoint outputs/owlvit/finetune_detector_main/best_model.pt \
    --prompts "a camouflaged animal" "an animal hidden in background" \
    --output-json "outputs/owlvit/final/ft_dets_${v}.json"
done
```

### 4) Evaluation (un seul script)

Fusion des detections (5 videos) puis evaluation:

```bash
python - << 'PY'
import json
from pathlib import Path
videos = ["arabian_horn_viper","arctic_fox","arctic_fox_1","arctic_fox_2","arctic_fox_3"]
merged = {"meta": {}, "detections": {}}
for i, v in enumerate(videos):
    p = Path(f"outputs/owlvit/final/ft_dets_{v}.json")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if i == 0:
        merged["meta"] = obj.get("meta", {})
    merged["detections"].update(obj.get("detections", {}))
out = Path("outputs/owlvit/final/ft_dets_5videos.json")
with out.open("w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)
print(out)
PY

python scripts/v4/evaluate_v4.py batch \
  --dets-json outputs/owlvit/final/ft_dets_5videos.json \
  --videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3 \
  --iou-threshold 0.5 \
  --score-threshold 0.08 \
  --max-det-per-frame 1 \
  --out-csv outputs/owlvit/final/ft_eval_moca_batch_5videos_s008.csv
```

### 5) Sweep de calibration (meme script)

```bash
python scripts/v4/evaluate_v4.py sweep \
  --dets-json outputs/owlvit/final/ft_dets_5videos.json \
  --videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3 \
  --iou-thresholds 0.5 \
  --score-thresholds 0.01,0.03,0.05,0.08,0.10 \
  --max-det-per-frame-list 1 \
  --out-grid-csv outputs/owlvit/final/ft_eval_sweep_grid.csv \
  --out-best-csv outputs/owlvit/final/ft_eval_sweep_best.csv
```

### 6) Comparaison baseline vs fine-tuned (meme script)

```bash
python scripts/v4/evaluate_v4.py master \
  --videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3 \
  --iou-threshold 0.5 \
  --max-det-per-frame 1 \
  --version "baseline|outputs/owlvit/final/dets_top1_refined_5videos.json|0.10|zero-shot refined" \
  --version "finetuned|outputs/owlvit/final/ft_f1sel_dets_top1_5videos.json|0.08|best-metric=f1 + threshold optimized" \
  --out-master-csv outputs/owlvit/final/v4_eval_master.csv
```
