# V5 Pipeline (YOLO zero-shot-like)

## Fichiers de code V5

- `scripts/v5/yolo_zeroshot_infer.py`
- `scripts/v4/evaluate_v4.py` (réutilisé pour l'évaluation)

## 5 vidéos MoCA

- `arabian_horn_viper`
- `arctic_fox`
- `arctic_fox_1`
- `arctic_fox_2`
- `arctic_fox_3`

## Pré-requis

- Modèle YOLOv8n disponible à:
  - `outputs/yolo/models/yolo_zeroshot_pretrained_coco/yolov8n.pt`
- Données MoCA:
  - `data/MoCA/JPEGImages/...`
  - `data/MoCA/Annotations/annotations.csv`

## Workflow minimum

### 1) Inference YOLO zero-shot-like (animal-only)

```bash
python scripts/v5/yolo_zeroshot_infer.py \
  --input-root data/MoCA/JPEGImages \
  --videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3 \
  --model outputs/yolo/models/yolo_zeroshot_pretrained_coco/yolov8n.pt \
  --animal-only \
  --conf 0.001 \
  --nms-iou 0.7 \
  --imgsz 640 \
  --top-k-per-frame 1 \
  --output-json outputs/yolo/final/yolo_zeroshot_coco_animalonly_dets_top1_5videos.json
```

### 2) Sweep de calibration (score threshold)

```bash
python scripts/v4/evaluate_v4.py sweep \
  --dets-json outputs/yolo/final/yolo_zeroshot_coco_animalonly_dets_top1_5videos.json \
  --videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3 \
  --iou-thresholds 0.5 \
  --score-thresholds 0.00,0.01,0.03,0.05,0.08,0.10 \
  --max-det-per-frame-list 1 \
  --out-grid-csv outputs/yolo/final/yolo_zeroshot_coco_animalonly_eval_sweep_grid_5videos.csv \
  --out-best-csv outputs/yolo/final/yolo_zeroshot_coco_animalonly_eval_sweep_best_5videos.csv
```

### 3) Comparaison V5 vs V4 (master)

```bash
python scripts/v4/evaluate_v4.py master \
  --videos arabian_horn_viper arctic_fox arctic_fox_1 arctic_fox_2 arctic_fox_3 \
  --iou-threshold 0.5 \
  --max-det-per-frame 1 \
  --version "v4_zeroshot|outputs/owlvit/final/dets_top1_refined_5videos.json|0.10|OWL-ViT zero-shot refined" \
  --version "v4_finetuned|outputs/owlvit/final/ft_f1sel_dets_top1_5videos.json|0.08|OWL-ViT fine-tuned" \
  --version "v5_yolo_zeroshot|outputs/yolo/final/yolo_zeroshot_coco_animalonly_dets_top1_5videos.json|0.01|YOLOv8n pretrained (animal-only)" \
  --out-master-csv outputs/yolo/final/yolo_zeroshot_vs_v4_master_5videos.csv
```

## Sorties principales

- Détections V5:
  - `outputs/yolo/final/yolo_zeroshot_coco_animalonly_dets_top1_5videos.json`
- Sweep:
  - `outputs/yolo/final/yolo_zeroshot_coco_animalonly_eval_sweep_grid_5videos.csv`
  - `outputs/yolo/final/yolo_zeroshot_coco_animalonly_eval_sweep_best_5videos.csv`
- Comparaison V4/V5:
  - `outputs/yolo/final/yolo_zeroshot_vs_v4_master_5videos.csv`
