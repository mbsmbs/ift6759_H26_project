# IFT6759 H26 Project

Détection d'animaux camouflés dans MoCA avec plusieurs familles d'approches:

- `V1/V2`: CNN (classification / vote majoritaire)
- `V3`: hybride mouvement + CLIP
- `V4`: OWL-ViT (zero-shot + fine-tuning)
- `V5`: YOLOv8n zero-shot-like (baseline de transférabilité)

## 1) Structure du dépôt

```text
ift6759_H26_project/
├── data/                          # Données locales (ignorées par git)
│   └── MoCA/
│       ├── Annotations/annotations.csv
│       └── JPEGImages/<video>/<frame>.jpg
├── outputs/
│   ├── owlvit/
│   │   ├── final/                 # Résultats finaux V4 (json/csv/figures)
│   │   └── models/                # Checkpoints fine-tuning V4
│   └── yolo/
│       ├── final/                 # Résultats finaux V5 (json/csv)
│       └── models/                # Modèle YOLO préentraîné utilisé
├── scripts/
│   ├── v1/                        # Scripts V1 (CNN)
│   ├── v2/                        # Scripts V2 (CNN temporel / vote)
│   ├── v3/                        # Scripts V3 (mouvement + CLIP)
│   ├── v4/                        # Scripts V4 (OWL-ViT)
│   └── v5/                        # Scripts V5 (YOLO baseline)
├── soumissions/                   # Rapports LaTeX/PDF (local, non versionné)
└── README.md
```

## 2) Versions Python et dépendances

Version recommandée:

- Python `3.11` (testé aussi avec `3.12` localement)

Installation rapide:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision transformers ultralytics pandas pillow matplotlib
```

Notes:

- Sur Apple Silicon, PyTorch peut tourner avec `mps`.
- Sur CPU, les fine-tunings sont lents.

## 3) Données (MoCA)

Le projet attend:

- `data/MoCA/Annotations/annotations.csv`
- `data/MoCA/JPEGImages/...`

Le dossier `data/` est ignoré par git pour éviter de pousser le dataset.

## 4) Scripts principaux (résumé simple)

### V4 (`scripts/v4`)

- `owlvit_infer.py`: inférence OWL-ViT et export JSON de détections.
- `finetune_detector.py`: fine-tuning OWL-ViT (avec `train_loss` et `val_loss` loggés).
- `evaluate_v4.py`: évaluation/sweep/master comparison (AP@0.5, précision, rappel, F1).
- `README_V4.md`: guide d'exécution V4.

### V5 (`scripts/v5`)

- `yolo_zeroshot_infer.py`: inférence YOLOv8n préentraîné (option `--animal-only`) et export JSON compatible V4.
- `README_V5.md`: guide d'exécution V5.

### V1/V2/V3

Les scripts de ces versions sont dans `scripts/v1`, `scripts/v2`, `scripts/v3` et suivent les pipelines développés par les responsables respectifs.

## 5) Reproduction minimale (V4 + V5)

### V4: OWL-ViT zero-shot/fine-tuning

Voir:

- `scripts/v4/README_V4.md`

### V5: YOLO zero-shot-like

Voir:

- `scripts/v5/README_V5.md`

## 6) Fichiers de sorties utiles

Exemples de sorties finales:

- V4:
  - `outputs/owlvit/final/dets_top1_refined_5videos.json`
  - `outputs/owlvit/final/ft_f1sel_dets_top1_5videos.json`
  - `outputs/owlvit/final/figures/v4_finetune_loss_curve.png`
- V5:
  - `outputs/yolo/final/yolo_zeroshot_coco_animalonly_dets_top1_5videos.json`
  - `outputs/yolo/final/yolo_zeroshot_vs_v4_master_5videos.csv`

## 7) Métriques

Les évaluations utilisent:

- IoU (`Intersection over Union`) avec seuil standard `0.5`
- `AP@0.5`
- Précision
- Rappel
- F1

Les scores sont reportés en macro (moyenne par vidéo).

## 8) Références principales

- MoCA dataset:  
  Li et al., *Moving Camouflaged Object Detection*, CVPR 2021.  
  https://arxiv.org/abs/2105.03248

- CLIP:  
  Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.  
  https://arxiv.org/abs/2103.00020

- OWL-ViT:  
  Minderer et al., *Simple Open-Vocabulary Object Detection with Vision Transformers*, ECCV 2022.  
  https://arxiv.org/abs/2205.06230

- YOLOv8 (Ultralytics docs):  
  https://docs.ultralytics.com
