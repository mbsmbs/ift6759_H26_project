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

### V1 (`scripts/v1`)

- `create_v1_dataset.py`: construit un dataset `train/test` (échantillonnage d'images par classe).
- `v1_train.py`: entraîne un ResNet18 et évalue par vote majoritaire au niveau dossier.

### V2 (`scripts/v2`)

- `create_v2_dataset.py`: construit un dataset `train/test` avec un split différent de V1.
- `v2_train.py`: entraîne un ResNet18 et applique aussi le vote majoritaire.

### V3 (`scripts/v3`)

- `main_run_v3.py`: pipeline principal mouvement + CLIP (génération de propositions, reranking CLIP, sortie JSON).
- `configA_dev.py ... configF_dev.py`: configurations expérimentales V3.
- `motion_proposals.py`: génération de régions candidates par mouvement.
- `clip_rerank.py`: score sémantique CLIP et reranking.
- `eval_moca_detection.py`, `eval_moca_batch.py`: évaluation AP/F1 sur MoCA.

### V4 (`scripts/v4`)

- `owlvit_infer.py`: inférence OWL-ViT et export JSON de détections.
- `finetune_detector.py`: fine-tuning OWL-ViT (avec `train_loss` et `val_loss` loggés).
- `evaluate_v4.py`: évaluation/sweep/master comparison (AP@0.5, précision, rappel, F1).
- `README_V4.md`: guide d'exécution V4.

### V5 (`scripts/v5`)

- `yolo_zeroshot_infer.py`: inférence YOLOv8n préentraîné (option `--animal-only`) et export JSON compatible V4.
- `README_V5.md`: guide d'exécution V5.

## 5) Reproduction minimale (V1 à V5)

### V1: dataset + entraînement

```bash
python scripts/v1/create_v1_dataset.py
python scripts/v1/v1_train.py
```

Important: les scripts V1 actuels contiennent des chemins Windows codés en dur (ex: `C:\\Users\\...`).  
Adapter `source_root`, `destination_root` et `DATA` avant exécution sur votre machine.

### V2: dataset + entraînement

```bash
python scripts/v2/create_v2_dataset.py
python scripts/v2/v2_train.py
```

Même remarque que V1: adapter les chemins locaux dans les scripts.

### V3: exécution + évaluation

```bash
# 1) Choisir une config (ex: E) dans main_run_v3.py
#    (import configE_dev as config)
python scripts/v3/main_run_v3.py

# 2) Évaluer le JSON produit
python scripts/v3/eval_moca_batch.py \
  --dets-json outputs/v3/dev_predictions_E.json \
  --annotations-csv data/MoCA/Annotations/annotations.csv \
  --iou-threshold 0.5 \
  --score-threshold 0.2 \
  --max-det-per-frame 1 \
  --output-csv outputs/v3/eval_v3_dev_E.csv
```

### V4: OWL-ViT zero-shot/fine-tuning

Voir:

- `scripts/v4/README_V4.md`

### V5: YOLO zero-shot-like

Voir:

- `scripts/v5/README_V5.md`

## 6) Fichiers de sorties utiles

Exemples de sorties finales (V1 à V5):

- V1:
  - `resnet_v1_cls.pth`
  - `resnet_v1_cls.joblib`
- V2:
  - `resnet_v2_cls.pth`
  - `resnet_v2_cls.joblib`
- V3:
  - `outputs/v3/dev_predictions_*.json` (selon config A..F)
  - `outputs/v3/eval_v3_*.csv` (si export batch activé)

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

## 9) Papiers locaux (`papers/`)

- `papers/2011.11630v1.pdf`
- `papers/2103.00020v1.pdf`
- `papers/Cheng_Implicit_Motion_Handling_for_Video_Camouflaged_Object_Detection_CVPR_2022_paper.pdf`
