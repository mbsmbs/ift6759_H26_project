import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Force le mode PyTorch-only pour éviter les imports TF/Keras.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from transformers import OwlViTForObjectDetection, OwlViTProcessor

from evaluate_v4 import evaluate_one as eval_det

# Sample = 1 image + its path + its bounding box
@dataclass
class Sample:
    frame_key: str
    image_path: Path
    gt_xyxy: Dict[str, float]

class MoCADetectionDataset(Dataset):
    # Dataset simple: une image + une boîte GT par frame.
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return {
            "frame_key": s.frame_key,
            "image": image,
            "gt_xyxy": s.gt_xyxy,
        }


def identity_collate(batch):
    # On garde la liste brute d'échantillons (PIL + dict), traitée ensuite par collate_train.
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning OWL-ViT detection sur MoCA (version minimale).")
    parser.add_argument("--annotations-csv", type=str, default="data/MoCA/Annotations/annotations.csv")
    parser.add_argument("--images-root", type=str, default="data/MoCA/JPEGImages")
    parser.add_argument("--model-name", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--output-dir", type=str, default="outputs/owlvit/finetune_detector")
    parser.add_argument("--prompts", nargs="+", default=["a camouflaged animal"])
    parser.add_argument("--train-videos", nargs="+", default=["arabian_horn_viper", "arctic_fox", "arctic_fox_1", "arctic_fox_2"])
    parser.add_argument("--val-videos", nargs="+", default=["arctic_fox_3"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--val-threshold", type=float, default=0.1, help="Seuil score utilisé en validation AP/F1.")
    parser.add_argument("--val-max-samples", type=int, default=200, help="Limite optionnelle de frames val pour aller plus vite.")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--loss-cls-weight", type=float, default=1.0)
    parser.add_argument("--loss-box-weight", type=float, default=5.0)
    parser.add_argument("--loss-iou-weight", type=float, default=2.0)
    parser.add_argument(
        "--best-metric",
        type=str,
        default="ap50",
        choices=["ap50", "f1", "mix"],
        help="Critère de sélection du meilleur checkpoint.",
    )
    parser.add_argument(
        "--best-mix-alpha",
        type=float,
        default=0.5,
        help="Poids AP50 dans le critère mix: score = alpha*AP50 + (1-alpha)*F1.",
    )
    return parser.parse_args()

# For reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Reads MoCA csv -> extracts + video +bounding box -> returns a list of annotations
def load_moca_rows(annotations_csv: Path) -> List[dict]:
    rows = []
    with annotations_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            file_key = row[1].lstrip("/")
            spatial = json.loads(row[4])  # [2, x, y, w, h]
            if int(spatial[0]) != 2:
                continue
            x, y, w, h = float(spatial[1]), float(spatial[2]), float(spatial[3]), float(spatial[4])
            rows.append(
                {
                    "frame_key": file_key,
                    "video": file_key.split("/", 1)[0],
                    "gt_xyxy": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                }
            )
    return rows

# Filter rows -> load valid images -> create samples -> return sorted dataset
def build_samples(rows: List[dict], images_root: Path, videos: List[str]) -> List[Sample]:
    keep = set(videos)
    out = []
    for r in rows:
        if r["video"] not in keep:
            continue
        image_path = images_root / r["frame_key"]
        if not image_path.exists():
            continue
        out.append(Sample(frame_key=r["frame_key"], image_path=image_path, gt_xyxy=r["gt_xyxy"]))
    return sorted(out, key=lambda s: s.frame_key)

# Converts box from (corners) -> (center + size) and scales it to [0, 1]
def xyxy_to_cxcywh_norm(box: Dict[str, float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cx = ((x1 + x2) / 2.0) / max(1.0, float(w))
    cy = ((y1 + y2) / 2.0) / max(1.0, float(h))
    bw = (x2 - x1) / max(1.0, float(w))
    bh = (y2 - y1) / max(1.0, float(h))
    # Clamp de sécurité dans [0,1].
    return [
        float(max(0.0, min(1.0, cx))),
        float(max(0.0, min(1.0, cy))),
        float(max(0.0, min(1.0, bw))),
        float(max(0.0, min(1.0, bh))),
    ]

# Transforms a batch -> model inputs + normalized boxes for training
def collate_train(batch, processor: OwlViTProcessor, prompts: List[str], device: torch.device):
    images = [b["image"] for b in batch]
    inputs = processor(text=[prompts] * len(images), images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gt_boxes = []
    for b in batch:
        w, h = b["image"].size
        box_norm = xyxy_to_cxcywh_norm(b["gt_xyxy"], w=w, h=h)
        gt_boxes.append(box_norm)
    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device)  # [B,4] cxcywh norm
    return inputs, gt_boxes

# Converts box from (center format) -> (corner format)
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    # Convertit des boîtes normalisées cxcywh -> xyxy.
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

# Measures how well each predicted box overlaps with the ground truth box
# 1.0 -> perfect match, 0.5 -> decent match, 0.0 -> no overlap 
def box_iou_one_to_many(gt_xyxy: torch.Tensor, pred_xyxy: torch.Tensor) -> torch.Tensor:
    # gt_xyxy: [B,4], pred_xyxy: [B,Q,4] -> IoU [B,Q]
    gx1, gy1, gx2, gy2 = gt_xyxy[:, 0:1], gt_xyxy[:, 1:2], gt_xyxy[:, 2:3], gt_xyxy[:, 3:4]
    px1, py1, px2, py2 = pred_xyxy[..., 0], pred_xyxy[..., 1], pred_xyxy[..., 2], pred_xyxy[..., 3]

    ix1 = torch.maximum(gx1, px1)
    iy1 = torch.maximum(gy1, py1)
    ix2 = torch.minimum(gx2, px2)
    iy2 = torch.minimum(gy2, py2)

    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih

    ga = torch.clamp(gx2 - gx1, min=0.0) * torch.clamp(gy2 - gy1, min=0.0)
    pa = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    union = ga + pa - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return iou


def compute_train_loss(outputs, gt_boxes_cxcywh: torch.Tensor, cls_w: float, box_w: float, iou_w: float):
    # Loss personnalisée pour OWL-ViT sans support natif des labels.
    # - on assigne la GT à la requête ayant le meilleur IoU
    # - cls: BCE sur logits (requête assignée = positive, autres = négatives)
    # - box: L1 sur la boîte assignée
    # - iou: 1 - IoU(assignée, GT)

    # 1. Get predicted boxes and logits
    pred_boxes = outputs.pred_boxes  # [B,Q,4] cxcywh norm
    logits = outputs.logits.squeeze(-1)  # [B,Q] pour 1 prompt

    # 2. Convert boxes to xyxy format
    gt_xyxy = cxcywh_to_xyxy(gt_boxes_cxcywh)  # [B,4]
    pred_xyxy = cxcywh_to_xyxy(pred_boxes)  # [B,Q,4]
    # 3. Compute IoU between GT and all predictions boxes
    ious = box_iou_one_to_many(gt_xyxy, pred_xyxy)  # [B,Q]
    best_idx = ious.argmax(dim=1)  # [B]

    # 4. Select the best predicted box and its IoU
    b_idx = torch.arange(pred_boxes.size(0), device=pred_boxes.device)
    best_boxes = pred_boxes[b_idx, best_idx]  # [B,4]
    best_ious = ious[b_idx, best_idx]  # [B]

    # 5. Build classification targets
    targets = torch.zeros_like(logits)
    targets[b_idx, best_idx] = 1.0
    # 6. Classification loss
    cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    # 7. Box regression loss
    box_loss = torch.nn.functional.l1_loss(best_boxes, gt_boxes_cxcywh)
    # 8. IoU loss
    iou_loss = 1.0 - best_ious.mean()

    # 9. Total loss for backprop
    total = cls_w * cls_loss + box_w * box_loss + iou_w * iou_loss
    # 10. Logging values
    parts = {
        "loss_total": float(total.item()),
        "loss_cls": float(cls_loss.item()),
        "loss_box": float(box_loss.item()),
        "loss_iou": float(iou_loss.item()),
        "mean_best_iou": float(best_ious.mean().item()),
    }
    return total, parts


@torch.no_grad()
def validate_model(
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    val_samples: List[Sample],
    prompts: List[str],
    device: torch.device,
    score_threshold: float,
    max_samples: int,
    cls_w: float,
    box_w: float,
    iou_w: float,
) -> Dict[str, float]:
    # Validation en mode "detection": AP/F1 via les métriques déjà utilisées dans le projet.
    model.eval()
    # Sekect validation subset
    subset = val_samples[:max_samples] if max_samples is not None and max_samples > 0 else val_samples

    dets_by_frame = {}
    gt_by_frame = {}
    val_losses = []
    for s in subset:
        # 1. load image
        image = Image.open(s.image_path).convert("RGB")
        # 2. Store ground truth
        gt_by_frame[s.frame_key] = s.gt_xyxy

        # 3. Prepare model inputs
        inputs = processor(text=[prompts], images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # 4. Run model
        outputs = model(**inputs)

        # 4b. Compute validation loss with the same loss definition as training
        w, h = image.size
        gt_box_norm = torch.tensor(
            [xyxy_to_cxcywh_norm(s.gt_xyxy, w=w, h=h)],
            dtype=torch.float32,
            device=device,
        )  # [1,4]
        val_loss, _ = compute_train_loss(
            outputs=outputs,
            gt_boxes_cxcywh=gt_box_norm,
            cls_w=cls_w,
            box_w=box_w,
            iou_w=iou_w,
        )
        val_losses.append(float(val_loss.item()))

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        # 5. Convert predictions into real detection boxes
        results = processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]

        frame_dets = []
        # 6. Build detection list
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.tolist()
            frame_dets.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(score.item()),
                    "prompt_id": int(label.item()),
                    "class_id": 0,
                }
            )
        # 7. Keep only the best detection
        dets_by_frame[s.frame_key] = sorted(frame_dets, key=lambda d: d["score"], reverse=True)[:1]

    # 8. Final evaluation
    res = eval_det(
        gt_by_frame=gt_by_frame,
        dets_by_frame=dets_by_frame,
        iou_thr=0.5,
        score_thr=score_threshold,
    )
    # 9. Metrics
    return {
        "val_loss": float(mean(val_losses)) if val_losses else 0.0,
        "val_ap50": float(res["ap50"]),
        "val_precision": float(res["op_precision"]),
        "val_recall": float(res["op_recall"]),
        "val_f1": float(res["op_f1"]),
        "val_num_frames": int(res["num_gt_frames"]),
    }

# Controls which parts of the model learn during training
def maybe_freeze(model: OwlViTForObjectDetection, freeze_text: bool, freeze_vision: bool):
    if not freeze_text and not freeze_vision:
        return
    for name, p in model.named_parameters():
        if freeze_text and ("text_model" in name):
            p.requires_grad = False
        if freeze_vision and ("vision_model" in name):
            p.requires_grad = False

# Counts total parameters and how many are trainable
def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    # Parse les arguments de CLI
    args = parse_args()
    set_seed(args.seed)

    # Preparer inputs et outputs
    annotations_csv = Path(args.annotations_csv)
    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chager modele OWL-ViT + processor -> device
    device = select_device(args.device)
    processor = OwlViTProcessor.from_pretrained(args.model_name)
    model = OwlViTForObjectDetection.from_pretrained(args.model_name)
    maybe_freeze(model, freeze_text=args.freeze_text, freeze_vision=args.freeze_vision)
    model.to(device)

    # Afficher le nombre de paramètres total et entrainable & les videos utilisees pour le train/val
    total_params, trainable_params = count_params(model)
    print(f"device={device}")
    print(f"params total={total_params:,} trainable={trainable_params:,}")
    print(f"train_videos={args.train_videos}")
    print(f"val_videos={args.val_videos}")

    # Charger les annotations MoCA -> filtrer par video -> creer des echantillons -> dataset trie
    rows = load_moca_rows(annotations_csv)
    train_samples = build_samples(rows, images_root=images_root, videos=args.train_videos)
    val_samples = build_samples(rows, images_root=images_root, videos=args.val_videos)
    print(f"train_samples={len(train_samples)} val_samples={len(val_samples)}")
    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError("train/val vide. Vérifier les chemins et la liste de vidéos.")

    # DataLoader avec collate personnalise pour preparer les batches d'entraînement
    ds_train = MoCADetectionDataset(train_samples)
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )

    # Optimizer sur les parametres entrainables du modele
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    # Boucle d'entraînement + validation + sauvegarde des checkpoints
    history = []
    best_score = -1.0
    best_ckpt = out_dir / "best_model.pt"
    best_epoch = -1

    # Loop d'entraînement sur les epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        batch_iou = []
        
        for batch in dl_train:
            inputs, gt_boxes = collate_train(batch, processor=processor, prompts=args.prompts, device=device)
            outputs = model(**inputs)
            loss, parts = compute_train_loss(
                outputs=outputs,
                gt_boxes_cxcywh=gt_boxes,
                cls_w=args.loss_cls_weight,
                box_w=args.loss_box_weight,
                iou_w=args.loss_iou_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))
            batch_iou.append(parts["mean_best_iou"])

        train_loss = mean(batch_losses) if batch_losses else 0.0
        train_mean_iou = mean(batch_iou) if batch_iou else 0.0
        val_metrics = validate_model(
            model=model,
            processor=processor,
            val_samples=val_samples,
            prompts=args.prompts,
            device=device,
            score_threshold=args.val_threshold,
            max_samples=args.val_max_samples,
            cls_w=args.loss_cls_weight,
            box_w=args.loss_box_weight,
            iou_w=args.loss_iou_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mean_best_iou": train_mean_iou,
            **val_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_iou={train_mean_iou:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_ap50={row['val_ap50']:.4f} val_f1={row['val_f1']:.4f} "
            f"val_p={row['val_precision']:.4f} val_r={row['val_recall']:.4f}"
        )

        if args.best_metric == "ap50":
            score = row["val_ap50"]
        elif args.best_metric == "f1":
            score = row["val_f1"]
        else:
            alpha = float(args.best_mix_alpha)
            score = alpha * row["val_ap50"] + (1.0 - alpha) * row["val_f1"]

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "prompts": args.prompts,
                    "args": vars(args),
                    "best_epoch": epoch,
                    "best_score": best_score,
                    "best_metric": args.best_metric,
                },
                best_ckpt,
            )

        if args.save_every_epoch:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                out_dir / f"checkpoint_epoch_{epoch:02d}.pt",
            )

    # Recapitulatif final + sauvegarde du resume de l'entraînement
    summary = {
        "model_name": args.model_name,
        "prompts": args.prompts,
        "device": str(device),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_score": best_score,
        "best_metric": args.best_metric,
        "best_epoch": best_epoch,
        "history": history,
        "args": vars(args),
    }
    summary_path = out_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {best_ckpt}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
