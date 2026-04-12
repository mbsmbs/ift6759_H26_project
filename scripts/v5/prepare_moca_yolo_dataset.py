import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_TRAIN_VIDEOS = [
    "arabian_horn_viper",
    "arctic_fox",
    "arctic_fox_1",
    "arctic_fox_2",
]
DEFAULT_VAL_VIDEOS = [
    "arctic_fox_3",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prépare un dataset YOLO (classe unique animal) à partir de MoCA annotations.csv."
    )
    parser.add_argument("--annotations-csv", type=str, default="data/MoCA/Annotations/annotations.csv")
    parser.add_argument("--images-root", type=str, default="data/MoCA/JPEGImages")
    parser.add_argument("--output-root", type=str, default="data/MoCA_YOLO")
    parser.add_argument("--train-videos", nargs="+", default=DEFAULT_TRAIN_VIDEOS)
    parser.add_argument("--val-videos", nargs="+", default=DEFAULT_VAL_VIDEOS)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copie les images (sinon, crée des liens symboliques pour économiser l'espace disque).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Supprime le dossier output-root avant de regénérer.",
    )
    return parser.parse_args()


def load_moca_rows(annotations_csv: Path) -> List[Tuple[str, Dict[str, float]]]:
    rows: List[Tuple[str, Dict[str, float]]] = []
    with annotations_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            frame_key = row[1].lstrip("/")  # ex: arabian_horn_viper/00000.jpg
            spatial = json.loads(row[4])  # [2, x, y, w, h]
            if int(spatial[0]) != 2:
                continue
            x, y, w, h = float(spatial[1]), float(spatial[2]), float(spatial[3]), float(spatial[4])
            rows.append(
                (
                    frame_key,
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    },
                )
            )
    return rows


def to_yolo_norm(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    cx = (x + (w / 2.0)) / float(max(1, img_w))
    cy = (y + (h / 2.0)) / float(max(1, img_h))
    bw = w / float(max(1, img_w))
    bh = h / float(max(1, img_h))
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return cx, cy, bw, bh


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, copy_images: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_yaml(output_root: Path):
    yaml_path = output_root / "moca_yolo.yaml"
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: animal",
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def main():
    args = parse_args()

    annotations_csv = Path(args.annotations_csv)
    images_root = Path(args.images_root)
    output_root = Path(args.output_root)

    if not annotations_csv.exists():
        raise FileNotFoundError(f"annotations.csv not found: {annotations_csv}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"images root not found: {images_root}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    images_train_dir = output_root / "images" / "train"
    images_val_dir = output_root / "images" / "val"
    labels_train_dir = output_root / "labels" / "train"
    labels_val_dir = output_root / "labels" / "val"
    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        ensure_dir(d)

    train_set = set(args.train_videos)
    val_set = set(args.val_videos)
    overlap = train_set.intersection(val_set)
    if overlap:
        raise ValueError(f"Train/Val overlap detected: {sorted(overlap)}")

    rows = load_moca_rows(annotations_csv)

    n_train = 0
    n_val = 0
    n_skipped = 0

    for frame_key, box in rows:
        video = frame_key.split("/", 1)[0]
        src_img = images_root / frame_key
        if not src_img.exists():
            n_skipped += 1
            continue

        split = None
        if video in train_set:
            split = "train"
        elif video in val_set:
            split = "val"
        else:
            n_skipped += 1
            continue

        stem = frame_key.replace("/", "__")
        dst_img = (images_train_dir if split == "train" else images_val_dir) / stem
        dst_lbl = (labels_train_dir if split == "train" else labels_val_dir) / f"{Path(stem).stem}.txt"

        place_image(src_img, dst_img, copy_images=args.copy_images)

        # Lire dimensions sans dépendance PIL: jpg header via pillow serait plus simple.
        # Ici on utilise Pillow pour robustesse.
        from PIL import Image  # import local pour éviter dépendance globale inutile en dehors de ce script

        with Image.open(src_img) as im:
            img_w, img_h = im.size

        cx, cy, bw, bh = to_yolo_norm(
            x=box["x"],
            y=box["y"],
            w=box["w"],
            h=box["h"],
            img_w=img_w,
            img_h=img_h,
        )
        # Classe unique = 0 (animal)
        dst_lbl.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

        if split == "train":
            n_train += 1
        else:
            n_val += 1

    yaml_path = write_yaml(output_root)

    print(f"Prepared YOLO dataset at: {output_root}")
    print(f"YAML: {yaml_path}")
    print(f"train_samples={n_train}")
    print(f"val_samples={n_val}")
    print(f"skipped={n_skipped}")
    print(f"copy_images={args.copy_images}")


if __name__ == "__main__":
    main()
