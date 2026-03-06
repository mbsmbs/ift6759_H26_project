import random, shutil
from pathlib import Path

source = Path(r"C:\Users\chikh\Downloads\MoCA\MoCA\V2")
destination = Path(r"C:\Users\chikh\Downloads\MoCA\MoCA\V2_cls")
ratio = 0.2

random.seed(42)


for split in ["train", "val"]:
    (destination/split).mkdir(parents=True, exist_ok=True)

for cls_dir in [p for p in source.iterdir()]:
    imgs = [p for p in cls_dir.iterdir()]
    if not imgs:
        continue

    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * ratio))
    val_imgs = imgs[:n_val]
    train_imgs = imgs[n_val:]

    (destination/"train"/cls_dir.name).mkdir(parents=True, exist_ok=True)
    (destination/"val"/cls_dir.name).mkdir(parents=True, exist_ok=True)

    for p in train_imgs:
        shutil.copy2(p, destination/"train"/cls_dir.name/p.name)
    for p in val_imgs:
        shutil.copy2(p, destination/"val"/cls_dir.name/p.name)

