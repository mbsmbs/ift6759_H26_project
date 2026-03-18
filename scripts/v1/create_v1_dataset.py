import os
import random
import shutil

# Chemins d'accès pour les données source et destination
source_root = r"C:\Users\chikh\OneDrive\Desktop\MoCA\JPEGImages"
destination_root = r"C:\Users\chikh\OneDrive\Desktop\MoCA\V1"

images_per_folder = 30
train_ratio = 29/30
SEED = 42

random.seed(SEED)

train_root = os.path.join(destination_root, "train")
test_root = os.path.join(destination_root, "test")

os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

for folder_name in os.listdir(source_root):
    source_folder = os.path.join(source_root, folder_name)

    if not os.path.isdir(source_folder):
        continue

    images = [
        f for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f))
    ]

    # Prévention du cas où il y a moins d'images que images_per_folder
    k = min(images_per_folder, len(images))
    chosen = random.sample(images, k)

    # Mélange puis split train/test
    random.shuffle(chosen)
    split_idx = int(train_ratio * len(chosen))

    train_images = chosen[:split_idx]
    test_images = chosen[split_idx:]

    train_class_folder = os.path.join(train_root, folder_name)
    test_class_folder = os.path.join(test_root, folder_name)

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    for image in train_images:
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(train_class_folder, image)
        shutil.copy2(src_path, dst_path)

    for image in test_images:
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(test_class_folder, image)
        shutil.copy2(src_path, dst_path)

print("Création du dataset train/test terminée.")