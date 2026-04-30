import os
import re
import random
import shutil
from collections import defaultdict

source_root = r"IFT6759_H26_PROJECT\data\MoCA\JPEGImages"
destination_root = r"IFT6759_H26_PROJECT\scripts\v1"

images_per_folder_train = 30
images_per_test_video = 1
SEED = 42
random.seed(SEED)

train_root = os.path.join(destination_root, "train")
test_root = os.path.join(destination_root, "test")

os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

pattern = re.compile(r"^(.*?)(?:_(\d+))?$")
animal_groups = defaultdict(list)

for folder_name in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    match = pattern.match(folder_name)
    if not match:
        continue

    animal_name = match.group(1)
    idx = int(match.group(2)) if match.group(2) is not None else 0
    animal_groups[animal_name].append((idx, folder_name))

for animal_name, folders in animal_groups.items():
    if len(folders) < 2:
        print(f"Ignoré : {animal_name}")
        continue

    folders.sort(key=lambda x: x[0])

    _, test_folder_name = folders[-1]
    train_folders = folders[:-1]

    train_class_folder = os.path.join(train_root, animal_name)
    test_class_folder = os.path.join(test_root, animal_name)

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    # TRAIN
    for _, train_folder_name in train_folders:
        source_folder = os.path.join(source_root, train_folder_name)

        images = [
            f for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f))
        ]

        k = min(images_per_folder_train, len(images))
        chosen_images = random.sample(images, k)

        for image in chosen_images:
            src_path = os.path.join(source_folder, image)
            dst_name = f"{train_folder_name}__{image}"
            dst_path = os.path.join(train_class_folder, dst_name)
            shutil.copy2(src_path, dst_path)

    source_test_folder = os.path.join(source_root, test_folder_name)

    test_images = [
        f for f in os.listdir(source_test_folder)
        if os.path.isfile(os.path.join(source_test_folder, f))
    ]

    if len(test_images) == 0:
        print(f"Aucune image dans le dossier test pour {animal_name}")
        continue

    chosen_test_images = random.sample(test_images, min(images_per_test_video, len(test_images)))

    for image in chosen_test_images:
        src_path = os.path.join(source_test_folder, image)
        dst_name = f"{test_folder_name}__{image}"
        dst_path = os.path.join(test_class_folder, dst_name)
        shutil.copy2(src_path, dst_path)

    print(f"{animal_name} -> train: {len(train_folders)} vidéos, test: 1 image depuis {test_folder_name}")

print("Création terminée.")