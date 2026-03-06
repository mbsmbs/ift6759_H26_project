import os
import random
import shutil
#Chemins d'accès pour les données source et destination
source_root = r"C:\Users\chikh\Downloads\MoCA\MoCA\JPEGImages"
destination_root = r"C:\Users\chikh\Downloads\MoCA\MoCA\V2"

images_per_folder = 10
SEED = 42 

random.seed(SEED)
os.makedirs(destination_root, exist_ok=True)

for folder_name in os.listdir(source_root):
    source_folder = os.path.join(source_root, folder_name)

    images = [f for f in os.listdir(source_folder)]
    #Prévention du cas où il y a moins d'images que images_per_folder
    k = min(images_per_folder, len(images))
    chosen = random.sample(images, k)

    destination_folder = os.path.join(destination_root, folder_name)
    os.makedirs(destination_folder, exist_ok=True)

    for image in chosen:
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(destination_folder, image)
        shutil.copy2(src_path, dst_path)
