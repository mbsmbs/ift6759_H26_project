import os
import random
import shutil

#Chemins d'accès pour les données source et destination
source_root = r"C:\Users\chikh\Downloads\MoCA\MoCA\JPEGImages"
destination_root = r"C:\Users\chikh\Downloads\MoCA\MoCA\V1"


os.makedirs(destination_root, exist_ok=True)

for folder_name in os.listdir(source_root):

    source_folder = os.path.join(source_root, folder_name)
    images = [f for f in os.listdir(source_folder)]
    random_image = random.choice(images)
    dst_folder = os.path.join(destination_root, folder_name)
    os.makedirs(dst_folder, exist_ok=True)
    src_image_path = os.path.join(source_folder, random_image)
    dst_image_path = os.path.join(dst_folder, random_image)
    shutil.copy2(src_image_path, dst_image_path)

 