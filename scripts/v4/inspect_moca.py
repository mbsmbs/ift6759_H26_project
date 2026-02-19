import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


ROOT = Path("data/MoCA")
CSV_PATH = ROOT / "Annotations" / "annotations.csv"
IMG_ROOT = ROOT / "JPEGImages"


def parse_spatial_coordinates(s: str):
    """
    VIA format: "[2,x,y,w,h]" where 2=rectangle
    Returns (x, y, w, h) as floats.
    """
    s = s.strip().strip('"')
    arr = json.loads(s)  # e.g. [2, 482.87, 225.391, 507.13, 191.739]
    assert int(arr[0]) == 2, f"Expected rectangle shape_id=2, got {arr[0]}"
    return float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])


def parse_motion(metadata_str: str):
    """
    metadata: {"1":"0"} where attribute id 1 = motion
    options: 0=locomotion, 1=subtle_motion, 2=still
    """
    metadata_str = metadata_str.strip().strip('"')
    md = json.loads(metadata_str)
    motion_id = int(md["1"])
    motion_map = {0: "locomotion", 1: "subtle_motion", 2: "still"}
    return motion_id, motion_map.get(motion_id, "unknown")


def main(nth_row: int = 1):
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        # skip comment/header lines starting with '#'
        data_rows = [row for row in reader if row and not row[0].startswith("#")]

    row = data_rows[nth_row]
    # columns: metadata_id, file_list, flags, temporal_coordinates, spatial_coordinates, metadata, ...
    file_list = row[1]                 # "/arabian_horn_viper/00000.jpg"
    spatial = row[4]                   # "[2,x,y,w,h]"
    metadata = row[5]                  # '{"1":"0"}'

    img_path = IMG_ROOT / file_list.lstrip("/")
    x, y, w, h = parse_spatial_coordinates(spatial)
    motion_id, motion_name = parse_motion(metadata)

    img = Image.open(img_path).convert("RGB")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"{img_path.name} | motion={motion_name} ({motion_id})")
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main(nth_row=0)  # change to 1,2,3... to inspect other rows