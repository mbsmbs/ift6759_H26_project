import random
import shutil
import re
from pathlib import Path
import pandas as pd
from PIL import Image

MOCA_ROOT = Path(r"C:\Users\chikh\Downloads\MoCA\MoCA")
V2_DIR = MOCA_ROOT/"V2"
CSV_PATH = MOCA_ROOT/"Annotations"/"annotations.csv"
OUT_DIR = MOCA_ROOT/"V2_YOLO_MULTICLASS"

VAL_RATIO = 0.2

random.seed(42)


def locate_data_start_and_header(csv_path: Path):
    """
    Trouve :
      - le header via '# CSV_HEADER = ...'
      - la première vraie ligne data (ex: 335_D667qyst,...)
    """

    header_list = None
    data_start = None

    data_row_re = re.compile(r"^\s*\"?\d+_[^,]+,")

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            raw = line.strip()
            raw_nq = raw.strip('"').strip("'").strip()

            if "CSV_HEADER" in raw_nq and "=" in raw_nq:
                after = raw_nq.split("=", 1)[1].strip()
                after = after.strip('"').strip("'").strip()
                header_list = [h.strip() for h in after.split(",") if h.strip()]

            if data_start is None and data_row_re.match(raw_nq):
                data_start = i
                break

    return data_start, header_list


def parse_spatial(spatial: str):
    """
    Convertit '[2,x,y,w,h]' -> (x,y,w,h)
    """
    if not isinstance(spatial, str):
        return None

    s = spatial.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None

    s = s[1:-1].strip()
    if not s:
        return None

    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 5:
        return None

    if parts[0] != "2":  # rectangle seulement
        return None

    try:
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        return x, y, w, h
    except ValueError:
        return None


def yolo_line_from_xywh(x, y, bw, bh, W, H, cls_id: int):
    x = max(0.0, min(x, W - 1))
    y = max(0.0, min(y, H - 1))
    bw = max(0.0, min(bw, W))
    bh = max(0.0, min(bh, H))

    xc = x + bw / 2.0
    yc = y + bh / 2.0

    return f"{cls_id} {xc / W:.6f} {yc / H:.6f} {bw / W:.6f} {bh / H:.6f}"


def animal_from_file_list(file_list: str) -> str:
    s = str(file_list).strip().replace("\\", "/")
    s = s.lstrip("/")
    return s.split("/", 1)[0]


data_start, header_list = locate_data_start_and_header(CSV_PATH)

df = pd.read_csv(CSV_PATH, skiprows=data_start, header=None)

if header_list:
    df = df.iloc[:, :len(header_list)]
    df.columns = header_list

needed = {"file_list", "spatial_coordinates"}

df["file_list"] = df["file_list"].astype(str).str.strip()
df["animal"] = df["file_list"].apply(animal_from_file_list)
df["parsed"] = df["spatial_coordinates"].apply(parse_spatial)

df = df[df["parsed"].notna()].copy()

animals = sorted(df["animal"].unique().tolist())
animal_to_id = {a: i for i, a in enumerate(animals)}

groups = {}
for _, r in df.iterrows():
    key = r["file_list"]
    cls_id = animal_to_id[r["animal"]]
    x, y, w, h = r["parsed"]
    groups.setdefault(key, []).append((cls_id, x, y, w, h))


folders = [p for p in V2_DIR.iterdir() if p.is_dir()]
if not folders:
    raise RuntimeError(f"Aucun sous-dossier trouvé dans {V2_DIR}")

random.shuffle(folders)
n_val = max(1, int(len(folders) * VAL_RATIO))
val_folders = set(folders[:n_val])

for split in ["train", "val"]:
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


for folder in folders:
    split = "val" if folder in val_folders else "train"

    for img_path in folder.iterdir():

        key = f"/{folder.name}/{img_path.name}"

        if key not in groups:
            continue

        out_name = f"{folder.name}__{img_path.name}"
        out_img = OUT_DIR / "images" / split / out_name

        with Image.open(img_path) as im:
            W, H = im.size

        lines = []
        for (cls_id, x, y, bw, bh) in groups[key]:
            lines.append(yolo_line_from_xywh(x, y, bw, bh, W, H, cls_id))

        shutil.copy2(img_path, out_img)

        out_lbl = OUT_DIR / "labels" / split / (Path(out_name).stem + ".txt")
        out_lbl.write_text("\n".join(lines), encoding="utf-8")


names_block = "\n".join([f"  {i}: {name}" for i, name in enumerate(animals)])

yaml_text = f"""path: {OUT_DIR}
train: images/train
val: images/val
names:
{names_block}
"""

(OUT_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")
