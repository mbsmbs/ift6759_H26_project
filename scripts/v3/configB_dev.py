from pathlib import Path

IMAGES_ROOT = Path("data/MoCA/JPEGImages")
VIDEO = None
VIDEO_LIST = "splits/dev_videos.txt"
OUTPUT_JSON = Path("outputs/v3/dev_predictions_B.json")

PROMPTS = [
    "a camouflaged animal blending into the background",
    "an animal blending with leaves, grass or rocks",
    "a barely visible animal in natural surroundings",
    "a hidden animal in the image",
    "a small animal moving in the scene",
    "subtle motion of an animal in nature",
    "natural scene without animals"
]

DIFF_THRESHOLD = 25
BLUR_KSIZE = 9
MORPH_KERNEL = 5
MORPH_ITERATIONS = 2
MIN_AREA = 400
MAX_AREA_RATIO = 0.25
BOX_EXPAND = 0.08
PROPOSAL_TOP_K = 10
PROPOSAL_NMS_IOU = 0.5

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_HF_MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K_PER_FRAME = 1
SCORE_ALPHA = 0.20
SCORE_BETA = 0.80
KEEP_FULL_FRAME_FALLBACK = False

MAX_FRAMES_PER_VIDEO = None