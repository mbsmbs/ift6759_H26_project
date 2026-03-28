from pathlib import Path

IMAGES_ROOT = Path("data/MoCA/JPEGImages")
VIDEO = None
VIDEO_LIST = "splits/dev_videos.txt"
OUTPUT_JSON = Path("outputs/v3/dev_predictions.json")

PROMPTS = [
    "a camouflaged animal",
    "an animal hidden in nature",
    "a hidden animal",
    "a camouflaged creature",
    "a creature hidden in nature",
    "a moving animal",
    "a moving creature",
]

DIFF_THRESHOLD = 25
BLUR_KSIZE = 15
MORPH_KERNEL = 5
MORPH_ITERATIONS = 2
MIN_AREA = 400
MAX_AREA_RATIO = 0.60
BOX_EXPAND = 0.15
PROPOSAL_TOP_K = 10
PROPOSAL_NMS_IOU = 0.5

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_HF_MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K_PER_FRAME = 1
SCORE_ALPHA = 0.35
SCORE_BETA = 0.65
KEEP_FULL_FRAME_FALLBACK = True

MAX_FRAMES_PER_VIDEO = None