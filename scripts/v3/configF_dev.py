from pathlib import Path

IMAGES_ROOT = Path("data/MoCA/JPEGImages")
VIDEO = None
VIDEO_LIST = "splits/dev_videos.txt"
OUTPUT_JSON = Path("outputs/v3/dev_predictions_E.json")

PROMPTS = [
    "a camouflaged animal",
    "an animal hidden in nature",
    "a hidden animal",
    "a moving animal",
]

DIFF_THRESHOLD = 12 
BLUR_KSIZE = 7       
MORPH_KERNEL = 5
MORPH_ITERATIONS = 2
MIN_AREA = 150       
MAX_AREA_RATIO = 0.40
BOX_EXPAND = 0.05    

PROPOSAL_TOP_K = 15
PROPOSAL_NMS_IOU = 0.4

CLIP_MODEL_NAME = "ViT-L-14" 
CLIP_PRETRAINED = "openai"
CLIP_HF_MODEL_NAME = "openai/clip-vit-base-patch16"

TOP_K_PER_FRAME = 1

SCORE_ALPHA = 0.30  # Motion weight
SCORE_BETA = 0.70   # CLIP weight

KEEP_FULL_FRAME_FALLBACK = False 

MAX_FRAMES_PER_VIDEO = None

FRAME_K = 6  # or whatever you want

