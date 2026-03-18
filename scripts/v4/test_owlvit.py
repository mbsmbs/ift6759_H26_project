import os

# Force l'usage PyTorch-only pour éviter les imports TF/Keras.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Script de test minimal: une image, quelques prompts, affichage des boîtes.
# Image MoCA de démonstration.
image_path = "data/MoCA/JPEGImages/arabian_horn_viper/00000.jpg"
image = Image.open(image_path).convert("RGB")

# Prompts textuels (composante sémantique du VLM).
texts = [["a camouflaged animal", "a snake", "an animal hidden in sand"]]

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Conversion des sorties du modèle vers des boîtes à l'échelle de l'image.
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs=outputs,
    threshold=0.15,
    target_sizes=target_sizes
)[0]

# Visualisation qualitative des détections.
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.imshow(image)

for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
    x1, y1, x2, y2 = box.tolist()
    rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor="blue", facecolor="none")
    ax.add_patch(rect)
    ax.text(x1, y1, f"{texts[0][label]}: {score:.2f}", color="blue")

ax.axis("off")
plt.show()
