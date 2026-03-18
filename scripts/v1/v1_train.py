import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import joblib

# Parametres
DATA = r"C:\Users\chikh\OneDrive\Desktop\MoCA\V1"
BATCH = 32
EPOCHS = 5
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device utilise :", device)

# Transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Chargement du dataset d'entrainement...")
train_ds = datasets.ImageFolder(root=DATA + r"\train", transform=transform_train)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

print("Nombre de classes :", len(train_ds.classes))
print("Classes :", train_ds.classes)
print("Nombre d'images train :", len(train_ds))
print("Nombre de batches train :", len(train_loader))

# Modele
print("Creation du modele ResNet18...")
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model.to(device)

# Optimiseur + loss
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# Entrainement
print("Debut de l'entrainement...")
for epoch in range(EPOCHS):
    model.train()
    running = 0.0

    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

        running += loss.item()

        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(
                f"epoch {epoch+1}/{EPOCHS} | "
                f"batch {batch_idx+1}/{len(train_loader)} | "
                f"loss_batch={loss.item():.4f}"
            )

    print(f"Fin epoch {epoch+1}/{EPOCHS} | loss_moyenne={running/len(train_loader):.4f}")

# Mapping indice -> nom de classe
idx_to_class = {i: c for i, c in enumerate(train_ds.classes)}

# Test sur V1/test avec vote majoritaire par dossier
test_root = DATA + r"\test"
print("\nDebut de la phase de test par vote majoritaire...")
print("Dossier test :", test_root)

model.eval()
folder_results = []
correct_folders = 0
total_folders = 0

with torch.no_grad():
    for true_class_name in sorted(os.listdir(test_root)):
        class_folder = os.path.join(test_root, true_class_name)

        if not os.path.isdir(class_folder):
            continue

        image_files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if len(image_files) == 0:
            print(f"Dossier ignore (vide) : {true_class_name}")
            continue

        print(f"\nTest du dossier : {true_class_name} | nb_images={len(image_files)}")
        predictions = []

        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            image = Image.open(image_path).convert("RGB")
            image = transform_test(image).unsqueeze(0).to(device)

            logits = model(image)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = idx_to_class[pred_idx]
            predictions.append(pred_class)

        majority_pred = Counter(predictions).most_common(1)[0][0]
        is_correct = (majority_pred == true_class_name)

        correct_folders += int(is_correct)
        total_folders += 1

        folder_results.append({
            "true_folder": true_class_name,
            "pred_folder": majority_pred,
            "num_images": len(image_files),
            "correct": is_correct
        })

        print(
            f"Prediction majoritaire : {majority_pred} | "
            f"correct={is_correct}"
        )

print("\nResultats finaux par dossier :")
for result in folder_results:
    print(
        f"dossier={result['true_folder']} | "
        f"prediction_majoritaire={result['pred_folder']} | "
        f"nb_images={result['num_images']} | "
        f"correct={result['correct']}"
    )

folder_accuracy = correct_folders / max(1, total_folders)
print(f"\nAccuracy par vote majoritaire sur les dossiers : {folder_accuracy:.3f}")

# Sauvegarde
print("\nSauvegarde du modele...")
torch.save({
    "state_dict": model.state_dict(),
    "classes": train_ds.classes
}, r"C:\Users\chikh\OneDrive\Desktop\MoCA\resnet_v1_cls.pth")

checkpoint_joblib = {
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "classes": train_ds.classes
}
joblib.dump(checkpoint_joblib, r"C:\Users\chikh\OneDrive\Desktop\MoCA\resnet_v1_cls.joblib")

print("Modele .pth sauvegarde")
print("Modele .joblib sauvegarde")
print("Execution terminee")