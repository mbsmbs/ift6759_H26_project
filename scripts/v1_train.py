import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import joblib

# Parametres de base de l'utilisateur
DATA = r"C:\Users\chikh\Downloads\MoCA\MoCA\V1"
BATCH = 16
EPOCHS = 5
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# Transformations pour les données d'entrainement
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# Application des transformations et creation du dataset
dataset = datasets.ImageFolder(root=DATA, transform=transform_train)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

# Création du modèle ResNet18 pré-entrainé
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.to(device)

# Définition de l'optimiseur et de la fonction de perte
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# Entrainement du modèle
model.train()

for epoch in range(EPOCHS):
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        running += loss.item()

    print(f"epoch {epoch+1}/{EPOCHS} | loss={running/len(loader):.4f}")

# Sauvegarde du modèle entrainé en format PyTorch
torch.save({
    "state_dict": model.state_dict(),
    "classes": dataset.classes
}, r"C:\Users\chikh\Downloads\MoCA\MoCA\resnet_v1_cls.pth")

# Sauvegarde du modèle en format joblib
checkpoint_joblib = {
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "classes": dataset.classes
}
joblib.dump(checkpoint_joblib, r"C:\Users\chikh\OneDrive\Desktop\MoCA\resnet_v1_cls.joblib")