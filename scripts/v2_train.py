import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import joblib

# Parametres de base de l'utilisateur
DATA = r"C:\Users\chikh\Downloads\MoCA\MoCA\V2_cls"
BATCH = 32
EPOCHS = 5
LR = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transformations pour les données d'entrainement et de validation
# pour qu'elles soient au format approprié
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Application des transformations et creation des datasets pour train et validation
train_ds = datasets.ImageFolder(root=DATA + r"\train", transform=transform_train)
val_ds   = datasets.ImageFolder(root=DATA + r"\val", transform=transform_val)

# Creation des dataloaders pour train et validation
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# Création du modèle ResNet18 pré-entrainé
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model.to(device)

# Définition de l'optimiseur et de la fonction de perte (entropie croisée)
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_finale = nn.CrossEntropyLoss()

# Fonction pour évaluer l'accuracy sur le set de validation
def eval_accuracy():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return correct / max(1, total)

# Entrainement du modèle
for epoch in range(EPOCHS):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_finale(logits, y)
        loss.backward()
        optim.step()

    print(f"epoch {epoch+1}/{EPOCHS} | loss={loss.item():.4f} | val_acc={eval_accuracy():.3f}")

# Sauvegarde du modèle entrainé en format PyTorch
torch.save({
    "state_dict": model.state_dict(),
    "classes": train_ds.classes
}, r"C:\Users\chikh\Downloads\MoCA\MoCA\resnet_v2_cls.pth")

# Sauvegarde du modèle en format joblib
checkpoint_joblib = {
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "classes": train_ds.classes}

joblib.dump(checkpoint_joblib, r"C:\Users\chikh\OneDrive\Desktop\MoCA\resnet_v2_cls.joblib")