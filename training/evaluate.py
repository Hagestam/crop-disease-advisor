"""
Evaluates the trained model on the test set and prints a full classification report.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CHECKPOINT_PATH = Path("models/checkpoints/best_efficientnet_b0.pth")
DATA_DIR        = Path("data/raw/PlantVillage")
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_CLASSES     = 38

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
class_names = ckpt["class_names"]

# Rebuild model
model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, NUM_CLASSES),
)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# Build test set (use last 10% of data)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
n = len(dataset)
torch.manual_seed(42)
indices = torch.randperm(n).tolist()
test_indices = indices[int(n * 0.90):]
test_set = Subset(dataset, test_indices)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Run evaluation
all_preds, all_labels = [], []
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()

top1_acc = correct / len(test_set)
print(f"\nTop-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
print("\nPer-class Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

# Confusion matrix (top 15 classes by frequency for readability)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=[c[:20] for c in class_names],
            yticklabels=[c[:20] for c in class_names])
plt.title(f"Confusion Matrix — Top-1 Acc: {top1_acc:.4f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
print("Saved confusion matrix to models/confusion_matrix.png")