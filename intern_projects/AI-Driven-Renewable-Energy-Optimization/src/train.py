# src/train.py

import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === CONFIG ===
DATA_DIR = '../data/solar_defects_dataset'
BATCH_SIZE = 32
NUM_EPOCHS = 50
MODEL_SAVE_PATH = '../saved_models/resnet18_classifier.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === TRANSFORMS ===
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # ImageNet stats
])

# === DATA LOADER ===
dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
class_names = dataset.classes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === LOAD MODEL WITH WEIGHTS ===
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# === UNFREEZE ALL LAYERS ===
for param in model.parameters():
    param.requires_grad = True

# === REPLACE FINAL CLASSIFIER HEAD ===
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# === LOSS, OPTIMIZER, SCHEDULER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

# === TRAINING LOOP ===
print("Training on:", DEVICE)
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()

    train_accuracy = correct / len(train_ds)

    # === VALIDATION ===
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()

    val_accuracy = val_correct / len(val_ds)

    # === Step the LR scheduler based on validation accuracy ===
    scheduler.step(val_accuracy)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {total_loss:.4f} | "
          f"Train Acc: {train_accuracy * 100:.2f}% | Val Acc: {val_accuracy * 100:.2f}%")

# === FINAL EVALUATION ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# === SAVE MODEL ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")
