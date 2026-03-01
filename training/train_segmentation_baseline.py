import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.brats_seg_dataset import BratsSegmentationDataset
from models.unet import UNet


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_DIR = "processed"
MASK_DIR = "processed_masks"

TRAIN_SPLIT = "splits/train.txt"
VAL_SPLIT = "splits/val.txt"

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
DEVICE = "cpu"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------
# Dataset
# -----------------------------
train_dataset = BratsSegmentationDataset(
    IMAGE_DIR, MASK_DIR, TRAIN_SPLIT
)

val_dataset = BratsSegmentationDataset(
    IMAGE_DIR, MASK_DIR, VAL_SPLIT
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")


# -----------------------------
# Model
# -----------------------------
model = UNet(in_channels=1, out_channels=1).to(DEVICE)

bce = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )


optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        loss = bce(outputs, masks) + dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = bce(outputs, masks) + dice_loss(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "segmentation_baseline_best.pth")
        )
        print("✅ Best model saved!")

    print("-" * 50)

print("\n🎉 Baseline Segmentation Training Completed!")