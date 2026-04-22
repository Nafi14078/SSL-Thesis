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

IMAGE_DIR      = "processed_ped_10k/images"
MASK_DIR       = "processed_ped_10k/masks"
TRAIN_SPLIT    = "train_ped.txt"
VAL_SPLIT      = "val_ped.txt"

PRETRAINED_PATH = "checkpoints/best_star_model.pth"   # ← STAR pretrained weights

BATCH_SIZE     = 4      # matches denoising/inpainting baseline
EPOCHS         = 20     # matches denoising/inpainting baseline
LR             = 1e-4   # matches denoising/inpainting baseline

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------
# Dataset
# -----------------------------

train_dataset = BratsSegmentationDataset(IMAGE_DIR, MASK_DIR, TRAIN_SPLIT)
val_dataset   = BratsSegmentationDataset(IMAGE_DIR, MASK_DIR, VAL_SPLIT)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")
print(f"Using device:  {DEVICE}")


# -----------------------------
# Model — Load STAR Weights
# -----------------------------

model = UNet(in_channels=1, out_channels=1).to(DEVICE)

print("🔄 Loading pretrained STAR weights...")
model.load_state_dict(
    torch.load(PRETRAINED_PATH, map_location=DEVICE),
    strict=False    # same as your denoising fine-tuning script
)
print("✅ Pretrained STAR weights loaded!")


# -----------------------------
# Freeze encoder for first 3 epochs
# then unfreeze for full fine-tuning
# (extra step vs denoising — protects pretrained features)
# -----------------------------

def freeze_encoder(model):
    for name, param in model.named_parameters():
        if name.startswith("enc") or name.startswith("bottleneck"):
            param.requires_grad = False
    print("🔒 Encoder frozen — warming up decoder only.")

def unfreeze_encoder(model):
    for param in model.parameters():
        param.requires_grad = True
    print("🔓 Encoder unfrozen — full fine-tuning active.")


# -----------------------------
# Loss Functions
# (identical to your denoising fine-tuning script)
# -----------------------------

bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1e-6):
    pred         = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

optimizer    = torch.optim.Adam(model.parameters(), lr=LR)
best_val_loss = float("inf")

FREEZE_EPOCHS = 3   # freeze encoder for first 3 epochs


# -----------------------------
# Training Loop
# -----------------------------

for epoch in range(EPOCHS):

    # ── Freeze / unfreeze schedule ─────────────────────────────────────────
    if epoch == 0:
        freeze_encoder(model)
    if epoch == FREEZE_EPOCHS:
        unfreeze_encoder(model)

    # -------- TRAIN --------
    model.train()
    train_loss = 0

    frozen_status = "frozen" if epoch < FREEZE_EPOCHS else "unfrozen"

    for images, masks in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} [{frozen_status}]"
    ):
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        outputs = model(images)
        loss    = bce(outputs, masks) + dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            outputs  = model(images)
            loss     = bce(outputs, masks) + dice_loss(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # -------- LOG --------
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] [{frozen_status}]")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")

    # -------- SAVE BEST --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "ped_from_star_best.pth")
        )
        print("✅ Best model saved!")

    print("-" * 50)

print("\n🎉 Fine-tuning from STAR on BraTS-PED Completed!")
