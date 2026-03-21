import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.brats_dataset_pretraining import BratsSSLDataset
from models.unet import UNet


# ======================================================
# CONFIGURATION
# ======================================================

DATA_DIR = "processed"
TRAIN_SPLIT = "splits/train.txt"
VAL_SPLIT = "splits/val.txt"

BATCH_SIZE = 8          # Safe for CPU
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")  # Force CPU


# ======================================================
# DATASET & DATALOADER
# ======================================================

train_dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    file_list_path=TRAIN_SPLIT,
    task="denoising"
)

val_dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    file_list_path=VAL_SPLIT,
    task="denoising"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # IMPORTANT (deterministic)
    num_workers=0           # CPU safe
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ======================================================
# MODEL
# ======================================================

model = UNet(in_channels=1, out_channels=1)
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ======================================================
# TRAINING LOOP
# ======================================================

train_losses = []
val_losses = []

best_val_loss = float("inf")

os.makedirs("checkpoints", exist_ok=True)

print("🚀 Starting Denoising Training on CPU...\n")

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    running_train_loss = 0.0

    for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(noisy)
        loss = criterion(outputs, clean)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * noisy.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for noisy, clean in val_loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            outputs = model(noisy)
            loss = criterion(outputs, clean)

            running_val_loss += loss.item() * noisy.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {epoch_train_loss:.6f}")
    print(f"Val Loss:   {epoch_val_loss:.6f}")

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "checkpoints/best_denoising_model.pth")
        print("✅ Best model saved!")

    print("-" * 50)


# ======================================================
# SAVE FINAL MODEL
# ======================================================

torch.save(model.state_dict(), "checkpoints/final_denoising_model.pth")


# ======================================================
# PLOT LOSS CURVES
# ======================================================

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Denoising Training Curve")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints/denoising_loss_curve.png")
plt.show()

print("\n🎉 Training Completed Successfully!")