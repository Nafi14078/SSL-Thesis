import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from datasets.brats_dataset import BratsSSLDataset
from models.unet import UNet

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "processed"
MAX_SAMPLES = 10000
BATCH_SIZE = 2
EPOCHS = 8
LR = 1e-4
VAL_SPLIT = 0.2

device = "cpu"

# ----------------------------
# DATASET
# ----------------------------
dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    task="inpainting",
    max_samples=MAX_SAMPLES
)
print("Total dataset size:", len(dataset))

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
print("Train size:", train_size)
print("Val size:", val_size)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# MODEL
# ----------------------------
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float("inf")

# ----------------------------
# TRAIN
# ----------------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for masked, clean in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        masked = masked.to(device)
        clean = clean.to(device)

        output = model(masked)
        loss = criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for masked, clean in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            masked = masked.to(device)
            clean = clean.to(device)

            output = model(masked)
            loss = criterion(output, clean)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/inpainting_best.pth")
        print("✅ Best model saved.\n")