import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.brats_dataset_pretraining_updated import BratsSSLDataset
from models.unet import UNet


# ======================================================
# CONFIGURATION  (unchanged)
# ======================================================

DATA_DIR    = "processed"
TRAIN_SPLIT = "splits/train.txt"
VAL_SPLIT   = "splits/val.txt"

BATCH_SIZE     = 8
EPOCHS         = 20
LEARNING_RATE  = 1e-3

DEVICE = torch.device("cpu")


# ======================================================
# EDGE-WEIGHTED MSE LOSS
# Standard MSE is dominated by the large flat background.
# This loss adds extra weight on high-gradient (boundary)
# regions so the model is penalised harder for errors at
# tumor edges — the exact regions TANS corrupts most.
# ======================================================

class EdgeWeightedMSELoss(nn.Module):
    """
    Loss = MSE_base  +  lambda_edge * MSE_edge_weighted

    edge_weight_map = 1 + alpha * |gradient(target)|
    so boundary pixels contribute more to the loss.
    """
    def __init__(self, alpha=4.0, lambda_edge=0.5):
        super().__init__()
        self.alpha       = alpha        # edge amplification factor
        self.lambda_edge = lambda_edge  # edge loss weight

    def _gradient_magnitude(self, t):
        """Simple finite-difference gradient magnitude on a BCHW tensor."""
        gx = torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1])   # (B,C,H,W-1)
        gy = torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :])   # (B,C,H-1,W)

        # Pad to match original size
        gx = torch.nn.functional.pad(gx, (0, 1))            # (B,C,H,W)
        gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))      # (B,C,H,W)

        return gx + gy                                        # (B,C,H,W)

    def forward(self, pred, target):
        # Base MSE
        mse = torch.mean((pred - target) ** 2)

        # Edge weight map from clean target
        with torch.no_grad():
            grad_mag    = self._gradient_magnitude(target)
            edge_weight = 1.0 + self.alpha * grad_mag        # ≥ 1 everywhere

        # Edge-weighted MSE
        weighted_mse = torch.mean(edge_weight * (pred - target) ** 2)

        return mse + self.lambda_edge * weighted_mse


# ======================================================
# DATASET & DATALOADER
# Datasets are recreated each epoch inside the loop so
# the curriculum scale (epoch/total) updates correctly.
# ======================================================

def make_loaders(epoch):
    train_ds = BratsSSLDataset(
        data_dir       = DATA_DIR,
        file_list_path = TRAIN_SPLIT,
        task           = "tans",
        epoch          = epoch,
        total_epochs   = EPOCHS
    )
    val_ds = BratsSSLDataset(
        data_dir       = DATA_DIR,
        file_list_path = VAL_SPLIT,
        task           = "tans",
        epoch          = epoch,
        total_epochs   = EPOCHS
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0
    )
    return train_loader, val_loader


# ======================================================
# MODEL
# ======================================================

model = UNet(in_channels=1, out_channels=1)
model.to(DEVICE)

criterion = EdgeWeightedMSELoss(alpha=4.0, lambda_edge=0.5)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ======================================================
# TRAINING LOOP
# ======================================================

train_losses = []
val_losses   = []
best_val_loss = float("inf")

os.makedirs("checkpoints", exist_ok=True)

print("🚀 Starting TANS v2 Denoising Training...\n")

for epoch in range(EPOCHS):

    # Recreate loaders so dataset sees correct epoch for curriculum
    train_loader, val_loader = make_loaders(epoch)

    curriculum_pct = int(((epoch / max(EPOCHS - 1, 1)) * 0.6 + 0.4) * 100)
    print(f"\n[Epoch {epoch+1}/{EPOCHS}]  Curriculum corruption scale: {curriculum_pct}%")

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    running_train_loss = 0.0

    for noisy, clean in tqdm(train_loader, desc=f"  Train"):

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(noisy)
        loss    = criterion(outputs, clean)

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
        for noisy, clean in tqdm(val_loader, desc=f"  Val  "):

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            outputs = model(noisy)
            loss    = criterion(outputs, clean)

            running_val_loss += loss.item() * noisy.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"  Train Loss : {epoch_train_loss:.6f}")
    print(f"  Val   Loss : {epoch_val_loss:.6f}")

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "checkpoints/best_denoising_model_tans.pth")
        print("  ✅ Best model saved!")

    print("-" * 50)


# ======================================================
# PLOT LOSS CURVES
# ======================================================

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Edge-Weighted MSE Loss")
plt.title("TANS v2 Denoising Training Curve")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints/denoising_loss_curve_tans.png")
plt.show()

print("\n🎉 TANS v2 Training Completed Successfully!")
