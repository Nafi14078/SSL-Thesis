import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from models.unet import UNet


# ======================================================
# CONFIGURATION
# ======================================================

DATA_DIR    = "processed"
TRAIN_SPLIT = "splits/train.txt"
VAL_SPLIT   = "splits/val.txt"

BATCH_SIZE    = 8       # matches denoising/inpainting baseline
EPOCHS        = 20      # matches denoising/inpainting baseline
LEARNING_RATE = 1e-3    # matches denoising/inpainting baseline

DEVICE = torch.device("cpu")   # Force CPU


# ======================================================
# STAR CORRUPTION STRATEGIES
# ======================================================

class STARCorruption:
    """
    Three novel FLAIR-specific corruption strategies.

    Stage 1 (epochs  1- 6): LSHC only
    Stage 2 (epochs  7-13): LSHC or GBE   (random per sample)
    Stage 3 (epochs 14-20): LSHC or GBE or FDSC (random per sample)
    """

    # --------------------------------------------------
    # 1. LSHC — Lesion-Simulating Hyperintensity Corruption
    # Teaches: what FLAIR tumour intensity looks like
    # --------------------------------------------------
    @staticmethod
    def lshc(img, num_blobs=3, min_radius=5, max_radius=20):
        H, W      = img.shape
        corrupted = img.copy()

        flat     = img.flatten()
        top_vals = flat[flat > np.percentile(flat, 90)]
        if len(top_vals) == 0:
            top_vals = np.array([0.8], dtype=np.float32)

        for _ in range(num_blobs):
            cx = np.random.randint(max_radius, max(H - max_radius, max_radius + 1))
            cy = np.random.randint(max_radius, max(W - max_radius, max_radius + 1))
            rx = np.random.randint(min_radius, max_radius + 1)
            ry = np.random.randint(min_radius, max_radius + 1)

            yy, xx  = np.ogrid[:H, :W]
            ellipse = ((yy - cx)**2 / rx**2 + (xx - cy)**2 / ry**2) <= 1.0

            blob_intensity       = float(np.random.choice(top_vals))
            blob                 = np.zeros((H, W), dtype=np.float32)
            blob[ellipse]        = blob_intensity
            blob                 = gaussian_filter(blob, sigma=max(rx, ry) / 2.5)
            corrupted            = np.clip(corrupted + blob, 0, 1)

        return corrupted.astype(np.float32), img.astype(np.float32)

    # --------------------------------------------------
    # 2. GBE — Gradient-Boundary Erosion
    # Teaches: precise tumour boundary delineation
    # --------------------------------------------------
    @staticmethod
    def gbe(img, alpha=0.85, top_gradient_pct=20, neighborhood_size=3):
        gy       = np.gradient(img, axis=0)
        gx       = np.gradient(img, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)

        threshold     = np.percentile(grad_mag, 100 - top_gradient_pct)
        boundary_mask = grad_mag >= threshold
        local_mean    = gaussian_filter(img, sigma=neighborhood_size / 2.0)

        corrupted                  = img.copy()
        corrupted[boundary_mask]   = (
            (1 - alpha) * img[boundary_mask] +
            alpha * local_mean[boundary_mask]
        )

        return corrupted.astype(np.float32), img.astype(np.float32)

    # --------------------------------------------------
    # 3. FDSC — Frequency-Decomposed Selective Corruption
    # Teaches: global tumour shape AND fine boundary texture
    # --------------------------------------------------
    @staticmethod
    def fdsc(img, low_sigma=8.0, noise_std=0.15):
        x_low  = gaussian_filter(img, sigma=low_sigma)
        x_high = img - x_low
        noise  = np.random.normal(0, noise_std, img.shape).astype(np.float32)

        if np.random.rand() < 0.5:
            corrupted = np.clip(x_low + noise, 0, 1) + x_high
        else:
            corrupted = x_low + (x_high + noise)

        return np.clip(corrupted, 0, 1).astype(np.float32), img.astype(np.float32)


# ======================================================
# STAR DATASET
# ======================================================

class STARSSLDataset(Dataset):
    """
    STAR Self-Supervised Dataset — drop-in replacement for BratsSSLDataset.

    Same interface as your existing BratsSSLDataset:
        __init__(data_dir, file_list_path, stage)
        __getitem__() → (corrupted_tensor, target_tensor)  (1, H, W)

    Stage is set externally from the training loop each epoch.
    """

    def __init__(self, data_dir, file_list_path, stage=1):
        self.data_dir = data_dir
        self.stage    = stage

        with open(file_list_path, "r") as f:
            self.files = f.read().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        img       = np.load(file_path).astype(np.float32)

        if img.max() > 1.0:
            img = img / (img.max() + 1e-8)

        corrupted, target = self._apply_star(img)

        return (
            torch.FloatTensor(corrupted).unsqueeze(0),  # (1, H, W)
            torch.FloatTensor(target).unsqueeze(0)       # (1, H, W)
        )

    def _apply_star(self, img):
        if self.stage == 1:
            return STARCorruption.lshc(img)

        elif self.stage == 2:
            if np.random.randint(0, 2) == 0:
                return STARCorruption.lshc(img)
            else:
                return STARCorruption.gbe(img)

        else:  # stage 3
            choice = np.random.randint(0, 3)
            if choice == 0:
                return STARCorruption.lshc(img)
            elif choice == 1:
                return STARCorruption.gbe(img)
            else:
                return STARCorruption.fdsc(img)


# ======================================================
# BOUNDARY-WEIGHTED MSE LOSS
# 3x higher penalty at tumour boundary pixels
# ======================================================

class BoundaryWeightedMSELoss(nn.Module):

    def __init__(self, boundary_weight=3.0, top_pct=20):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.top_pct         = top_pct

    def _boundary_mask(self, target):
        pad  = nn.functional.pad(target, (1, 1, 1, 1), mode='replicate')
        gx   = pad[:, :, 1:-1, 2:]  - pad[:, :, 1:-1, :-2]
        gy   = pad[:, :, 2:,  1:-1] - pad[:, :, :-2,  1:-1]
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)

        B = grad.shape[0]
        q = torch.quantile(
            grad.view(B, -1),
            1.0 - self.top_pct / 100.0,
            dim=1
        ).view(B, 1, 1, 1)

        return (grad >= q).float()

    def forward(self, pred, target):
        mask       = self._boundary_mask(target)
        weight_map = 1.0 + (self.boundary_weight - 1.0) * mask
        return ((pred - target)**2 * weight_map).mean()


# ======================================================
# CURRICULUM STAGE HELPER
# Epochs  1 -  6 → Stage 1 (LSHC)
# Epochs  7 - 13 → Stage 2 (LSHC + GBE)
# Epochs 14 - 20 → Stage 3 (LSHC + GBE + FDSC)
# ======================================================

def get_stage(epoch, total_epochs=20):
    frac = epoch / total_epochs
    if frac <= 0.30:
        return 1
    elif frac <= 0.65:
        return 2
    else:
        return 3


# ======================================================
# MODEL
# ======================================================

model = UNet(in_channels=1, out_channels=1)
model.to(DEVICE)

criterion = BoundaryWeightedMSELoss(boundary_weight=3.0, top_pct=20)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ======================================================
# TRAINING LOOP
# ======================================================

train_losses = []
val_losses   = []

best_val_loss = float("inf")

os.makedirs("checkpoints", exist_ok=True)

stage_names = {1: "LSHC", 2: "LSHC+GBE", 3: "LSHC+GBE+FDSC"}

print("🚀 Starting STAR Pretraining on CPU...\n")

for epoch in range(EPOCHS):

    stage = get_stage(epoch + 1, EPOCHS)

    # ── Rebuild dataset with correct stage each epoch ──────────────────────
    train_dataset = STARSSLDataset(DATA_DIR, TRAIN_SPLIT, stage=stage)
    val_dataset   = STARSSLDataset(DATA_DIR, VAL_SPLIT,   stage=stage)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,      # deterministic — matches your denoising script
        num_workers=0       # CPU safe
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    running_train_loss = 0.0

    for corrupted, clean in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} [Stage {stage}: {stage_names[stage]}]"
    ):
        corrupted = corrupted.to(DEVICE)
        clean     = clean.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(corrupted)
        loss    = criterion(outputs, clean)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * corrupted.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for corrupted, clean in val_loader:

            corrupted = corrupted.to(DEVICE)
            clean     = clean.to(DEVICE)

            outputs = model(corrupted)
            loss    = criterion(outputs, clean)

            running_val_loss += loss.item() * corrupted.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}] — Stage {stage} [{stage_names[stage]}]")
    print(f"Train Loss: {epoch_train_loss:.6f}")
    print(f"Val Loss:   {epoch_val_loss:.6f}")

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "checkpoints/best_star_model.pth")
        print("✅ Best model saved!")

    print("-" * 50)




# ======================================================
# PLOT LOSS CURVES
# ======================================================

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Boundary-Weighted MSE Loss")
plt.title("STAR Pretraining Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints/star_loss_curve.png")
plt.show()

print("\n🎉 STAR Pretraining Completed Successfully!")
