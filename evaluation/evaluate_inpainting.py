import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from skimage.metrics import structural_similarity as ssim

from datasets.brats_dataset import BratsSSLDataset
from models.unet import UNet

# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = "processed"
CHECKPOINT_PATH = "checkpoints/inpainting_best_32_single.pth"
MAX_SAMPLES = 10000
BATCH_SIZE = 2
VAL_SPLIT = 0.2
DEVICE = "cpu"

EPS = 1e-10  # Prevent divide-by-zero

# -----------------------------
# Dataset
# -----------------------------
dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    task="inpainting",
    max_samples=MAX_SAMPLES
)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Model
# -----------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# Evaluation (Masked Region Only)
# -----------------------------
total_psnr = 0
total_ssim = 0
num_images = 0

with torch.no_grad():
    for masked, clean, masks in val_loader:

        masked = masked.to(DEVICE)
        clean = clean.to(DEVICE)
        masks = masks.to(DEVICE)

        output = model(masked)

        output = output.cpu().numpy()
        clean = clean.cpu().numpy()
        masks = masks.cpu().numpy()

        for i in range(output.shape[0]):

            pred = np.clip(output[i, 0], 0, 1)
            gt = np.clip(clean[i, 0], 0, 1)
            mask = masks[i, 0]

            # Only masked region
            pred_masked = pred[mask == 1]
            gt_masked = gt[mask == 1]

            if len(pred_masked) == 0:
                continue

            # ---- Stable PSNR ----
            mse = np.mean((gt_masked - pred_masked) ** 2)
            psnr_value = 10 * np.log10(1.0 / (mse + EPS))
            total_psnr += psnr_value

            # ---- Full-image SSIM ----
            total_ssim += ssim(gt, pred, data_range=1.0)

            num_images += 1

avg_psnr = total_psnr / num_images
avg_ssim = total_ssim / num_images

print("\n===== INPAINTING EVALUATION RESULTS =====")
print(f"Masked Region PSNR: {avg_psnr:.4f} dB")
print(f"Full Image SSIM:    {avg_ssim:.4f}")
print("==========================================")