import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from datasets.brats_dataset import BratsSSLDataset
from models.unet import UNet


# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = "processed"
CHECKPOINT_PATH = "checkpoints/denoising_best.pth"
MAX_SAMPLES = 10000
BATCH_SIZE = 2
VAL_SPLIT = 0.2
DEVICE = "cpu"


# -----------------------------
# Load Dataset
# -----------------------------
dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    task="denoising",
    max_samples=MAX_SAMPLES
)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation samples: {len(val_dataset)}")


# -----------------------------
# Load Model
# -----------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# Evaluation
# -----------------------------
total_psnr = 0
total_ssim = 0
num_images = 0

with torch.no_grad():
    for noisy, clean in val_loader:

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        output = model(noisy)

        output = output.cpu().numpy()
        clean = clean.cpu().numpy()

        for i in range(output.shape[0]):

            pred = output[i, 0]
            gt = clean[i, 0]

            # Ensure values in [0,1]
            pred = np.clip(pred, 0, 1)
            gt = np.clip(gt, 0, 1)

            total_psnr += psnr(gt, pred, data_range=1.0)
            total_ssim += ssim(gt, pred, data_range=1.0)

            num_images += 1


avg_psnr = total_psnr / num_images
avg_ssim = total_ssim / num_images


print("\n===== DENOISING EVALUATION RESULTS =====")
print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print("=========================================")