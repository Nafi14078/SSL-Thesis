import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from datasets.brats_dataset import BratsSSLDataset
from models.unet import UNet


# ======================================================
# SETTINGS
# ======================================================

DEVICE = torch.device("cpu")

DATA_DIR = "processed"
VAL_SPLIT_PATH = "splits/val.txt"
CHECKPOINT_PATH = "checkpoints/best_denoising_model.pth"

BATCH_SIZE = 8   # Can increase slightly for faster eval


# ======================================================
# Load Validation Dataset (Deterministic)
# ======================================================

val_dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    file_list_path=VAL_SPLIT_PATH,
    task="denoising"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation samples: {len(val_dataset)}")


# ======================================================
# Load Model
# ======================================================

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


# ======================================================
# Evaluation Loop
# ======================================================

total_psnr = 0.0
total_ssim = 0.0
num_images = 0

with torch.no_grad():
    for noisy, clean in tqdm(val_loader, desc="Evaluating"):

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        output = model(noisy)

        output = output.cpu().numpy()
        clean = clean.cpu().numpy()

        for i in range(output.shape[0]):

            pred = output[i, 0]
            gt = clean[i, 0]

            pred = np.clip(pred, 0, 1)
            gt = np.clip(gt, 0, 1)

            total_psnr += psnr(gt, pred, data_range=1.0)
            total_ssim += ssim(gt, pred, data_range=1.0)

            num_images += 1


avg_psnr = total_psnr / num_images
avg_ssim = total_ssim / num_images


# ======================================================
# Print Results
# ======================================================

print("\n===== DENOISING EVALUATION RESULTS =====")
print(f"Total images evaluated: {num_images}")
print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print("=========================================")