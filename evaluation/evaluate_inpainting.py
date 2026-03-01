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
CHECKPOINT_PATH = "checkpoints/best_inpainting_model.pth"

BATCH_SIZE = 8


# ======================================================
# Load Validation Dataset (Deterministic)
# ======================================================

val_dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    file_list_path=VAL_SPLIT_PATH,
    task="inpainting"
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
# Evaluation Loop (Masked Region Only)
# ======================================================

total_psnr = 0.0
total_ssim = 0.0
num_images = 0

with torch.no_grad():
    for masked_img, original, mask in tqdm(val_loader, desc="Evaluating"):

        masked_img = masked_img.to(DEVICE)
        original = original.to(DEVICE)
        mask = mask.to(DEVICE)

        output = model(masked_img)

        output = output.cpu().numpy()
        original = original.cpu().numpy()
        mask = mask.cpu().numpy()

        for i in range(output.shape[0]):

            pred = output[i, 0]
            gt = original[i, 0]
            m = mask[i, 0]

            # Only evaluate masked region
            masked_pixels = m > 0.5

            if np.sum(masked_pixels) == 0:
                continue

            pred_masked = pred[masked_pixels]
            gt_masked = gt[masked_pixels]

            pred_masked = np.clip(pred_masked, 0, 1)
            gt_masked = np.clip(gt_masked, 0, 1)

            total_psnr += psnr(gt_masked, pred_masked, data_range=1.0)
            total_ssim += ssim(gt_masked, pred_masked, data_range=1.0)

            num_images += 1


avg_psnr = total_psnr / num_images
avg_ssim = total_ssim / num_images


# ======================================================
# Print Results
# ======================================================

print("\n===== INPAINTING EVALUATION RESULTS =====")
print(f"Total images evaluated: {num_images}")
print(f"Average Masked PSNR: {avg_psnr:.4f} dB")
print(f"Average Masked SSIM: {avg_ssim:.4f}")
print("==========================================")