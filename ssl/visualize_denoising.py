import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from datasets.brats_dataset import BratsSSLDataset
from models.unet import UNet


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cpu"
DATA_DIR = "processed"
CHECKPOINT_PATH = "checkpoints/denoising_best.pth"
SAVE_DIR = "results"
SAVE_NAME = "denoising_visualization.png"


# -----------------------------
# Create results folder
# -----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# Load Dataset
# -----------------------------
dataset = BratsSSLDataset(
    data_dir=DATA_DIR,
    task="denoising",
    max_samples=50
)

noisy, clean = dataset[10]


# -----------------------------
# Load Model
# -----------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    output = model(noisy.unsqueeze(0))

output = output.squeeze().cpu().numpy()
noisy = noisy.squeeze().cpu().numpy()
clean = clean.squeeze().cpu().numpy()

# Clip values
output = np.clip(output, 0, 1)
clean = np.clip(clean, 0, 1)

# -----------------------------
# Compute Metrics
# -----------------------------
image_psnr = psnr(clean, output, data_range=1.0)
image_ssim = ssim(clean, output, data_range=1.0)


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Noisy Input")
plt.imshow(noisy, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Denoised\nPSNR: {image_psnr:.2f} dB\nSSIM: {image_ssim:.4f}")
plt.imshow(output, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Ground Truth")
plt.imshow(clean, cmap="gray")
plt.axis("off")

plt.tight_layout()

save_path = os.path.join(SAVE_DIR, SAVE_NAME)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Visualization saved at: {save_path}")
print(f"PSNR: {image_psnr:.2f} dB")
print(f"SSIM: {image_ssim:.4f}")