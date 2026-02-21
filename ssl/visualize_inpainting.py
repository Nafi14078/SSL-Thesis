import os
import torch
import matplotlib.pyplot as plt

from datasets.brats_dataset import BratsSSLDataset
from models.unet import UNet

device = "cpu"

# -----------------------------
# Create results folder
# -----------------------------
os.makedirs("results", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = BratsSSLDataset(
    data_dir="processed",
    task="inpainting",
    max_samples=100
)

masked, clean = dataset[10]

# -----------------------------
# Load Model
# -----------------------------
model = UNet().to(device)
model.load_state_dict(torch.load("checkpoints/inpainting_best.pth", map_location=device))
model.eval()

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    output = model(masked.unsqueeze(0))

# -----------------------------
# Convert to numpy
# -----------------------------
masked = masked.squeeze().numpy()
clean = clean.squeeze().numpy()
output = output.squeeze().numpy().clip(0, 1)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Masked Input")
plt.imshow(masked, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Reconstructed Output")
plt.imshow(output, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Ground Truth")
plt.imshow(clean, cmap="gray")
plt.axis("off")

plt.tight_layout()

# -----------------------------
# Save
# -----------------------------
save_path = "results/inpainting_example.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ Inpainting visualization saved to {save_path}")