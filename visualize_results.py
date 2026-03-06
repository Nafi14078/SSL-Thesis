import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.brats_seg_dataset import BratsSegmentationDataset
from models.unet import UNet


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_DIR = "processed"
MASK_DIR = "processed_masks"
VAL_SPLIT = "splits/val.txt"

CHECKPOINT_PATH = "checkpoints/segmentation_baseline_best.pth"

BATCH_SIZE = 1
DEVICE = "cpu"

SAVE_DIR = "analysis_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# DATASET
# -----------------------------
dataset = BratsSegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    VAL_SPLIT
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -----------------------------
# MODEL
# -----------------------------
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# 1️⃣ SEGMENTATION VISUALIZATION
# -----------------------------
print("Generating segmentation visualizations...")

for i, (img, mask) in enumerate(loader):

    if i >= 5:
        break

    img = img.to(DEVICE)
    mask = mask.to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(img))
        pred = (pred > 0.5).float()

    img_np = img.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1,3, figsize=(12,4))

    ax[0].imshow(img_np, cmap="gray")
    ax[0].set_title("Input MRI")

    ax[1].imshow(mask_np, cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred_np, cmap="gray")
    ax[2].set_title("Predicted Mask")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/segmentation_result_{i}.png")
    plt.close()


print("Segmentation visualizations saved.")


# -----------------------------
# 2️⃣ TRAINING CURVE VISUALIZATION
# -----------------------------
print("Generating training curves...")

# Example losses (replace with your real logs if available)
train_loss = [
0.69,0.58,0.51,0.46,0.41,0.37,0.34,0.30,0.28,0.25,
0.23,0.22,0.21,0.20,0.19,0.18,0.17,0.16,0.17,0.16
]

val_loss = [
0.65,0.56,0.50,0.47,0.43,0.40,0.36,0.33,0.30,0.28,
0.26,0.25,0.23,0.21,0.20,0.19,0.15,0.16,0.17,0.17
]

plt.figure()

plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")

plt.legend()

plt.savefig(f"{SAVE_DIR}/training_curve.png")
plt.close()

print("Training curve saved.")


# -----------------------------
# 3️⃣ FEATURE MAP VISUALIZATION
# -----------------------------
print("Generating feature map visualization...")

img, _ = dataset[0]
img = img.unsqueeze(0).to(DEVICE)

features = []

def hook(module, input, output):
    features.append(output)

# automatically hook FIRST convolution layer
first_conv = None
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        first_conv = module
        break

first_conv.register_forward_hook(hook)

with torch.no_grad():
    model(img)

feature_map = features[0].squeeze().cpu().numpy()

num_features = min(6, feature_map.shape[0])

fig, axes = plt.subplots(1, num_features, figsize=(15,3))

for i in range(num_features):
    axes[i].imshow(feature_map[i], cmap="gray")
    axes[i].set_title(f"Feature {i}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_maps.png")
plt.close()

print("Feature maps saved.")