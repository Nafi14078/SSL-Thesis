import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.brats_seg_dataset import BratsSegmentationDataset
from models.unet import UNet


# ----------------------------------
# SETTINGS
# ----------------------------------
IMAGE_DIR = "processed_ped_10k/images"
MASK_DIR = "processed_ped_10k/masks"
VAL_SPLIT = "val_ped.txt"

BASELINE_MODEL = "checkpoints/ped_baseline_best.pth"
DENOISING_MODEL = "checkpoints/ped_from_denoising_best.pth"
INPAINTING_MODEL = "checkpoints/ped_from_inpainting_best.pth"

DEVICE = "cpu"
BATCH_SIZE = 1

RESULTS_DIR = "results/brats_ped_ssl_all"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------
# Metrics
# ----------------------------------
def compute_metrics(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return dice.item(), iou.item(), precision.item(), recall.item(), f1.item()


# ----------------------------------
# Dataset
# ----------------------------------
dataset = BratsSegmentationDataset(IMAGE_DIR, MASK_DIR, VAL_SPLIT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# ----------------------------------
# Load Models
# ----------------------------------
baseline = UNet(1, 1).to(DEVICE)
baseline.load_state_dict(torch.load(BASELINE_MODEL, map_location=DEVICE))
baseline.eval()

denoising = UNet(1, 1).to(DEVICE)
denoising.load_state_dict(torch.load(DENOISING_MODEL, map_location=DEVICE))
denoising.eval()

inpainting = UNet(1, 1).to(DEVICE)
inpainting.load_state_dict(torch.load(INPAINTING_MODEL, map_location=DEVICE))
inpainting.eval()


# ----------------------------------
# Evaluation
# ----------------------------------
baseline_metrics = []
denoising_metrics = []
inpainting_metrics = []

sample_images = []
sample_masks = []
baseline_preds = []
denoising_preds = []
inpainting_preds = []

with torch.no_grad():
    for images, masks in tqdm(loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        b_out = baseline(images)
        d_out = denoising(images)
        i_out = inpainting(images)

        baseline_metrics.append(compute_metrics(b_out, masks))
        denoising_metrics.append(compute_metrics(d_out, masks))
        inpainting_metrics.append(compute_metrics(i_out, masks))

        if len(sample_images) < 5:
            sample_images.append(images.cpu())
            sample_masks.append(masks.cpu())
            baseline_preds.append(b_out.cpu())
            denoising_preds.append(d_out.cpu())
            inpainting_preds.append(i_out.cpu())


baseline_metrics = np.array(baseline_metrics)
denoising_metrics = np.array(denoising_metrics)
inpainting_metrics = np.array(inpainting_metrics)


# ----------------------------------
# SUMMARY TABLE
# ----------------------------------
def summarize(metrics):
    return metrics.mean(axis=0)


b_mean = summarize(baseline_metrics)
d_mean = summarize(denoising_metrics)
i_mean = summarize(inpainting_metrics)

print("\n===== SUMMARY TABLE =====")
print("Model | Dice | IoU | Precision | Recall | F1")

print(f"Baseline  | {b_mean[0]:.4f} | {b_mean[1]:.4f} | {b_mean[2]:.4f} | {b_mean[3]:.4f} | {b_mean[4]:.4f}")
print(f"Inpainting| {i_mean[0]:.4f} | {i_mean[1]:.4f} | {i_mean[2]:.4f} | {i_mean[3]:.4f} | {i_mean[4]:.4f}")
print(f"Denoising | {d_mean[0]:.4f} | {d_mean[1]:.4f} | {d_mean[2]:.4f} | {d_mean[3]:.4f} | {d_mean[4]:.4f}")


# ----------------------------------
# BAR GRAPH
# ----------------------------------
labels = ["Dice", "IoU", "Precision", "Recall", "F1"]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(9, 5))
plt.bar(x - width, b_mean, width, label="Baseline")
plt.bar(x, i_mean, width, label="Inpainting")
plt.bar(x + width, d_mean, width, label="Denoising")

plt.xticks(x, labels)
plt.legend()
plt.title("Baseline vs SSL (Inpainting vs Denoising)")
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "comparison_bar.png"), dpi=300)
plt.close()


# ----------------------------------
# BOX PLOT
# ----------------------------------
plt.figure(figsize=(7, 5))
plt.boxplot([
    baseline_metrics[:, 0],
    inpainting_metrics[:, 0],
    denoising_metrics[:, 0]
])

plt.xticks([1, 2, 3], ["Baseline", "Inpainting", "Denoising"])
plt.ylabel("Dice Score")
plt.title("Dice Distribution")
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "dice_boxplot.png"), dpi=300)
plt.close()


# ----------------------------------
# QUALITATIVE GRID
# ----------------------------------
fig, axes = plt.subplots(5, 5, figsize=(15, 12))

for i in range(5):

    img = sample_images[i][0, 0]
    mask = sample_masks[i][0, 0]

    b = (torch.sigmoid(baseline_preds[i])[0, 0] > 0.5)
    d = (torch.sigmoid(denoising_preds[i])[0, 0] > 0.5)
    inp = (torch.sigmoid(inpainting_preds[i])[0, 0] > 0.5)

    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title("FLAIR")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("GT")

    axes[i, 2].imshow(b, cmap="gray")
    axes[i, 2].set_title("Baseline")

    axes[i, 3].imshow(inp, cmap="gray")
    axes[i, 3].set_title("Inpainting")

    axes[i, 4].imshow(d, cmap="gray")
    axes[i, 4].set_title("Denoising")

    for j in range(5):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "qualitative_grid.png"), dpi=300)
plt.close()


# ----------------------------------
# ERROR HEATMAP (Baseline vs Denoising)
# ----------------------------------
plt.figure(figsize=(6, 6))

img = sample_images[0][0, 0]
b = (torch.sigmoid(baseline_preds[0])[0, 0] > 0.5)
d = (torch.sigmoid(denoising_preds[0])[0, 0] > 0.5)

error = b.float() - d.float()

plt.imshow(img, cmap="gray")
plt.imshow(error, alpha=0.5)
plt.title("Error Map (Baseline - Denoising)")
plt.axis("off")

plt.savefig(os.path.join(RESULTS_DIR, "error_heatmap.png"), dpi=300)
plt.close()


print(f"\nAll results saved to: {RESULTS_DIR}")