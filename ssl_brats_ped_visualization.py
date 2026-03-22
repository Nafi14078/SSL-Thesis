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
SSL_MODEL = "checkpoints/ped_from_denoising_best.pth"

DEVICE = "cpu"
BATCH_SIZE = 1

# RESULTS FOLDER
RESULTS_DIR = "results/brats_ped_ssl"
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
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# ----------------------------------
# Load Models
# ----------------------------------
baseline = UNet(1, 1).to(DEVICE)
baseline.load_state_dict(torch.load(BASELINE_MODEL, map_location=DEVICE))
baseline.eval()

ssl = UNet(1, 1).to(DEVICE)
ssl.load_state_dict(torch.load(SSL_MODEL, map_location=DEVICE))
ssl.eval()


# ----------------------------------
# Evaluation
# ----------------------------------
baseline_metrics = []
ssl_metrics = []

sample_images = []
sample_masks = []
baseline_preds = []
ssl_preds = []

with torch.no_grad():
    for images, masks in tqdm(loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        b_out = baseline(images)
        s_out = ssl(images)

        b_metrics = compute_metrics(b_out, masks)
        s_metrics = compute_metrics(s_out, masks)

        baseline_metrics.append(b_metrics)
        ssl_metrics.append(s_metrics)

        # store some difficult cases
        if len(sample_images) < 5:
            sample_images.append(images.cpu())
            sample_masks.append(masks.cpu())
            baseline_preds.append(b_out.cpu())
            ssl_preds.append(s_out.cpu())


baseline_metrics = np.array(baseline_metrics)
ssl_metrics = np.array(ssl_metrics)


# ----------------------------------
# SUMMARY TABLE
# ----------------------------------
def summarize(metrics):
    return metrics.mean(axis=0)


b_mean = summarize(baseline_metrics)
s_mean = summarize(ssl_metrics)

print("\n===== SUMMARY TABLE =====")
print("Model | Dice | IoU | Precision | Recall | F1")
print(
    f"Baseline | {b_mean[0]:.4f} | {b_mean[1]:.4f} | {b_mean[2]:.4f} | {b_mean[3]:.4f} | {b_mean[4]:.4f}"
)
print(
    f"SSL | {s_mean[0]:.4f} | {s_mean[1]:.4f} | {s_mean[2]:.4f} | {s_mean[3]:.4f} | {s_mean[4]:.4f}"
)


# ----------------------------------
# BAR GRAPH
# ----------------------------------
labels = ["Dice", "IoU", "Precision", "Recall", "F1"]

plt.figure(figsize=(8, 5))
plt.bar(np.arange(5) - 0.2, b_mean, width=0.4, label="Baseline")
plt.bar(np.arange(5) + 0.2, s_mean, width=0.4, label="SSL")
plt.xticks(range(5), labels)
plt.legend()
plt.title("Baseline vs SSL Comparison")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "comparison_bar.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ----------------------------------
# BOX PLOT
# ----------------------------------
plt.figure(figsize=(6, 5))
plt.boxplot([baseline_metrics[:, 0], ssl_metrics[:, 0]])
plt.xticks([1, 2], ["Baseline", "SSL"])
plt.ylabel("Dice Score")
plt.title("Dice Distribution")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "dice_boxplot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ----------------------------------
# QUALITATIVE COMPARISON GRID
# ----------------------------------
fig, axes = plt.subplots(5, 4, figsize=(12, 12))

for i in range(5):

    img = sample_images[i][0, 0]
    mask = sample_masks[i][0, 0]

    b = (torch.sigmoid(baseline_preds[i])[0, 0] > 0.5)
    s = (torch.sigmoid(ssl_preds[i])[0, 0] > 0.5)

    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title("FLAIR")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Ground Truth")

    axes[i, 2].imshow(b, cmap="gray")
    axes[i, 2].set_title("Baseline")

    axes[i, 3].imshow(s, cmap="gray")
    axes[i, 3].set_title("SSL")

    for j in range(4):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "qualitative_grid.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ----------------------------------
# ERROR HEATMAP
# ----------------------------------
plt.figure(figsize=(6, 6))

img = sample_images[0][0, 0]
b = (torch.sigmoid(baseline_preds[0])[0, 0] > 0.5)
s = (torch.sigmoid(ssl_preds[0])[0, 0] > 0.5)

error = b.float() - s.float()

plt.imshow(img, cmap="gray")
plt.imshow(error, alpha=0.5)
plt.title("Error Map (Baseline - SSL)")
plt.axis("off")

plt.savefig(
    os.path.join(RESULTS_DIR, "error_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ----------------------------------
# BLAND ALTMAN PLOT
# ----------------------------------
gt_area = []
pred_area = []

for i in range(len(sample_masks)):
    gt = sample_masks[i].sum()
    pr = (torch.sigmoid(ssl_preds[i]) > 0.5).sum()

    gt_area.append(gt.item())
    pred_area.append(pr.item())

gt_area = np.array(gt_area)
pred_area = np.array(pred_area)

mean = (gt_area + pred_area) / 2
diff = pred_area - gt_area

plt.figure(figsize=(6, 5))
plt.scatter(mean, diff)
plt.axhline(diff.mean(), linestyle="--")
plt.title("Bland-Altman Plot")
plt.xlabel("Mean Tumor Area")
plt.ylabel("Difference")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "bland_altman.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"\nAll results saved to: {RESULTS_DIR}")