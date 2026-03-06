import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.brats_seg_dataset import BratsSegmentationDataset
from models.unet import UNet


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_DIR = "processed"
MASK_DIR = "processed_masks"
VAL_SPLIT = "splits/val.txt"

CHECKPOINT_PATH = "checkpoints/segmentation_from_inpainting_best.pth"

BATCH_SIZE = 4
DEVICE = "cpu"


# -----------------------------
# Dice + IoU Metrics
# -----------------------------
def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )


def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)


# -----------------------------
# Dataset
# -----------------------------
val_dataset = BratsSegmentationDataset(
    IMAGE_DIR, MASK_DIR, VAL_SPLIT
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"Validation samples: {len(val_dataset)}")


# -----------------------------
# Load Model
# -----------------------------
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# Evaluation Loop
# -----------------------------
total_dice = 0
total_iou = 0
num_batches = 0

with torch.no_grad():
    for images, masks in tqdm(val_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        total_dice += dice_score(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()

        num_batches += 1


avg_dice = total_dice / num_batches
avg_iou = total_iou / num_batches


print("\n===== SEGMENTATION (FROM INPAINTING) RESULTS =====")
print(f"Average Dice Score: {avg_dice:.4f}")
print(f"Average IoU Score:  {avg_iou:.4f}")
print("=================================================")