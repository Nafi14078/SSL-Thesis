import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.brats_dataset_different_sample import BratsSegmentationDataset
from models.unet import UNet


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_DIR = "processed"
MASK_DIR = "processed_masks"
VAL_SPLIT = "segmentation_splits/val.txt"

CHECKPOINT_PATH = "checkpoints/segmentation_from_denoising_best(different 10k).pth"

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Dice + IoU
# -----------------------------
def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    return ((2 * intersection + smooth) / (union + smooth)).mean()


def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection

    return ((intersection + smooth) / (union + smooth)).mean()


# -----------------------------
# Dataset
# -----------------------------
val_dataset = BratsSegmentationDataset(
    IMAGE_DIR, MASK_DIR, VAL_SPLIT
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Validation samples: {len(val_dataset)}")
print(f"Using device: {DEVICE}")


# -----------------------------
# Load Model
# -----------------------------
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# Evaluation
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


print("\n===== SEGMENTATION (FROM DENOISING - Different 10k sample) RESULTS =====")
print(f"Average Dice Score: {avg_dice:.4f}")
print(f"Average IoU Score:  {avg_iou:.4f}")
print("============================================================")