import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BratsSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, split_file):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        with open(split_file, "r") as f:
            self.files = f.read().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        # Image path
        img_path = os.path.join(self.image_dir, filename)

        # 🔥 FIX: map img → mask
        mask_filename = filename.replace("img", "mask")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load numpy arrays
        image = np.load(img_path)   # shape: (1, H, W)
        mask = np.load(mask_path)   # shape: (H, W)

        # Convert to tensor
        image = torch.FloatTensor(image)              # (1, H, W)
        mask = torch.FloatTensor(mask).unsqueeze(0)   # (1, H, W)

        # Resize to 128x128
        image = F.interpolate(
            image.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(128, 128),
            mode="nearest"
        ).squeeze(0)

        return image, mask