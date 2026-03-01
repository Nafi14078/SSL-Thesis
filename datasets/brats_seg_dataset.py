import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BratsSegmentationDataset(Dataset):
    """
    Segmentation dataset using:
        - processed/        (FLAIR slices)
        - processed_masks/  (tumor masks)
        - splits/train.txt / val.txt
    """

    def __init__(self, image_dir, mask_dir, split_file):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        with open(split_file, "r") as f:
            self.files = f.read().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        image = np.load(img_path)
        mask = np.load(mask_path)

        image = torch.FloatTensor(image).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)

        return image, mask