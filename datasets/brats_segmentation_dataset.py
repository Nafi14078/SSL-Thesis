import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BratsSegmentationDataset(Dataset):
    def __init__(self, data_dir):
        self.images = sorted(os.listdir(os.path.join(data_dir, "images")))
        self.masks = sorted(os.listdir(os.path.join(data_dir, "masks")))
        self.data_dir = data_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.data_dir, "images", self.images[idx]))
        mask = np.load(os.path.join(self.data_dir, "masks", self.masks[idx]))

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask