import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class BratsSSLDataset(Dataset):
    """
    Self-Supervised Dataset for BRaTS 2021 2D slices.

    Supports:
        - Denoising
        - Inpainting

    Parameters:
        data_dir (str): Path to processed .npy slices
        task (str): "denoising" or "inpainting"
        max_samples (int): Use subset for faster experimentation
    """

    def __init__(self, data_dir, task="denoising", max_samples=None):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)

        # Shuffle files for randomness
        random.shuffle(self.files)

        # Optional subset
        if max_samples is not None:
            self.files = self.files[:max_samples]

        self.task = task

    # ----------------------------
    # DENOISING
    # ----------------------------
    def add_noise(self, img, noise_std=0.1):
        noise = np.random.normal(0, noise_std, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 1)

    # ----------------------------
    # INPAINTING
    # ----------------------------
    def mask_image(self, img, mask_size=48, num_masks=3):
        h, w = img.shape
        masked_img = img.copy()

        for _ in range(num_masks):
            x = np.random.randint(0, h - mask_size)
            y = np.random.randint(0, w - mask_size)

            masked_img[x:x + mask_size, y:y + mask_size] = 0

        return masked_img

    # ----------------------------
    # LENGTH
    # ----------------------------
    def __len__(self):
        return len(self.files)

    # ----------------------------
    # GET ITEM
    # ----------------------------
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        img = np.load(file_path)

        if self.task == "denoising":
            input_img = self.add_noise(img)
            target = img

            return (
                torch.FloatTensor(input_img).unsqueeze(0),
                torch.FloatTensor(target).unsqueeze(0)
            )

        elif self.task == "inpainting":
            masked_img = self.mask_image(img)

            return (
                torch.FloatTensor(masked_img).unsqueeze(0),
                torch.FloatTensor(img).unsqueeze(0)
            )

        else:
            raise ValueError("Task must be either 'denoising' or 'inpainting'")