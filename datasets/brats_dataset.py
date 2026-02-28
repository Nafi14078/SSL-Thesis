import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BratsSSLDataset(Dataset):
    """
    Self-Supervised Dataset for BRaTS 2021 2D slices.

    Supports:
        - Denoising
        - Inpainting

    Parameters:
        data_dir (str): Path to processed .npy slices
        file_list_path (str): Path to train.txt or val.txt
        task (str): "denoising" or "inpainting"
    """

    def __init__(self, data_dir, file_list_path, task="denoising"):

        self.data_dir = data_dir
        self.task = task

        # Load fixed file list (deterministic split)
        with open(file_list_path, "r") as f:
            self.files = f.read().splitlines()

    # ======================================================
    # DENOISING
    # ======================================================
    def add_noise(self, img, noise_std=0.1):
        noise = np.random.normal(0, noise_std, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 1)

    # ======================================================
    # INPAINTING
    # ======================================================
    def mask_image(self, img, mask_size=32, num_masks=1):
        h, w = img.shape
        masked_img = img.copy()
        mask = np.zeros_like(img)

        placed_masks = []

        for _ in range(num_masks):

            attempts = 0
            while True:
                x = np.random.randint(0, h - mask_size)
                y = np.random.randint(0, w - mask_size)

                new_box = (x, y, x + mask_size, y + mask_size)

                overlap = False

                # Check overlap with previous masks
                for (px1, py1, px2, py2) in placed_masks:
                    if not (
                        new_box[2] <= px1 or
                        new_box[0] >= px2 or
                        new_box[3] <= py1 or
                        new_box[1] >= py2
                    ):
                        overlap = True
                        break

                if not overlap:
                    placed_masks.append(new_box)
                    masked_img[x:x + mask_size, y:y + mask_size] = 0
                    mask[x:x + mask_size, y:y + mask_size] = 1
                    break

                attempts += 1
                if attempts > 50:
                    break

        return masked_img, mask

    # ======================================================
    # LENGTH
    # ======================================================
    def __len__(self):
        return len(self.files)

    # ======================================================
    # GET ITEM
    # ======================================================
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

            masked_img, mask = self.mask_image(img)

            return (
                torch.FloatTensor(masked_img).unsqueeze(0),
                torch.FloatTensor(img).unsqueeze(0),
                torch.FloatTensor(mask).unsqueeze(0)
            )

        else:
            raise ValueError("Task must be either 'denoising' or 'inpainting'")