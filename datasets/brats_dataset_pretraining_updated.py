import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BratsSSLDataset(Dataset):
    """
    Self-Supervised Dataset for BRaTS 2021 2D slices

    Supports:
        - standard denoising
        - TANS denoising (Tumor-Prior Adaptive Noise Sculpting)

    Usage:
        task="denoising"
        task="tans"

    Returns:
        input_img, target_img
    """

    def __init__(self, data_dir, file_list_path, task="tans"):

        self.data_dir = data_dir
        self.task = task

        with open(file_list_path, "r") as f:
            self.files = f.read().splitlines()

    # ======================================================
    # STANDARD GAUSSIAN DENOISING
    # ======================================================
    def add_noise(self, img, noise_std=0.10):
        noise = np.random.normal(0, noise_std, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 1)

    # ======================================================
    # TANS: Tumor-Prior Adaptive Noise Sculpting
    # ======================================================
    def tans_noise(self, img):

        h, w = img.shape

        # ----------------------------------------------
        # 1. Pseudo Tumor Prior Map
        # high intensity + high gradient + local variance
        # ----------------------------------------------

        gx = np.zeros_like(img)
        gy = np.zeros_like(img)

        gx[:, 1:] = np.abs(img[:, 1:] - img[:, :-1])
        gy[1:, :] = np.abs(img[1:, :] - img[:-1, :])

        grad = gx + gy

        # Local variance approximation
        mean_val = img.mean()
        variance_map = np.abs(img - mean_val)

        # High intensity prior
        intensity_prior = img.copy()

        prior = (
            0.5 * intensity_prior +
            0.3 * grad +
            0.2 * variance_map
        )

        # Normalize prior map
        prior = prior - prior.min()
        if prior.max() > 0:
            prior = prior / prior.max()

        # ----------------------------------------------
        # 2. Region Adaptive Noise
        # ----------------------------------------------

        noisy = img.copy()

        # Background region
        bg_mask = prior < 0.25

        # Normal tissue region
        mid_mask = (prior >= 0.25) & (prior < 0.60)

        # Tumor-like / complex region
        high_mask = prior >= 0.60

        # ----------------------------------------------
        # Zone A: light noise
        # ----------------------------------------------
        noise_bg = np.random.normal(0, 0.03, img.shape)
        noisy[bg_mask] += noise_bg[bg_mask]

        # ----------------------------------------------
        # Zone B: medium noise
        # ----------------------------------------------
        noise_mid = np.random.normal(0, 0.08, img.shape)
        noisy[mid_mask] += noise_mid[mid_mask]

        # ----------------------------------------------
        # Zone C: strong sculpted corruption
        # ----------------------------------------------
        noise_high = np.random.normal(0, 0.15, img.shape)
        noisy[high_mask] += noise_high[high_mask]

        # Random dropout in tumor-like regions
        dropout = (np.random.rand(h, w) < 0.08) & high_mask
        noisy[dropout] = 0

        # ----------------------------------------------
        # Random patch corruption near highest prior
        # ----------------------------------------------
        if np.random.rand() > 0.5:

            ys, xs = np.where(high_mask)

            if len(xs) > 0:
                idx = np.random.randint(len(xs))

                cx = xs[idx]
                cy = ys[idx]

                patch = 12

                x1 = max(0, cx - patch)
                x2 = min(w, cx + patch)

                y1 = max(0, cy - patch)
                y2 = min(h, cy + patch)

                noisy[y1:y2, x1:x2] += np.random.normal(
                    0, 0.20, (y2 - y1, x2 - x1)
                )

        # ----------------------------------------------
        # Final clip
        # ----------------------------------------------
        noisy = np.clip(noisy, 0, 1)

        return noisy

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

        # Ensure 2D
        if img.ndim == 3:
            img = img.squeeze()

        if self.task == "denoising":

            input_img = self.add_noise(img)
            target = img

        elif self.task == "tans":

            input_img = self.tans_noise(img)
            target = img

        else:
            raise ValueError("task must be denoising or tans")

        return (
            torch.FloatTensor(input_img).unsqueeze(0),
            torch.FloatTensor(target).unsqueeze(0)
        )