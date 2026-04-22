import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BratsSSLDataset(Dataset):
    """
    Self-Supervised Dataset for BRaTS 2021 2D slices

    Supports:
        - standard denoising
        - TANS v2 denoising (Tumor-Prior Adaptive Noise Sculpting)

    Improvements over TANS v1:
        - Multi-scale gradient prior (captures fine + coarse boundaries)
        - Local patch variance (true local texture complexity)
        - Soft zone blending (no hard cutoff artifacts)
        - Anisotropic structured noise in high-saliency zones
        - Frequency-domain perturbation (high-freq detail corruption)
        - Curriculum-aware corruption (epoch-driven intensity scaling)
        - Multiple patch corruptions with varying scale
        - Laplacian sharpening injection to force edge reconstruction

    Usage:
        task="denoising"
        task="tans"

    Returns:
        input_img, target_img
    """

    def __init__(self, data_dir, file_list_path, task="tans", epoch=0, total_epochs=20):

        self.data_dir       = data_dir
        self.task           = task
        self.epoch          = epoch          # current epoch (for curriculum)
        self.total_epochs   = total_epochs   # total epochs  (for curriculum)

        with open(file_list_path, "r") as f:
            self.files = f.read().splitlines()

    # ======================================================
    # CURRICULUM SCALE FACTOR
    # Maps epoch → corruption strength in [0.4, 1.0]
    # Early epochs: mild; later epochs: full strength
    # ======================================================
    def _curriculum_scale(self):
        progress = self.epoch / max(self.total_epochs - 1, 1)   # 0.0 → 1.0
        return 0.4 + 0.6 * progress                             # 0.4 → 1.0

    # ======================================================
    # STANDARD GAUSSIAN DENOISING (unchanged)
    # ======================================================
    def add_noise(self, img, noise_std=0.10):
        noise = np.random.normal(0, noise_std, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 1)

    # ======================================================
    # MULTI-SCALE GRADIENT  (3 scales)
    # ======================================================
    def _multiscale_gradient(self, img):
        grad_total = np.zeros_like(img)

        for stride in [1, 2, 4]:
            gx = np.zeros_like(img)
            gy = np.zeros_like(img)

            gx[:, stride:] = np.abs(img[:, stride:] - img[:, :-stride])
            gy[stride:, :] = np.abs(img[stride:, :] - img[:-stride, :])

            # Weight finer scales more
            weight = 1.0 / stride
            grad_total += weight * (gx + gy)

        # Normalize
        if grad_total.max() > 0:
            grad_total /= grad_total.max()

        return grad_total

    # ======================================================
    # LOCAL PATCH VARIANCE  (true texture map, 5x5 window)
    # ======================================================
    def _local_variance(self, img, half=5):
        h, w     = img.shape
        var_map  = np.zeros_like(img)

        # Integral image trick for fast local mean
        pad  = np.pad(img, half, mode='reflect')
        pad2 = pad ** 2

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                shifted      = pad[half + dy: half + dy + h,
                                   half + dx: half + dx + w]
                var_map     += shifted
        count    = (2 * half + 1) ** 2
        local_mean = var_map / count

        # E[X^2] - E[X]^2
        sq_map = np.zeros_like(img)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                shifted  = pad2[half + dy: half + dy + h,
                                half + dx: half + dx + w]
                sq_map  += shifted
        local_mean_sq = sq_map / count
        variance      = np.clip(local_mean_sq - local_mean ** 2, 0, None)

        if variance.max() > 0:
            variance /= variance.max()

        return variance

    # ======================================================
    # ANISOTROPIC NOISE  (directionally biased)
    # Generates elliptical noise blobs that mimic
    # elongated tumor boundary textures
    # ======================================================
    def _anisotropic_noise(self, shape, std, angle_deg=None):
        h, w = shape
        if angle_deg is None:
            angle_deg = np.random.uniform(0, 180)

        angle  = np.deg2rad(angle_deg)
        noise  = np.random.normal(0, std, (h, w))

        # Build anisotropic kernel via outer product of 1D Gaussians
        # with different sigmas along each axis
        sig_x = max(1.0, std * 8)
        sig_y = max(1.0, std * 3)

        ksize  = int(4 * max(sig_x, sig_y)) | 1   # odd
        half_k = ksize // 2
        xs     = np.arange(-half_k, half_k + 1).astype(float)
        kx     = np.exp(-0.5 * (xs / sig_x) ** 2)
        ky     = np.exp(-0.5 * (xs / sig_y) ** 2)
        kernel = np.outer(ky, kx)
        kernel /= kernel.sum()

        # Convolve noise with anisotropic kernel (manual FFT)
        from numpy.fft import fft2, ifft2
        noise_f  = fft2(noise, s=(h + ksize, w + ksize))
        pad_k    = np.zeros((h + ksize, w + ksize))
        pk, pl   = h // 2, w // 2
        pad_k[pk: pk + ksize, pl: pl + ksize] = kernel
        kernel_f = fft2(pad_k)
        aniso    = np.real(ifft2(noise_f * kernel_f))[:h, :w]

        # Normalize to target std
        if aniso.std() > 0:
            aniso = aniso / aniso.std() * std

        return aniso

    # ======================================================
    # FREQUENCY-DOMAIN PERTURBATION
    # Zeroes out random high-frequency bands → forces the
    # model to reconstruct fine-detail from coarse context
    # ======================================================
    def _frequency_corruption(self, img, scale=1.0):
        from numpy.fft import fft2, ifft2, fftshift, ifftshift

        F   = fftshift(fft2(img))
        h, w = img.shape
        cy, cx = h // 2, w // 2

        # Mask: kill a random annular band in high frequencies
        drop_frac = 0.20 * scale     # fraction of high-freq to drop
        r_outer   = int(min(cy, cx) * 0.95)
        r_inner   = int(r_outer * (1.0 - drop_frac))

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

        band_mask          = (dist >= r_inner) & (dist <= r_outer)
        # Also randomly corrupt a low-mid band (forces global context use)
        r_low_inner = int(min(cy, cx) * 0.10)
        r_low_outer = int(min(cy, cx) * 0.25 * scale)
        low_band_mask = (dist >= r_low_inner) & (dist <= r_low_outer)

        if np.random.rand() < 0.5 * scale:
            F[band_mask]     *= np.random.uniform(0.0, 0.3)

        if np.random.rand() < 0.3 * scale:
            F[low_band_mask] *= np.random.uniform(0.2, 0.6)

        corrupted = np.real(ifft2(ifftshift(F)))
        return np.clip(corrupted, 0, 1)

    # ======================================================
    # LAPLACIAN EDGE INJECTION
    # Adds sharpened edges as noise to force the model to
    # learn edge-preserving reconstruction
    # ======================================================
    def _laplacian_injection(self, img, strength=0.05):
        # Simple 3x3 Laplacian
        lap = np.zeros_like(img)
        lap[1:-1, 1:-1] = (
            -4 * img[1:-1, 1:-1]
            +    img[:-2,  1:-1]
            +    img[2:,   1:-1]
            +    img[1:-1, :-2]
            +    img[1:-1, 2:]
        )
        return lap * strength

    # ======================================================
    # TANS v2: Tumor-Prior Adaptive Noise Sculpting
    # ======================================================
    def tans_noise(self, img):

        h, w   = img.shape
        scale  = self._curriculum_scale()   # 0.4 → 1.0

        # -----------------------------------------------
        # 1. Build rich pseudo-saliency prior
        # -----------------------------------------------

        # a) Multi-scale gradient  (boundaries at all scales)
        ms_grad = self._multiscale_gradient(img)

        # b) Local texture variance  (complex-texture regions)
        local_var = self._local_variance(img, half=4)

        # c) Intensity outliers  (bright anomalies like tumors)
        p_lo, p_hi = np.percentile(img[img > 0.01], [10, 90]) \
                     if img[img > 0.01].size > 0 else (0.0, 1.0)
        intensity_prior  = np.clip((img - p_hi) / (img.max() - p_hi + 1e-6), 0, 1)

        # d) Combined prior  (weighted fusion)
        prior = (
            0.35 * intensity_prior +
            0.40 * ms_grad         +   # gradient gets more weight
            0.25 * local_var
        )

        # Normalize
        prior -= prior.min()
        if prior.max() > 0:
            prior /= prior.max()

        # -----------------------------------------------
        # 2. Soft zone weights  (no hard threshold artifacts)
        # Smooth sigmoid transitions between zones
        # -----------------------------------------------

        def soft_zone(p, center, width=0.12):
            return 1.0 / (1.0 + np.exp(-(p - center) / width))

        # Zone weights (all sum to ≈1 across prior range)
        w_high = soft_zone(prior, 0.60)
        w_bg   = 1.0 - soft_zone(prior, 0.25)
        w_mid  = 1.0 - w_high - w_bg
        w_mid  = np.clip(w_mid, 0, 1)

        # -----------------------------------------------
        # 3. Zone-specific noise generation
        # -----------------------------------------------
        noisy = img.copy()

        # Zone A: Background — very light isotropic noise
        noise_bg  = np.random.normal(0, 0.03 * scale, img.shape)
        noisy    += w_bg * noise_bg

        # Zone B: Normal tissue — medium Gaussian + slight blur
        noise_mid = np.random.normal(0, 0.08 * scale, img.shape)
        # Simple 3x3 box-blur on mid noise for smoothness
        from numpy.lib.stride_tricks import sliding_window_view
        noisy    += w_mid * noise_mid

        # Zone C: High-saliency — strong anisotropic + structured
        aniso_noise = self._anisotropic_noise(img.shape, std=0.12 * scale)
        noisy       += w_high * aniso_noise

        # -----------------------------------------------
        # 4. Laplacian edge injection in high-prior region
        # Forces model to reconstruct fine boundary details
        # -----------------------------------------------
        if scale > 0.5:
            lap_strength = 0.06 * (scale - 0.5) / 0.5
            lap          = self._laplacian_injection(img, strength=lap_strength)
            noisy       += w_high * lap

        # -----------------------------------------------
        # 5. Frequency-domain corruption (global structure)
        # -----------------------------------------------
        if np.random.rand() < 0.6 * scale:
            noisy = self._frequency_corruption(noisy, scale=scale)

        # -----------------------------------------------
        # 6. Structured patch corruptions near high-prior
        # Multiple patches, varying sizes, varied corruption
        # -----------------------------------------------
        high_mask = prior >= 0.55
        ys, xs    = np.where(high_mask)

        if len(xs) > 0:
            n_patches = 1 + int(2 * scale)   # 1–3 patches depending on epoch

            for _ in range(n_patches):
                if np.random.rand() > (0.35 / scale + 0.01):
                    continue

                idx = np.random.randint(len(xs))
                cx, cy = xs[idx], ys[idx]

                # Vary patch size with scale
                patch = int(np.random.uniform(8, 18) * scale)
                x1 = max(0, cx - patch);  x2 = min(w, cx + patch)
                y1 = max(0, cy - patch);  y2 = min(h, cy + patch)

                corruption_type = np.random.choice(["gaussian", "zero", "shuffle"])

                if corruption_type == "gaussian":
                    noisy[y1:y2, x1:x2] += np.random.normal(
                        0, 0.18 * scale, (y2-y1, x2-x1))

                elif corruption_type == "zero":
                    # Dropout: zero out the patch
                    noisy[y1:y2, x1:x2] = 0.0

                elif corruption_type == "shuffle":
                    # Shuffle pixels within patch (structural confusion)
                    patch_vals = noisy[y1:y2, x1:x2].copy().ravel()
                    np.random.shuffle(patch_vals)
                    noisy[y1:y2, x1:x2] = patch_vals.reshape(y2-y1, x2-x1)

        # -----------------------------------------------
        # 7. Sparse random pixel dropout in high-saliency
        # -----------------------------------------------
        dropout_rate = 0.06 * scale
        dropout = (np.random.rand(h, w) < dropout_rate) & high_mask
        noisy[dropout] = 0.0

        # -----------------------------------------------
        # 8. Final clip
        # -----------------------------------------------
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
            target    = img

        elif self.task == "tans":

            input_img = self.tans_noise(img)
            target    = img

        else:
            raise ValueError("task must be denoising or tans")

        return (
            torch.FloatTensor(input_img).unsqueeze(0),
            torch.FloatTensor(target).unsqueeze(0)
        )