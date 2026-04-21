import os
import nibabel as nib
import numpy as np

# =========================
# PATHS (EDIT THESE)
# =========================
DATASET_PATH = r"E:\SSL Thesis\brats_ped_dataset\ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"
SAVE_IMG_PATH = r"E:\SSL Thesis\processed_ped_10k\images"
SAVE_MASK_PATH = r"E:\SSL Thesis\processed_ped_10k\masks"

os.makedirs(SAVE_IMG_PATH, exist_ok=True)
os.makedirs(SAVE_MASK_PATH, exist_ok=True)

# =========================
# SETTINGS
# ========================
MAX_SLICES = float("inf")  # limit total slices for faster experimentation
MIN_TUMOR_PIXELS = 0   # filter empty slices

count = 0

# =========================
# PROCESSING
# =========================
for patient in os.listdir(DATASET_PATH):
    patient_path = os.path.join(DATASET_PATH, patient)

    if not os.path.isdir(patient_path):
        continue

    try:
        # Load FLAIR (t2f) and mask
        flair_path = os.path.join(patient_path, f"{patient}-t2f.nii.gz")
        seg_path   = os.path.join(patient_path, f"{patient}-seg.nii.gz")

        if not os.path.exists(flair_path) or not os.path.exists(seg_path):
            continue

        flair = nib.load(flair_path).get_fdata()
        seg   = nib.load(seg_path).get_fdata()

    except Exception as e:
        print(f"❌ Skipping {patient}: {e}")
        continue

    # =========================
    # Slice-wise processing
    # =========================
    for i in range(flair.shape[2]):

        mask_slice = seg[:, :, i]

        # Skip empty / near-empty masks
        if np.sum(mask_slice) < MIN_TUMOR_PIXELS:
            continue

        img_slice = flair[:, :, i]

        # Normalize (important)
        img_slice = (img_slice - np.mean(img_slice)) / (np.std(img_slice) + 1e-8)

        # Add channel dimension → (1, H, W)
        img_slice = np.expand_dims(img_slice, axis=0)

        # Convert mask to binary (recommended)
        mask_slice = (mask_slice > 0).astype(np.float32)

        # Save
        np.save(os.path.join(SAVE_IMG_PATH, f"img_{count}.npy"), img_slice.astype(np.float32))
        np.save(os.path.join(SAVE_MASK_PATH, f"mask_{count}.npy"), mask_slice)

        count += 1

        # Stop at 10k
        if count >= MAX_SLICES:
            break

    if count >= MAX_SLICES:
        break

print(f"\n✅ Preprocessing Done! Saved {count} slices.")