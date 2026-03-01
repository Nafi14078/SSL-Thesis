import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ==============================
# PATH CONFIG
# ==============================
RAW_DIR = "data/raw"
PROCESSED_DIR = "processed"
MASK_SAVE_DIR = "processed_masks"
SPLITS_DIR = "splits"

os.makedirs(MASK_SAVE_DIR, exist_ok=True)

# ==============================
# LOAD 10K FIXED SPLIT FILES
# ==============================
train_list = open(os.path.join(SPLITS_DIR, "train.txt")).read().splitlines()
val_list = open(os.path.join(SPLITS_DIR, "val.txt")).read().splitlines()

all_files = train_list + val_list

print(f"Total slices to process: {len(all_files)}")

# ==============================
# PROCESS EACH SLICE
# ==============================
for filename in tqdm(all_files):

    # Example: BraTS2021_00033_144.npy
    base = filename.replace(".npy", "")

    # Split at LAST underscore
    patient_id, slice_idx = base.rsplit("_", 1)
    slice_idx = int(slice_idx)

    # Path to segmentation volume
    seg_path = os.path.join(
        RAW_DIR,
        patient_id,
        f"{patient_id}_seg.nii.gz"
    )

    if not os.path.exists(seg_path):
        print(f"Missing segmentation file: {seg_path}")
        continue

    # Load 3D segmentation volume
    seg_nii = nib.load(seg_path)
    seg_volume = seg_nii.get_fdata()

    # Extract same slice index
    seg_slice = seg_volume[:, :, slice_idx]

    # Convert to binary tumor mask (whole tumor)
    # Tumor labels in BraTS are 1,2,4 → make them 1
    seg_slice = (seg_slice > 0).astype(np.float32)

    # Resize to 128x128 (same as your processed slices)
    # Only needed if preprocessing resized images
    if seg_slice.shape[0] != 128:
        from skimage.transform import resize
        seg_slice = resize(
            seg_slice,
            (128, 128),
            order=0,  # IMPORTANT: keep labels discrete
            preserve_range=True,
            anti_aliasing=False
        )

    # Save mask
    save_path = os.path.join(MASK_SAVE_DIR, filename)
    np.save(save_path, seg_slice)

print("✅ Segmentation mask creation completed successfully!")