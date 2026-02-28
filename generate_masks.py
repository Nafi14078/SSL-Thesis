import os
import numpy as np
import nibabel as nib

RAW_DIR = "data/raw"
IMAGE_DIR = "processed/images"
MASK_SAVE_DIR = "processed/masks"

os.makedirs(MASK_SAVE_DIR, exist_ok=True)

image_files = os.listdir(IMAGE_DIR)

for file in image_files:
    if not file.endswith(".npy"):
        continue

    # Example filename: BraTS2021_00000_35.npy
    parts = file.replace(".npy", "").split("_")
    case = "_".join(parts[:-1])
    slice_idx = int(parts[-1])

    seg_path = os.path.join(RAW_DIR, case, f"{case}_seg.nii.gz")

    if not os.path.exists(seg_path):
        continue

    seg_volume = nib.load(seg_path).get_fdata()

    slice_mask = seg_volume[:, :, slice_idx]
    slice_mask = slice_mask.astype(np.uint8)

    save_path = os.path.join(MASK_SAVE_DIR, file)
    if not os.path.exists(save_path):
        np.save(save_path, slice_mask)

print("Mask generation complete for used slices only!")