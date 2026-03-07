import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from collections import defaultdict

IMAGE_DIR = "processed"
MASK_DIR = "processed_masks"
RAW_DIR = "data/raw"

os.makedirs(MASK_DIR, exist_ok=True)

files = sorted(os.listdir(IMAGE_DIR))

print("Total slices:", len(files))

# group slices by case
case_slices = defaultdict(list)

for name in files:

    mask_path = os.path.join(MASK_DIR, name)

    if os.path.exists(mask_path):
        continue

    base = name.replace(".npy", "")
    parts = base.split("_")

    case_id = parts[0] + "_" + parts[1]
    slice_idx = int(parts[2])

    case_slices[case_id].append((slice_idx, name))


print("Cases needing masks:", len(case_slices))

generated = 0

for case_id in tqdm(case_slices, desc="Processing BraTS cases"):

    seg_path = os.path.join(
        RAW_DIR,
        case_id,
        case_id + "_seg.nii.gz"
    )

    if not os.path.exists(seg_path):
        print("Missing:", seg_path)
        continue

    seg_volume = nib.load(seg_path).get_fdata()

    for slice_idx, filename in case_slices[case_id]:

        mask_slice = seg_volume[:, :, slice_idx]

        mask_slice = (mask_slice > 0).astype(np.float32)

        mask_path = os.path.join(MASK_DIR, filename)

        np.save(mask_path, mask_slice)

        generated += 1

print("\nMasks generated:", generated)