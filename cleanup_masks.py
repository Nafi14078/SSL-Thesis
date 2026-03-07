import os
from tqdm import tqdm

MASK_DIR = "processed_masks"

ssl_split_files = [
    "splits/train.txt",
    "splits/val.txt"
]

seg_split_files = [
    "segmentation_splits/train.txt",
    "segmentation_splits/val.txt"
]

print("Reading split files...")

keep_slices = set()

# read SSL slices
for file in ssl_split_files:
    with open(file) as f:
        for line in f:
            keep_slices.add(line.strip())

# read segmentation slices
for file in seg_split_files:
    with open(file) as f:
        for line in f:
            keep_slices.add(line.strip())

print("Total slices to KEEP:", len(keep_slices))

all_masks = os.listdir(MASK_DIR)

delete_count = 0
keep_count = 0

print("\nCleaning masks...")

for mask in tqdm(all_masks):

    slice_id = mask.replace(".png", "")

    if slice_id not in keep_slices:
        os.remove(os.path.join(MASK_DIR, mask))
        delete_count += 1
    else:
        keep_count += 1

print("\nFinished cleanup")
print("Masks kept:", keep_count)
print("Masks deleted:", delete_count)