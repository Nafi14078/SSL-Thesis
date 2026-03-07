import os

IMAGE_DIR = "processed"
MASK_DIR = "processed_masks"

OUTPUT_SPLIT_DIR = "segmentation_splits"
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)

SSL_COUNT = 10000
SEG_COUNT = 10000

TRAIN_COUNT = 8000
VAL_COUNT = 2000


# -----------------------------
# Collect all slice filenames
# -----------------------------
all_files = sorted(os.listdir(IMAGE_DIR))

print("Total slices found:", len(all_files))

# -----------------------------
# Skip SSL dataset
# -----------------------------
seg_files = all_files[SSL_COUNT:SSL_COUNT + SEG_COUNT]

print("Segmentation dataset size:", len(seg_files))

# -----------------------------
# Create train/val split
# -----------------------------
train_files = seg_files[:TRAIN_COUNT]
val_files = seg_files[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]

print("Train:", len(train_files))
print("Val:", len(val_files))

# -----------------------------
# Save split files
# -----------------------------
train_path = os.path.join(OUTPUT_SPLIT_DIR, "train.txt")
val_path = os.path.join(OUTPUT_SPLIT_DIR, "val.txt")

with open(train_path, "w") as f:
    for name in train_files:
        f.write(name + "\n")

with open(val_path, "w") as f:
    for name in val_files:
        f.write(name + "\n")

print("\nSegmentation splits created:")
print(train_path)
print(val_path)