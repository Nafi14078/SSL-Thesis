import os
import random

DATA_DIR = "processed"
TOTAL_SAMPLES = 10000
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Get all slice files
all_files = os.listdir(DATA_DIR)
all_files.sort()  # Important for reproducibility

# Select first 10k deterministically
selected_files = all_files[:TOTAL_SAMPLES]

# Shuffle once (fixed seed)
random.shuffle(selected_files)

train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)

train_files = selected_files[:train_size]
val_files = selected_files[train_size:]

os.makedirs("splits", exist_ok=True)

with open("splits/train.txt", "w") as f:
    for file in train_files:
        f.write(file + "\n")

with open("splits/val.txt", "w") as f:
    for file in val_files:
        f.write(file + "\n")

print("✅ Fixed 80/20 split created successfully!")
print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")