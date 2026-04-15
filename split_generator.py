import os
import random

DATA_DIR = "processed"
OUTPUT_DIR = "splits_ssl_50k"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_files = os.listdir(DATA_DIR)
random.shuffle(all_files)

selected_files = all_files[:50000]

train_split = selected_files[:40000]
val_split = selected_files[40000:50000]

with open(os.path.join(OUTPUT_DIR, "train.txt"), "w") as f:
    f.write("\n".join(train_split))

with open(os.path.join(OUTPUT_DIR, "val.txt"), "w") as f:
    f.write("\n".join(val_split))

print("✅ 50k split created!")