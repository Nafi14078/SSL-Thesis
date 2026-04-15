import os

TRAIN_SPLIT = "splits_ssl_50k/train.txt"
VAL_SPLIT = "splits_ssl_50k/val.txt"

def read_split(file):
    with open(file, "r") as f:
        return [line.strip() for line in f.readlines()]

train_files = read_split(TRAIN_SPLIT)
val_files = read_split(VAL_SPLIT)

all_files = train_files + val_files

subjects = set()

for file in all_files:
    subject_id = "_".join(file.split("_")[:2])
    subjects.add(subject_id)

print("Total slices:", len(all_files))
print("Unique subjects:", len(subjects))