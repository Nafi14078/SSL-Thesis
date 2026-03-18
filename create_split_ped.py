import os, random

IMG_PATH = r"E:\SSL Thesis\processed_ped_10k\images"

files = os.listdir(IMG_PATH)
random.shuffle(files)

train = files[:4000]
val = files[4000:]

with open("train_ped.txt", "w") as f:
    f.write("\n".join(train))

with open("val_ped.txt", "w") as f:
    f.write("\n".join(val))

print("✅ Split created!")