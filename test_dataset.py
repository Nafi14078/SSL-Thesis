from datasets.brats_dataset import BratsSSLDataset
import matplotlib.pyplot as plt
import torch

# -------------------------------
# Choose task here:
# "denoising" or "inpainting"
# -------------------------------
TASK = "inpainting"

if TASK == "denoising":

    dataset = BratsSSLDataset(
        data_dir="processed",
        file_list_path="splits/train.txt",
        task="denoising"
    )

    noisy, clean = dataset[0]

    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title("Noisy")
    plt.imshow(noisy.squeeze().numpy(), cmap="gray")

    plt.subplot(1,2,2)
    plt.title("Clean")
    plt.imshow(clean.squeeze().numpy(), cmap="gray")

    plt.tight_layout()
    plt.show()


elif TASK == "inpainting":

    dataset = BratsSSLDataset(
        data_dir="processed",
        file_list_path="splits/train.txt",
        task="inpainting"
    )

    masked, original, mask = dataset[0]

    print("Masked shape:", masked.shape)
    print("Original shape:", original.shape)
    print("Mask shape:", mask.shape)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Masked Input")
    plt.imshow(masked.squeeze().numpy(), cmap="gray")

    plt.subplot(1,3,2)
    plt.title("Original")
    plt.imshow(original.squeeze().numpy(), cmap="gray")

    plt.subplot(1,3,3)
    plt.title("Mask")
    plt.imshow(mask.squeeze().numpy(), cmap="gray")

    plt.tight_layout()
    plt.show()

else:
    raise ValueError("TASK must be 'denoising' or 'inpainting'")