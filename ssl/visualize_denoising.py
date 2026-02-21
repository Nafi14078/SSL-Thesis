import torch
import matplotlib.pyplot as plt
from datasets.brats_dataset import BratsSSLDataset
from models.unet import UNet

device = "cpu"

dataset = BratsSSLDataset("processed", task="denoising", max_samples=50)
noisy, clean = dataset[10]

model = UNet().to(device)
model.load_state_dict(torch.load("checkpoints/denoising_best.pth", map_location=device))
model.eval()

with torch.no_grad():
    output = model(noisy.unsqueeze(0))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Noisy Input")
plt.imshow(noisy.squeeze(), cmap="gray")

plt.subplot(1,3,2)
plt.title("Denoised Output")
plt.imshow(output.squeeze(), cmap="gray")

plt.subplot(1,3,3)
plt.title("Ground Truth")
plt.imshow(clean.squeeze(), cmap="gray")

plt.tight_layout()
plt.show()