from datasets.brats_dataset import BratsSSLDataset
import matplotlib.pyplot as plt

dataset = BratsSSLDataset("processed", task="denoising")

noisy, clean = dataset[0]

plt.subplot(1,2,1)
plt.title("Noisy")
plt.imshow(noisy.squeeze(), cmap="gray")

plt.subplot(1,2,2)
plt.title("Clean")
plt.imshow(clean.squeeze(), cmap="gray")

plt.show()