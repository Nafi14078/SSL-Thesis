import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

def normalize(volume):
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
    return volume

def process_brats(brats_path, output_path, modality="flair"):
    os.makedirs(output_path, exist_ok=True)

    patients = os.listdir(brats_path)

    for patient in tqdm(patients):
        patient_path = os.path.join(brats_path, patient)

        if not os.path.isdir(patient_path):
            continue

        for file in os.listdir(patient_path):
            if modality in file and file.endswith(".nii.gz"):

                file_path = os.path.join(patient_path, file)
                img = nib.load(file_path)
                volume = img.get_fdata()

                volume = normalize(volume)

                for i in range(volume.shape[2]):
                    slice_2d = volume[:, :, i]

                    # Skip almost empty slices
                    if np.sum(np.abs(slice_2d)) < 50:
                        continue

                    slice_2d = cv2.resize(slice_2d, (128, 128))
                    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)

                    save_name = f"{patient}_{i}.npy"
                    np.save(os.path.join(output_path, save_name), slice_2d)

    print("✅ Preprocessing Completed Successfully!")