import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2
from pathlib import Path

class CBISDDSMImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_size=512):
        if isinstance(csv_path, (str, Path)):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = csv_path.copy()

        # IMPORTANT: Use the label column already in your CSV
        if "label" not in self.df.columns:
            raise ValueError("The CSV must contain a 'label' column.")

        # Pre-process patient IDs for consistency
        self.df["patient_id"] = self.df["patient_id"].astype(str).str.extract(r"(P_\d+)")
        
        self.transform = transform
        self.target_size = target_size

        # Print the distribution to your terminal so you can verify it worked
        dist = self.df["label"].value_counts().to_dict()
        print(f"Dataset Loaded | Label Distribution: {dist}")

    def __len__(self):
        return len(self.df)

    def _load_dicom(self, path):
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_dicom(Path(row["file_path"]))
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": int(row["label"]), # Use the existing CSV value
            "patient_id": row["patient_id"]
        }