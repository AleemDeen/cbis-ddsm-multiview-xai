import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2
from pathlib import Path


class CBISDDSMMultiViewDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_size=512):
        if isinstance(csv_path, (str, Path)):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = csv_path.copy()

        self.transform = transform
        self.target_size = target_size

        dist = self.df["label"].value_counts().to_dict()
        print(f"Multi-View Dataset Loaded | Label Distribution: {dist}")

    def __len__(self):
        return len(self.df)

    def _load_dicom(self, path):
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)

        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img

        img = cv2.resize(
            img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        img -= img.min()
        if img.max() > 0:
            img /= img.max()

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cc_image = self._load_dicom(row["cc_path"])
        mlo_image = self._load_dicom(row["mlo_path"])

        cc_image = torch.from_numpy(cc_image).unsqueeze(0)
        mlo_image = torch.from_numpy(mlo_image).unsqueeze(0)

        if self.transform:
            cc_image = self.transform(cc_image)
            mlo_image = self.transform(mlo_image)

        return {
            "cc_image": cc_image,
            "mlo_image": mlo_image,
            "label": int(row["label"]),
            "patient_id": row["patient_id"],
            "laterality": row["laterality"],
        }