from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2


class CBISDDSMImageDataset(Dataset):
    """
    PyTorch Dataset for CBIS-DDSM full mammogram images.

    Preprocessing (locked):
    - Convert MONOCHROME1 -> MONOCHROME2
    - Resize to 512x512
    - Per-image min-max normalisation
    """

    def __init__(self, csv_path, transform=None, target_size=512):
        if isinstance(csv_path, str) or isinstance(csv_path, Path):
            self.df = pd.read_csv(csv_path)

            # --------------------------------------------------
            # Normalise patient_id to match split files (P_XXXXX)
            # --------------------------------------------------
            self.df["patient_id"] = (
                self.df["patient_id"]
                .astype(str)
                .str.extract(r"(P_\d+)")
            )

        else:
            self.df = csv_path.copy()

        self.transform = transform
        self.target_size = target_size

        if self.df.empty:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.df)

    def _load_dicom(self, path):
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)

        # Handle photometric interpretation
        if hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                img = img.max() - img

        # Resize to fixed resolution
        img = cv2.resize(
            img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA
        )

        # Min-max normalisation
        img -= img.min()
        if img.max() > 0:
            img /= img.max()

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["file_path"])
        image = self._load_dicom(img_path)

        # Add channel dimension: (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        label = 1 if "MALIGNANT" in str(row["path_text"]).upper() else 0

        sample = {
            "image": image,
            "label": label,
            "view": row["view"],
            "laterality": row["laterality"],
            "case_folder": row["case_folder"],
            "patient_id": row["patient_id"],
        }

        return sample
