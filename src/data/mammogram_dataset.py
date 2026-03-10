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

        if "label" not in self.df.columns:
            raise ValueError("The CSV must contain a 'label' column.")

        # Extract case ID only (P_XXXXX)
        self.df["patient_id"] = (
            self.df["patient_id"]
            .astype(str)
            .str.extract(r"(P_\d+)")
        )

        self.transform = transform
        self.target_size = target_size

        dist = self.df["label"].value_counts().to_dict()
        print(f"Dataset Loaded | Label Distribution: {dist}")

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

    def _load_roi_mask(self, file_path):
        scan_path = Path(file_path)

        # Identify ROI folder
        scan_folder = scan_path.parents[2].name
        roi_folder_name = scan_folder + "_1"
        roi_root = scan_path.parents[3] / roi_folder_name

        if not roi_root.exists():
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        dcm_files = list(roi_root.glob("*/*/*.dcm"))
        if len(dcm_files) == 0:
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        ds = pydicom.dcmread(str(dcm_files[0]))
        mask = ds.pixel_array.astype(np.float32)

        mask = cv2.resize(
            mask,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST
        )

        mask -= mask.min()
        if mask.max() > 0:
            mask /= mask.max()

        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_path = row["file_path"]

        # Use preprocessed .pt tensors if available (much faster than DICOM)
        if "image_pt_path" in self.df.columns and "mask_pt_path" in self.df.columns:
            image = torch.load(row["image_pt_path"], weights_only=True)
            mask  = torch.load(row["mask_pt_path"],  weights_only=True)
        else:
            image = torch.from_numpy(self._load_dicom(Path(file_path))).unsqueeze(0)
            mask  = torch.from_numpy(self._load_roi_mask(file_path)).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": int(row["label"]),
            "patient_id": row["patient_id"],
            "file_path": file_path,
            "roi_mask": mask
        }