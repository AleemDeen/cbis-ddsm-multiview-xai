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

    def _load_roi_mask(self, file_path):
        scan_path = Path(file_path)

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
            interpolation=cv2.INTER_NEAREST,
        )

        mask -= mask.min()
        if mask.max() > 0:
            mask /= mask.max()

        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        has_pt = (
            "cc_pt_path" in self.df.columns
            and "mlo_pt_path" in self.df.columns
            and "cc_mask_pt_path" in self.df.columns
            and "mlo_mask_pt_path" in self.df.columns
        )

        if has_pt:
            cc_image  = torch.load(row["cc_pt_path"],       weights_only=True)
            mlo_image = torch.load(row["mlo_pt_path"],      weights_only=True)
            cc_mask   = torch.load(row["cc_mask_pt_path"],  weights_only=True)
            mlo_mask  = torch.load(row["mlo_mask_pt_path"], weights_only=True)
        else:
            cc_image  = torch.from_numpy(self._load_dicom(row["cc_path"])).unsqueeze(0)
            mlo_image = torch.from_numpy(self._load_dicom(row["mlo_path"])).unsqueeze(0)
            cc_mask   = torch.from_numpy(self._load_roi_mask(row["cc_path"])).unsqueeze(0)
            mlo_mask  = torch.from_numpy(self._load_roi_mask(row["mlo_path"])).unsqueeze(0)

        if self.transform:
            cc_image  = self.transform(cc_image)
            mlo_image = self.transform(mlo_image)

        return {
            "cc_image":   cc_image,
            "mlo_image":  mlo_image,
            "cc_mask":    cc_mask,
            "mlo_mask":   mlo_mask,
            "label":      int(row["label"]),
            "patient_id": row["patient_id"],
            "laterality": row["laterality"],
            "cc_path":    row["cc_path"],
            "mlo_path":   row["mlo_path"],
        }
