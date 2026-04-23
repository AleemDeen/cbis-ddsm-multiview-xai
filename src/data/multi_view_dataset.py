import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2
from pathlib import Path


class CBISDDSMMultiViewDataset(Dataset):
    """
    PyTorch Dataset for paired CC + MLO mammogram views from CBIS-DDSM.

    Each item bundles both views of the same breast alongside their ROI masks
    and the shared malignancy label. Providing CC and MLO together allows the
    model to reason jointly across both projections at training and inference time.

    Accepts either a CSV path or a pre-filtered DataFrame. When .pt tensor
    paths are available in the CSV the dataset loads pre-cached tensors rather
    than decoding DICOMs — recommended for any multi-epoch training run.
    """

    def __init__(self, csv_path, transform=None, target_size=512):
        if isinstance(csv_path, (str, Path)):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = csv_path.copy()

        self.transform   = transform
        self.target_size = target_size

        dist = self.df["label"].value_counts().to_dict()
        print(f"Multi-View Dataset Loaded | Label Distribution: {dist}")

    def __len__(self):
        return len(self.df)

    def _load_dicom(self, path):
        """Decode a DICOM mammogram to a normalised float32 array at target_size."""
        ds  = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)

        # Some CBIS-DDSM acquisitions use MONOCHROME1 (inverted) photometric
        # interpretation. Correcting this ensures bright pixels always mean
        # dense tissue regardless of scanner convention.
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img

        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        img -= img.min()
        if img.max() > 0:
            img /= img.max()

        return img

    def _load_roi_mask(self, file_path):
        """
        Load the ROI binary mask DICOM for a given mammogram scan.

        CBIS-DDSM places masks in a sibling folder named <scan_folder>_1.
        A blank mask is returned when the folder is missing, which happens for
        some benign cases that lack an annotated ROI.
        """
        scan_path = Path(file_path)

        scan_folder     = scan_path.parents[2].name
        roi_folder_name = scan_folder + "_1"
        roi_root        = scan_path.parents[3] / roi_folder_name

        if not roi_root.exists():
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        dcm_files = list(roi_root.glob("*/*/*.dcm"))
        if len(dcm_files) == 0:
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        ds   = pydicom.dcmread(str(dcm_files[0]))
        mask = ds.pixel_array.astype(np.float32)

        # INTER_NEAREST preserves the hard binary edges of the ROI annotation
        mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        mask -= mask.min()
        if mask.max() > 0:
            mask /= mask.max()

        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Check whether all four .pt paths are present in the CSV
        has_pt = (
            "cc_pt_path"       in self.df.columns
            and "mlo_pt_path"      in self.df.columns
            and "cc_mask_pt_path"  in self.df.columns
            and "mlo_mask_pt_path" in self.df.columns
        )

        if has_pt:
            # Load pre-cached tensors — far faster than decoding DICOMs each epoch
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
            # Apply the same transform independently to each view — they are
            # different projections of the same breast, not the same image
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
