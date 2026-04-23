import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2
from pathlib import Path


class CBISDDSMImageDataset(Dataset):
    """
    PyTorch Dataset for the single-view CBIS-DDSM mammogram collection.

    Each item corresponds to one mammogram image (CC or MLO) with its
    binary malignancy label and ROI mask.

    The dataset accepts either a CSV path or a pre-filtered DataFrame directly,
    which allows split-specific subsets to be constructed outside this class
    and passed in without redundant file I/O.

    If preprocessed .pt tensor paths are present in the CSV the dataset loads
    those instead of decoding DICOMs at each step — roughly 10× faster.
    """

    def __init__(self, csv_path, transform=None, target_size=512):
        if isinstance(csv_path, (str, Path)):
            self.df = pd.read_csv(csv_path)
        else:
            # Accept a pre-sliced DataFrame (e.g. from build_dataloaders)
            self.df = csv_path.copy()

        if "label" not in self.df.columns:
            raise ValueError("The CSV must contain a 'label' column.")

        # Normalise patient IDs to the P_XXXXX format used in the split files.
        # The raw CBIS-DDSM metadata sometimes includes suffixes that break lookups.
        self.df["patient_id"] = (
            self.df["patient_id"]
            .astype(str)
            .str.extract(r"(P_\d+)")
        )

        self.transform   = transform
        self.target_size = target_size

        dist = self.df["label"].value_counts().to_dict()
        print(f"Dataset Loaded | Label Distribution: {dist}")

    def __len__(self):
        return len(self.df)

    def _load_dicom(self, path):
        """Decode a DICOM mammogram to a normalised float32 array at target_size."""
        ds  = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)

        # MONOCHROME1 images store bright tissue as low pixel values (inverted).
        # Flipping them ensures that bright pixels always represent dense tissue
        # regardless of which acquisition convention was used.
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img

        # INTER_AREA downsampling averages pixel blocks — preferred over bilinear
        # for significant size reductions as it avoids aliasing artefacts
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        # Min-max normalise to [0, 1] so all inputs share the same dynamic range
        img -= img.min()
        if img.max() > 0:
            img /= img.max()

        return img

    def _load_roi_mask(self, file_path):
        """
        Locate and load the ROI binary mask DICOM associated with this scan.

        CBIS-DDSM stores ROI masks in a sibling folder named <scan_folder>_1.
        If no mask is found (e.g. for some benign cases), a blank mask is returned
        so training batches remain consistently shaped.
        """
        scan_path = Path(file_path)

        # Navigate two levels up from the DICOM file to get the series folder,
        # then look for the sibling ROI folder at the study level
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

        # INTER_NEAREST preserves the hard binary boundary of the ROI annotation.
        # Bilinear interpolation would create fractional values along the edges
        # which would introduce ambiguity for the segmentation loss.
        mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        mask -= mask.min()
        if mask.max() > 0:
            mask /= mask.max()

        return mask

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        file_path = row["file_path"]

        # Prefer cached .pt tensors if available — DICOM decoding is expensive
        # and becomes a bottleneck when training with many workers
        if "image_pt_path" in self.df.columns and "mask_pt_path" in self.df.columns:
            image = torch.load(row["image_pt_path"], weights_only=True)
            mask  = torch.load(row["mask_pt_path"],  weights_only=True)
        else:
            image = torch.from_numpy(self._load_dicom(Path(file_path))).unsqueeze(0)
            mask  = torch.from_numpy(self._load_roi_mask(file_path)).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return {
            "image":      image,
            "label":      int(row["label"]),
            "patient_id": row["patient_id"],
            "file_path":  file_path,
            "roi_mask":   mask,
        }
