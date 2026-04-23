"""
One-time preprocessing script: converts all DICOM images and ROI masks to
512×512 float32 PyTorch tensors and writes a new CSV referencing them.

Decoding DICOMs at each training step is slow — this script performs the
conversion once and caches the result as .pt files. Training scripts detect
the *_pt.csv automatically and prefer it over the raw DICOM CSV, giving
roughly a 10× speedup per epoch for large datasets.

Run from the project root:
    python -m src.data.preprocess_to_pt

Output:
    data_processed/pt/images/<idx>.pt
    data_processed/pt/masks/<idx>.pt
    data_processed/indexed_full_mammogram_images_with_labels_pt.csv
"""

import pandas as pd
import torch
import pydicom
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


TARGET_SIZE = 512
CSV_IN      = Path("data_processed/indexed_full_mammogram_images_with_labels.csv")
CSV_OUT     = Path("data_processed/indexed_full_mammogram_images_with_labels_pt.csv")
OUT_DIR     = Path("data_processed/pt")

IMG_DIR  = OUT_DIR / "images"
MASK_DIR = OUT_DIR / "masks"


def load_dicom_image(path: Path, size: int) -> torch.Tensor:
    """Decode a DICOM mammogram and return a (1, H, W) float32 tensor in [0, 1]."""
    ds  = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)

    # Correct for MONOCHROME1 photometric interpretation (inverted intensity)
    if (
        hasattr(ds, "PhotometricInterpretation")
        and ds.PhotometricInterpretation == "MONOCHROME1"
    ):
        img = img.max() - img

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


def load_roi_mask(file_path: Path, size: int) -> torch.Tensor:
    """
    Load the binary ROI mask for a given mammogram scan.

    Navigates the CBIS-DDSM folder structure to find the sibling ROI folder,
    then returns the first DICOM found resized to size×size. Returns a blank
    tensor when no mask folder exists.
    """
    scan_folder = file_path.parents[2].name
    roi_folder  = scan_folder + "_1"
    roi_root    = file_path.parents[3] / roi_folder

    if not roi_root.exists():
        return torch.zeros(1, size, size, dtype=torch.float32)

    dcm_files = list(roi_root.glob("*/*/*.dcm"))
    if not dcm_files:
        return torch.zeros(1, size, size, dtype=torch.float32)

    ds   = pydicom.dcmread(str(dcm_files[0]))
    mask = ds.pixel_array.astype(np.float32)
    # INTER_NEAREST preserves the hard binary boundary of the annotation
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask -= mask.min()
    if mask.max() > 0:
        mask /= mask.max()

    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_IN)
    print(f"Total samples: {len(df)}")

    image_pt_paths = []
    mask_pt_paths  = []
    failed_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        img_out  = IMG_DIR  / f"{idx}.pt"
        mask_out = MASK_DIR / f"{idx}.pt"

        # Skip samples already converted — safe to re-run after interruption
        if img_out.exists() and mask_out.exists():
            image_pt_paths.append(str(img_out))
            mask_pt_paths.append(str(mask_out))
            continue

        try:
            file_path = Path(row["file_path"])
            image     = load_dicom_image(file_path, TARGET_SIZE)
            mask      = load_roi_mask(file_path, TARGET_SIZE)

            torch.save(image, img_out)
            torch.save(mask,  mask_out)

            image_pt_paths.append(str(img_out))
            mask_pt_paths.append(str(mask_out))

        except Exception as e:
            print(f"\n  [WARN] idx={idx} failed: {e}")
            failed_indices.append(idx)
            image_pt_paths.append(None)
            mask_pt_paths.append(None)

    df["image_pt_path"] = image_pt_paths
    df["mask_pt_path"]  = mask_pt_paths

    # Remove any rows where conversion failed so the output CSV is clean
    before = len(df)
    df = df.dropna(subset=["image_pt_path", "mask_pt_path"])
    if before - len(df) > 0:
        print(f"\n  Dropped {before - len(df)} failed rows.")

    df.to_csv(CSV_OUT, index=False)
    print(f"\nDone. Saved {len(df)} rows to {CSV_OUT}")
    print(f"Preprocessed tensors: {OUT_DIR}")


if __name__ == "__main__":
    main()
