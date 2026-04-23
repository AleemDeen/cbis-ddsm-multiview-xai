"""
One-time preprocessing script: converts all multi-view DICOM images and ROI
masks to 512×512 float32 PyTorch tensors and writes a new CSV referencing them.

Each multi-view case produces four .pt files: CC image, MLO image, CC mask,
and MLO mask. The resulting *_pt.csv is detected automatically by all
multi-view training and evaluation scripts.

Run from the project root:
    python -m src.data.preprocess_multi_view_to_pt

Output:
    data_processed/pt/mv_images/<idx>_cc.pt
    data_processed/pt/mv_images/<idx>_mlo.pt
    data_processed/pt/mv_masks/<idx>_cc.pt
    data_processed/pt/mv_masks/<idx>_mlo.pt
    data_processed/indexed_multi_view_cases_pt.csv
"""

import pandas as pd
import torch
import pydicom
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


TARGET_SIZE = 512
CSV_IN      = Path("data_processed/indexed_multi_view_cases.csv")
CSV_OUT     = Path("data_processed/indexed_multi_view_cases_pt.csv")
OUT_DIR     = Path("data_processed/pt")

IMG_DIR  = OUT_DIR / "mv_images"
MASK_DIR = OUT_DIR / "mv_masks"


def load_dicom_image(path: Path, size: int) -> torch.Tensor:
    """Decode a DICOM mammogram and return a (1, H, W) float32 tensor in [0, 1]."""
    ds  = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)

    # Correct for MONOCHROME1 (inverted) photometric interpretation
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
    Load the binary ROI mask associated with a given scan path.

    Returns a blank tensor when the mask folder does not exist, which is
    expected for some benign cases in CBIS-DDSM.
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
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask -= mask.min()
    if mask.max() > 0:
        mask /= mask.max()

    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_IN)
    print(f"Total multi-view cases: {len(df)}")

    cc_pt_paths       = []
    mlo_pt_paths      = []
    cc_mask_pt_paths  = []
    mlo_mask_pt_paths = []
    failed_indices    = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        cc_img_out   = IMG_DIR  / f"{idx}_cc.pt"
        mlo_img_out  = IMG_DIR  / f"{idx}_mlo.pt"
        cc_mask_out  = MASK_DIR / f"{idx}_cc.pt"
        mlo_mask_out = MASK_DIR / f"{idx}_mlo.pt"

        # Resume safely if the script is interrupted mid-run
        if (
            cc_img_out.exists() and mlo_img_out.exists()
            and cc_mask_out.exists() and mlo_mask_out.exists()
        ):
            cc_pt_paths.append(str(cc_img_out))
            mlo_pt_paths.append(str(mlo_img_out))
            cc_mask_pt_paths.append(str(cc_mask_out))
            mlo_mask_pt_paths.append(str(mlo_mask_out))
            continue

        try:
            cc_path  = Path(row["cc_path"])
            mlo_path = Path(row["mlo_path"])

            cc_image  = load_dicom_image(cc_path,  TARGET_SIZE)
            mlo_image = load_dicom_image(mlo_path, TARGET_SIZE)
            cc_mask   = load_roi_mask(cc_path,  TARGET_SIZE)
            mlo_mask  = load_roi_mask(mlo_path, TARGET_SIZE)

            torch.save(cc_image,  cc_img_out)
            torch.save(mlo_image, mlo_img_out)
            torch.save(cc_mask,   cc_mask_out)
            torch.save(mlo_mask,  mlo_mask_out)

            cc_pt_paths.append(str(cc_img_out))
            mlo_pt_paths.append(str(mlo_img_out))
            cc_mask_pt_paths.append(str(cc_mask_out))
            mlo_mask_pt_paths.append(str(mlo_mask_out))

        except Exception as e:
            print(f"\n  [WARN] idx={idx} failed: {e}")
            failed_indices.append(idx)
            cc_pt_paths.append(None)
            mlo_pt_paths.append(None)
            cc_mask_pt_paths.append(None)
            mlo_mask_pt_paths.append(None)

    df["cc_pt_path"]       = cc_pt_paths
    df["mlo_pt_path"]      = mlo_pt_paths
    df["cc_mask_pt_path"]  = cc_mask_pt_paths
    df["mlo_mask_pt_path"] = mlo_mask_pt_paths

    before = len(df)
    df = df.dropna(subset=["cc_pt_path", "mlo_pt_path", "cc_mask_pt_path", "mlo_mask_pt_path"])
    if before - len(df) > 0:
        print(f"\n  Dropped {before - len(df)} failed rows.")

    df.to_csv(CSV_OUT, index=False)
    print(f"\nDone. Saved {len(df)} rows to {CSV_OUT}")
    print(f"Preprocessed tensors: {OUT_DIR}")


if __name__ == "__main__":
    main()
