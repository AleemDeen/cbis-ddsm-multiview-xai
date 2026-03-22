"""
Generates ground-truth ROI mask overlay PNGs for all example patients.

For each patient × view (CC / MLO) the script produces:
  - <patient>_<view>_original.png      — plain mammogram
  - <patient>_<view>_roi_overlay.png   — mammogram + ground-truth ROI mask

Run from project root:
    python export_example_masks.py
"""

import io
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
OUT_DIR  = ROOT / "examples"
CSV_PATH = ROOT / "data_processed" / "indexed_multi_view_cases_pt.csv"

PATIENTS = ["P_00001", "P_00004", "P_00005", "P_00007", "P_00009"]

# Colour for the ROI contour / fill overlay (green)
ROI_COLOUR = (0, 220, 80)   # RGB


# ── helpers ───────────────────────────────────────────────────────────────────

def load_dicom(path: Path, size: int = 512) -> np.ndarray:
    """Load a DICOM as a float32 [0,1] array."""
    ds  = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


def load_mask(path: Path, size: int = 512) -> np.ndarray:
    """Load a .pt mask tensor as a float32 [0,1] numpy array."""
    t = torch.load(str(path), map_location="cpu", weights_only=True)
    if t.dim() == 3:          # (1, H, W)
        t = t.squeeze(0)
    mask = t.numpy().astype(np.float32)
    if mask.shape != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask -= mask.min()
    if mask.max() > 0:
        mask /= mask.max()
    return mask


def save_png(arr: np.ndarray, path: Path):
    Image.fromarray(arr).save(path)
    print(f"  -> {path.name}")


def make_original_png(img: np.ndarray) -> np.ndarray:
    """Grayscale mammogram as uint8 RGB."""
    g = (img * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def make_overlay_png(img: np.ndarray, mask: np.ndarray,
                     has_roi: bool) -> np.ndarray:
    """
    Overlay the ground-truth ROI mask on the mammogram.

    - Semi-transparent green fill inside the ROI region.
    - Bright green contour drawn around the ROI boundary.
    - If the mask is empty (benign with no annotated ROI) a text label is added.
    """
    rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Binarise mask at 0.5
    binary = (mask >= 0.5).astype(np.uint8)

    if has_roi and binary.sum() > 0:
        # Semi-transparent green fill
        fill        = np.zeros_like(rgb)
        fill[:, :] = ROI_COLOUR
        alpha_fill  = 0.30
        mask3       = binary[:, :, np.newaxis].astype(np.float32)
        rgb         = (rgb * (1 - alpha_fill * mask3) + fill * alpha_fill * mask3).astype(np.uint8)

        # Contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, ROI_COLOUR, 3)

        # Small legend
        cv2.putText(rgb, "Ground-truth ROI", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOUR, 2, cv2.LINE_AA)
    else:
        label = "No ROI annotation (benign)" if not has_roi else "Mask empty"
        cv2.putText(rgb, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    return rgb


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_PATH)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")
    df = df[df["case_id"].isin(PATIENTS)].set_index("case_id")

    print(f"Generating ROI overlays -> {OUT_DIR}/\n")

    for pid in PATIENTS:
        row    = df.loc[pid]
        label  = int(row["label"])         # 1 = malignant, 0 = benign
        lat    = row["laterality"].upper()
        dx     = "MALIGNANT" if label == 1 else "BENIGN"

        print(f"{pid}  {lat}  {dx}")

        for view in ("CC", "MLO"):
            view_l = view.lower()

            # Find the DICOM in examples/
            dcm_glob = list(OUT_DIR.glob(f"{pid}*{view}*.dcm"))
            if not dcm_glob:
                print(f"  [skip] no DICOM found for {pid} {view}")
                continue
            dcm_path = dcm_glob[0]

            # Mask path from CSV
            mask_col  = f"{view_l}_mask_pt_path"
            mask_path = ROOT / row[mask_col]

            img  = load_dicom(dcm_path)
            mask = load_mask(mask_path) if mask_path.exists() else np.zeros((512, 512), np.float32)

            # Benign cases have ROI annotations too (benign findings) — include them
            has_roi = mask_path.exists() and mask.max() > 0.1

            stem = f"{pid}_{lat}_{dx}_{view}"
            save_png(make_original_png(img),             OUT_DIR / f"{stem}_original.png")
            save_png(make_overlay_png(img, mask, has_roi), OUT_DIR / f"{stem}_roi_overlay.png")

        print()

    print(f"Done — {len(list(OUT_DIR.glob('*.png')))} PNG(s) in {OUT_DIR}/")


if __name__ == "__main__":
    main()
