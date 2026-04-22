"""
Generates ground-truth ROI mask overlay PNGs for all example patients.

For each patient × view (CC / MLO) the script produces:
  - <patient>_<view>_original.png      — plain mammogram
  - <patient>_<view>_roi_overlay.png   — mammogram + ground-truth ROI mask

Run from project root:
    python export_example_masks.py
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
OUT_DIR  = ROOT / "examples"
CSV_PATH = ROOT / "data_processed" / "indexed_multi_view_cases_pt.csv"

PATIENTS = ["P_00001", "P_00004", "P_00005", "P_00007", "P_00009"]

# Colour for the ROI contour / fill overlay (green)
ROI_COLOUR = (0, 220, 80)   # RGB


# ── helpers ───────────────────────────────────────────────────────────────────

def load_dicom(path: Path, size: int = 512):
    """Load a DICOM as a float32 [0,1] array. Returns (img, orig_shape)."""
    ds  = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    orig_shape = img.shape  # (H, W) before resize
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img, orig_shape


def load_mask_from_dicom(scan_path: Path, orig_shape: tuple, size: int = 512) -> np.ndarray:
    """
    Load the binary ROI mask directly from the CBIS-DDSM DICOM folder.

    Selects the correct DICOM by choosing the file with the lowest pixel
    coverage (the true binary mask, not a cropped patch image). Resizes the
    mask to match the mammogram's original dimensions before downsampling to
    size×size, ensuring spatial alignment.
    """
    scan_folder = scan_path.parents[2].name
    roi_root    = scan_path.parents[3] / (scan_folder + "_1")

    if not roi_root.exists():
        return np.zeros((size, size), dtype=np.float32)

    dcm_files = sorted(roi_root.glob("*/*/*.dcm"))
    if not dcm_files:
        return np.zeros((size, size), dtype=np.float32)

    # Pick the DICOM with the lowest nonzero-pixel coverage — that is the
    # binary mask. A cropped patch image fills most of its pixels.
    best_mask     = None
    best_coverage = float("inf")
    for dcm_file in dcm_files:
        ds     = pydicom.dcmread(str(dcm_file))
        pixels = ds.pixel_array.astype(np.float32)
        pmax   = pixels.max()
        if pmax == 0:
            continue
        coverage = (pixels > pmax * 0.5).mean()
        if coverage < best_coverage:
            best_coverage = coverage
            best_mask     = pixels

    if best_mask is None:
        return np.zeros((size, size), dtype=np.float32)

    # Align mask to mammogram's original dimensions.
    # Padding is used instead of stretch-resize: stretching distorts spatial
    # positions when dimensions differ slightly, causing systematic offsets.
    mam_h, mam_w = orig_shape
    mask_h, mask_w = best_mask.shape
    print(f"    mammogram: {mam_h}×{mam_w}  |  mask: {mask_h}×{mask_w}")
    if (mask_h, mask_w) != (mam_h, mam_w):
        # Pad smaller dimension with zeros to match mammogram canvas size
        pad_h = max(mam_h - mask_h, 0)
        pad_w = max(mam_w - mask_w, 0)
        best_mask = np.pad(best_mask,
                           ((0, pad_h), (0, pad_w)),
                           mode="constant", constant_values=0)
        # If mask is larger, crop to mammogram size
        best_mask = best_mask[:mam_h, :mam_w]

    mask = cv2.resize(best_mask, (size, size), interpolation=cv2.INTER_NEAREST)
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

            # Load mammogram from the ORIGINAL path in the CSV — this guarantees
            # the same coordinate space as the mask DICOM (also derived from
            # this path). The examples/ copy may have different dimensions.
            scan_col  = f"{view_l}_path"
            scan_path = Path(row[scan_col])

            if not scan_path.exists():
                print(f"  [skip] original DICOM not found: {scan_path}")
                continue

            img, orig_shape = load_dicom(scan_path)
            mask = load_mask_from_dicom(scan_path, orig_shape)

            # Benign cases have ROI annotations too (benign findings) — include them
            has_roi = mask.max() > 0.1

            stem = f"{pid}_{lat}_{dx}_{view}"
            save_png(make_original_png(img),             OUT_DIR / f"{stem}_original.png")
            save_png(make_overlay_png(img, mask, has_roi), OUT_DIR / f"{stem}_roi_overlay.png")

        print()

    print(f"Done — {len(list(OUT_DIR.glob('*.png')))} PNG(s) in {OUT_DIR}/")


if __name__ == "__main__":
    main()
