"""
Diagnostic script — prints DICOM dimensions, pixel spacing, and mask centroid
for each example patient to identify the root cause of mask misalignment.

Run from project root:
    python diagnose_mask_alignment.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pydicom

ROOT     = Path(__file__).parent
CSV_PATH = ROOT / "data_processed" / "indexed_multi_view_cases_pt.csv"
PATIENTS = ["P_00001", "P_00004", "P_00005", "P_00007", "P_00009"]


def dicom_info(path: Path) -> dict:
    ds  = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    ps  = getattr(ds, "PixelSpacing", None)
    return {
        "path":    path.name,
        "shape":   arr.shape,           # (H, W)
        "dtype":   str(arr.dtype),
        "min":     arr.min(),
        "max":     arr.max(),
        "spacing": [round(float(v), 4) for v in ps] if ps else None,
        "photo":   getattr(ds, "PhotometricInterpretation", "?"),
    }


def find_mask_dicoms(scan_path: Path):
    scan_folder = scan_path.parents[2].name
    roi_root    = scan_path.parents[3] / (scan_folder + "_1")
    if not roi_root.exists():
        return [], roi_root
    return sorted(roi_root.glob("*/*/*.dcm")), roi_root


def main():
    df = pd.read_csv(CSV_PATH)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")
    df = df[df["case_id"].isin(PATIENTS)].set_index("case_id")

    for pid in PATIENTS:
        row = df.loc[pid]
        print(f"\n{'='*60}")
        print(f"  {pid}  label={row['label']}  lat={row['laterality']}")
        print(f"{'='*60}")

        for view in ("CC", "MLO"):
            view_l    = view.lower()
            scan_path = Path(row[f"{view_l}_path"])
            print(f"\n  [{view}]  {scan_path.name}")

            # ── Mammogram DICOM ──────────────────────────────────────────
            if scan_path.exists():
                info = dicom_info(scan_path)
                print(f"    Mammogram  shape={info['shape']}  "
                      f"spacing={info['spacing']}  "
                      f"photo={info['photo']}")
            else:
                print(f"    Mammogram  NOT FOUND: {scan_path}")
                continue

            # ── Mask DICOMs ──────────────────────────────────────────────
            dcm_files, roi_root = find_mask_dicoms(scan_path)
            if not dcm_files:
                print(f"    Mask folder not found or empty: {roi_root}")
                continue

            print(f"    Mask folder: {roi_root.name}  ({len(dcm_files)} DICOM(s))")
            for dcm in dcm_files:
                info  = dicom_info(dcm)
                arr   = pydicom.dcmread(str(dcm)).pixel_array.astype(np.float32)
                pmax  = arr.max()
                cov   = float((arr > pmax * 0.5).mean()) if pmax > 0 else 0.0

                # Centroid of bright pixels
                binary = arr > pmax * 0.5
                if binary.any():
                    ys, xs = np.where(binary)
                    cx, cy = xs.mean() / info['shape'][1], ys.mean() / info['shape'][0]
                    centroid = f"centroid=({cx:.3f}, {cy:.3f}) [normalised x,y]"
                else:
                    centroid = "centroid=N/A (empty)"

                print(f"      {dcm.name}  shape={info['shape']}  "
                      f"coverage={cov:.3f}  spacing={info['spacing']}  "
                      f"{centroid}")

            # ── Preprocessed .pt mask ────────────────────────────────────
            import torch
            pt_col  = f"{view_l}_mask_pt_path"
            pt_path = ROOT / row[pt_col]
            if pt_path.exists():
                t = torch.load(str(pt_path), map_location="cpu", weights_only=True)
                if t.dim() == 3:
                    t = t.squeeze(0)
                arr   = t.numpy()
                binary = arr > 0.5
                if binary.any():
                    ys, xs = np.where(binary)
                    cx, cy = xs.mean() / arr.shape[1], ys.mean() / arr.shape[0]
                    centroid = f"centroid=({cx:.3f}, {cy:.3f}) [normalised x,y]"
                else:
                    centroid = "centroid=N/A (empty)"
                print(f"    .pt mask   shape={arr.shape}  "
                      f"coverage={binary.mean():.3f}  {centroid}")
            else:
                print(f"    .pt mask   NOT FOUND")

    print("\nDone.")


if __name__ == "__main__":
    main()
