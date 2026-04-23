"""
DICOM indexing script: walks the raw CBIS-DDSM folder tree, reads DICOM
headers, infers view (CC/MLO) and laterality (LEFT/RIGHT) from metadata,
and writes a CSV of full mammogram image paths.

This is the first step in the data pipeline. The output CSV is then joined
with the case-level pathology labels in build_case_level_labels.py.

Run from the project root:
    python -m src.data.index_and_filter_series
"""

from pathlib import Path
import pandas as pd
import pydicom

from src.utils.config import load_config


def safe_str(x) -> str:
    """Return str(x) or an empty string if x is None."""
    return str(x) if x is not None else ""


def infer_view_from_text(text: str) -> str:
    """
    Infer CC or MLO view from the file path when the DICOM ViewPosition tag is absent.

    CBIS-DDSM folder names reliably encode the view (e.g. Calc-Train_P_00001_LEFT_CC),
    so falling back to path-text parsing recovers the view for most missing tags.
    """
    t = text.upper()
    if "_CC" in t or " CC" in t:
        return "CC"
    if "_MLO" in t or " MLO" in t:
        return "MLO"
    return ""


def infer_laterality_from_text(text: str) -> str:
    """
    Infer LEFT or RIGHT laterality from the file path when the DICOM tag is absent.
    """
    t = text.upper()
    if "_LEFT_" in t or " LEFT" in t or t.endswith("_LEFT"):
        return "LEFT"
    if "_RIGHT_" in t or " RIGHT" in t or t.endswith("_RIGHT"):
        return "RIGHT"
    return ""


def is_full_mammogram(series_desc: str) -> bool:
    """
    Return True if the series description indicates a full mammogram image.

    CBIS-DDSM uses the description "full mammogram images" for whole-breast
    acquisitions. ROI crops and mask series are explicitly excluded so they
    do not contaminate the image index.
    """
    s = (series_desc or "").lower()
    if "full mammogram" in s:
        return True
    if any(x in s for x in ["cropped", "mask", "roi"]):
        return False
    return False


def main():
    cfg     = load_config()
    raw_dir = Path(cfg["RAW_DATA_DIR"])
    out_dir = Path(cfg["PROCESSED_DATA_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather all candidate DICOM files — CBIS-DDSM uses .dcm extensions throughout
    files = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".dcm", "")
    ]

    rows = []
    for fp in files:
        try:
            # stop_before_pixels=True reads only the header, which is much faster
            # than loading the full pixel data for every file in the tree
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)

            rows.append({
                "file_path":          str(fp),
                "file_size_bytes":    fp.stat().st_size,
                "case_folder":        fp.parent.name,
                "patient_id":         safe_str(getattr(ds, "PatientID", None)),
                "study_uid":          safe_str(getattr(ds, "StudyInstanceUID", None)),
                "series_uid":         safe_str(getattr(ds, "SeriesInstanceUID", None)),
                "modality":           safe_str(getattr(ds, "Modality", None)),
                "series_description": safe_str(getattr(ds, "SeriesDescription", None)),
                "view_position":      safe_str(getattr(ds, "ViewPosition", None)),
                "image_laterality":   safe_str(getattr(ds, "ImageLaterality", None)),
            })
        except Exception:
            # Skip files that cannot be read as DICOMs (e.g. metadata .txt files)
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("No DICOMs indexed. Check RAW_DATA_DIR in configs/base.yaml.")
        return

    # Upper-case path text used as fallback for view and laterality inference
    df["path_text"] = df["file_path"].str.upper()

    # View: prefer the DICOM ViewPosition tag; fall back to path-text parsing
    df["view"] = df["view_position"].str.upper()
    df.loc[~df["view"].isin(["CC", "MLO"]), "view"] = (
        df.loc[~df["view"].isin(["CC", "MLO"]), "path_text"]
        .apply(infer_view_from_text)
    )

    # Laterality: prefer the ImageLaterality tag; normalise L/R abbreviations
    df["laterality"] = (
        df["image_laterality"]
        .str.upper()
        .replace({"L": "LEFT", "R": "RIGHT"})
    )
    df.loc[~df["laterality"].isin(["LEFT", "RIGHT"]), "laterality"] = (
        df.loc[~df["laterality"].isin(["LEFT", "RIGHT"]), "path_text"]
        .apply(infer_laterality_from_text)
    )

    df["is_full_mammogram"] = df["series_description"].apply(is_full_mammogram)

    full_df = df[df["is_full_mammogram"]].copy()

    # Keep only MG modality images when that tag is present in the dataset
    if "MG" in set(full_df["modality"]):
        full_df = full_df[(full_df["modality"] == "MG") | (full_df["modality"] == "")]

    # Retain only images with known view and laterality
    full_df = full_df[
        full_df["view"].isin(["CC", "MLO"]) &
        full_df["laterality"].isin(["LEFT", "RIGHT"])
    ].copy()

    # Deduplicate: keep the largest file per (case folder, view) combination.
    # CBIS-DDSM occasionally duplicates series; the largest file is typically
    # the full-resolution acquisition rather than a thumbnail.
    full_df = full_df.sort_values("file_size_bytes", ascending=False)
    full_df = full_df.drop_duplicates(subset=["case_folder", "view"], keep="first")

    out_csv = out_dir / "indexed_full_mammogram_images.csv"
    full_df.to_csv(out_csv, index=False)

    print("Saved:")
    print("-", out_csv)
    print()
    print("Counts:")
    print("Full mammogram images:", len(full_df))


if __name__ == "__main__":
    main()
