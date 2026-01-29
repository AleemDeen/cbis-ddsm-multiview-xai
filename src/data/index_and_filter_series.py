from pathlib import Path
import pandas as pd
import pydicom

from src.utils.config import load_config


def safe_str(x):
    return str(x) if x is not None else ""


def infer_view_from_text(text: str) -> str:
    t = text.upper()
    if "_CC" in t or " CC" in t:
        return "CC"
    if "_MLO" in t or " MLO" in t:
        return "MLO"
    return ""


def infer_laterality_from_text(text: str) -> str:
    t = text.upper()
    if "_LEFT_" in t or " LEFT" in t or t.endswith("_LEFT"):
        return "LEFT"
    if "_RIGHT_" in t or " RIGHT" in t or t.endswith("_RIGHT"):
        return "RIGHT"
    return ""


def is_full_mammogram(series_desc: str) -> bool:
    s = (series_desc or "").lower()

    # CBIS-DDSM uses "full mammogram images"
    if "full mammogram" in s:
        return True

    # Exclude derived / non-image series
    if any(x in s for x in ["cropped", "mask", "roi"]):
        return False

    return False


def main():
    cfg = load_config()
    raw_dir = Path(cfg["RAW_DATA_DIR"])
    out_dir = Path(cfg["PROCESSED_DATA_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather candidate files (.dcm or no extension)
    files = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".dcm", "")
    ]

    rows = []
    for fp in files:
        try:
            ds = pydicom.dcmread(
                str(fp),
                stop_before_pixels=True,
                force=True
            )

            rows.append({
                "file_path": str(fp),
                "file_size_bytes": fp.stat().st_size,
                "case_folder": fp.parent.name,

                "patient_id": safe_str(getattr(ds, "PatientID", None)),
                "study_uid": safe_str(getattr(ds, "StudyInstanceUID", None)),
                "series_uid": safe_str(getattr(ds, "SeriesInstanceUID", None)),
                "modality": safe_str(getattr(ds, "Modality", None)),
                "series_description": safe_str(getattr(ds, "SeriesDescription", None)),
                "view_position": safe_str(getattr(ds, "ViewPosition", None)),
                "image_laterality": safe_str(getattr(ds, "ImageLaterality", None)),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("No DICOMs indexed. Check RAW_DATA_DIR.")
        return

    # Fallback text for inference
    df["path_text"] = df["file_path"].str.upper()

    # View
    df["view"] = df["view_position"].str.upper()
    df.loc[~df["view"].isin(["CC", "MLO"]), "view"] = (
        df.loc[~df["view"].isin(["CC", "MLO"]), "path_text"]
        .apply(infer_view_from_text)
    )

    # Laterality
    df["laterality"] = (
        df["image_laterality"]
        .str.upper()
        .replace({"L": "LEFT", "R": "RIGHT"})
    )
    df.loc[~df["laterality"].isin(["LEFT", "RIGHT"]), "laterality"] = (
        df.loc[~df["laterality"].isin(["LEFT", "RIGHT"]), "path_text"]
        .apply(infer_laterality_from_text)
    )

    # Full mammogram filter
    df["is_full_mammogram"] = df["series_description"].apply(is_full_mammogram)

    full_df = df[df["is_full_mammogram"]].copy()

    # Enforce MG modality if present
    if "MG" in set(full_df["modality"]):
        full_df = full_df[
            (full_df["modality"] == "MG") | (full_df["modality"] == "")
        ]

    # Keep only valid images
    full_df = full_df[
        full_df["view"].isin(["CC", "MLO"]) &
        full_df["laterality"].isin(["LEFT", "RIGHT"])
    ].copy()

    # Deduplicate (keep largest file per folder/view)
    full_df = full_df.sort_values("file_size_bytes", ascending=False)
    full_df = full_df.drop_duplicates(
        subset=["case_folder", "view"],
        keep="first"
    )

    # Save final index
    out_csv = out_dir / "indexed_full_mammogram_images.csv"
    full_df.to_csv(out_csv, index=False)

    print("Saved:")
    print("-", out_csv)
    print()
    print("Counts:")
    print("Full mammogram images:", len(full_df))


if __name__ == "__main__":
    main()
