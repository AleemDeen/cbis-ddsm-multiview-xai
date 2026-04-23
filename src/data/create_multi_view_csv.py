import pandas as pd
from pathlib import Path

INPUT_CSV  = "data_processed/indexed_full_mammogram_images_with_labels.csv"
OUTPUT_CSV = "data_processed/indexed_multi_view_cases.csv"


def main():
    df = pd.read_csv(INPUT_CSV)

    # Retain only full mammogram images — the index also contains ROI crops
    # and other series that are not suitable for whole-image classification
    df = df[df["is_full_mammogram"].astype(str).str.upper() == "TRUE"]

    # The dual-branch model requires exactly one CC and one MLO per breast,
    # so only those two standard views are kept
    df = df[df["view"].isin(["CC", "MLO"])]

    multi_view_rows = []

    # Pair CC and MLO images by patient and breast side (laterality).
    # Each unique (patient_id, laterality) combination is one multi-view case.
    grouped = df.groupby(["patient_id", "laterality"])

    for (patient_id, laterality), group in grouped:
        cc  = group[group["view"] == "CC"]
        mlo = group[group["view"] == "MLO"]

        # Both views are required — skip cases where either is missing
        if len(cc) >= 1 and len(mlo) >= 1:
            cc_row  = cc.iloc[0]
            mlo_row = mlo.iloc[0]

            # Discard cases where CC and MLO carry different labels.
            # This should not happen in CBIS-DDSM but occasional indexing
            # errors in the metadata can cause mismatches.
            if cc_row["label"] != mlo_row["label"]:
                continue

            multi_view_rows.append({
                "patient_id": patient_id,
                "laterality": laterality,
                "cc_path":    cc_row["file_path"],
                "mlo_path":   mlo_row["file_path"],
                "label":      cc_row["label"],
            })

    multi_df = pd.DataFrame(multi_view_rows)

    print("Total multi-view samples:", len(multi_df))
    print(multi_df["label"].value_counts())

    multi_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
