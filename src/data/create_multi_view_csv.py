import pandas as pd
from pathlib import Path

INPUT_CSV = "data_processed/indexed_full_mammogram_images_with_labels.csv"
OUTPUT_CSV = "data_processed/indexed_multi_view_cases.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    # Keep only full mammograms
    df = df[df["is_full_mammogram"].astype(str).str.upper() == "TRUE"]

    # Keep only CC and MLO views
    df = df[df["view"].isin(["CC", "MLO"])]

    multi_view_rows = []

    # Group by patient + breast side
    grouped = df.groupby(["patient_id", "laterality"])

    for (patient_id, laterality), group in grouped:

        cc = group[group["view"] == "CC"]
        mlo = group[group["view"] == "MLO"]

        # Require exactly one of each
        if len(cc) >= 1 and len(mlo) >= 1:
            cc_row = cc.iloc[0]
            mlo_row = mlo.iloc[0]

            # Sanity check: labels should match
            if cc_row["label"] != mlo_row["label"]:
                continue

            multi_view_rows.append({
                "patient_id": patient_id,
                "laterality": laterality,
                "cc_path": cc_row["file_path"],
                "mlo_path": mlo_row["file_path"],
                "label": cc_row["label"]
            })

    multi_df = pd.DataFrame(multi_view_rows)

    print("Total multi-view samples:", len(multi_df))
    print(multi_df["label"].value_counts())

    multi_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()