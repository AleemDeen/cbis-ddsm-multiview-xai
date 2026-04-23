import pandas as pd

# Paths to the raw CBIS-DDSM metadata CSVs and the mammogram image index
MAMMO_CSV = "data_processed/indexed_full_mammogram_images.csv"

# The dataset ships four separate metadata files split by pathology type and
# train/test partition — combining them gives a complete label table
CASE_CSVS = [
    "data_raw/cbis_ddsm_metadata/calc_case_description_train_set.csv",
    "data_raw/cbis_ddsm_metadata/calc_case_description_test_set.csv",
    "data_raw/cbis_ddsm_metadata/mass_case_description_train_set.csv",
    "data_raw/cbis_ddsm_metadata/mass_case_description_test_set.csv",
]

OUT_CSV = "data_processed/indexed_full_mammogram_images_with_labels.csv"


# Load the indexed mammogram image paths and normalise patient IDs
df_mammo = pd.read_csv(MAMMO_CSV)
df_mammo["patient_id"] = (
    df_mammo["patient_id"]
    .astype(str)
    .str.extract(r"(P_\d+)")
)


# Combine all four metadata files into a single lesion-level label table
dfs = [pd.read_csv(path) for path in CASE_CSVS]
df_cases = pd.concat(dfs, ignore_index=True)
df_cases = df_cases[["patient_id", "pathology"]].copy()
df_cases["patient_id"] = df_cases["patient_id"].astype(str)


def pathology_to_label(x):
    """Convert a pathology string to a binary label: 1 = malignant, 0 = benign."""
    return 1 if "MALIGNANT" in str(x).upper() else 0


df_cases["label"] = df_cases["pathology"].apply(pathology_to_label)


# Collapse to one label per patient: malignant if ANY lesion is malignant.
# A patient with both benign and malignant findings is treated as malignant
# because the clinical outcome is determined by the most dangerous lesion.
df_case_labels = (
    df_cases
    .groupby("patient_id")["label"]
    .max()
    .reset_index()
)


# Merge the case-level labels onto the per-image mammogram index
df_final = df_mammo.merge(df_case_labels, on="patient_id", how="left")


# Sanity check — any image without a label indicates a patient not found
# in the metadata CSVs (typically a data download issue)
print("Images without labels:", df_final["label"].isna().sum())
print("Label distribution:")
print(df_final["label"].value_counts(dropna=False))


# Ensure the patient_id column matches the P_XXXXX format used in split files
df_final["patient_id"] = (
    df_final["patient_id"]
    .astype(str)
    .str.extract(r"(P_\d+)")
)


df_final.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
