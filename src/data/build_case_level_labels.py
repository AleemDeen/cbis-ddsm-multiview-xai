import pandas as pd

# -----------------------------
# Paths
# -----------------------------
MAMMO_CSV = "data_processed/indexed_full_mammogram_images.csv"

CASE_CSVS = [
    "data_raw/cbis_ddsm_metadata/calc_case_description_train_set.csv",
    "data_raw/cbis_ddsm_metadata/calc_case_description_test_set.csv",
    "data_raw/cbis_ddsm_metadata/mass_case_description_train_set.csv",
    "data_raw/cbis_ddsm_metadata/mass_case_description_test_set.csv",
]

OUT_CSV = "data_processed/indexed_full_mammogram_images_with_labels.csv"


# -----------------------------
# 1. Load mammogram index
# -----------------------------
df_mammo = pd.read_csv(MAMMO_CSV)

# Extract core patient ID, e.g. P_00005
df_mammo["patient_id"] = (
    df_mammo["patient_id"]
    .astype(str)
    .str.extract(r"(P_\d+)")
)



# -----------------------------
# 2. Load and combine lesion metadata
# -----------------------------
dfs = []
for path in CASE_CSVS:
    df = pd.read_csv(path)
    dfs.append(df)

df_cases = pd.concat(dfs, ignore_index=True)

df_cases = df_cases[["patient_id", "pathology"]].copy()
df_cases["patient_id"] = df_cases["patient_id"].astype(str)


# -----------------------------
# 3. Convert pathology → binary label
# -----------------------------
def pathology_to_label(x):
    return 1 if "MALIGNANT" in str(x).upper() else 0

df_cases["label"] = df_cases["pathology"].apply(pathology_to_label)


# -----------------------------
# 4. Collapse lesion → case level
# -----------------------------
# If ANY lesion is malignant, case is malignant
df_case_labels = (
    df_cases
    .groupby("patient_id")["label"]
    .max()
    .reset_index()
)


# -----------------------------
# 5. Merge labels into mammograms
# -----------------------------
df_final = df_mammo.merge(
    df_case_labels,
    on="patient_id",
    how="left"
)


# -----------------------------
# 6. Sanity checks
# -----------------------------
print("Images without labels:", df_final["label"].isna().sum())
print("Label distribution:")
print(df_final["label"].value_counts(dropna=False))


# --------------------------------------------------
# Normalise patient_id to match split files (P_XXXXX)
# --------------------------------------------------
df_final["patient_id"] = (
    df_final["patient_id"]
    .astype(str)
    .str.extract(r"(P_\d+)")
)


# -----------------------------
# 7. Save
# -----------------------------
df_final.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
