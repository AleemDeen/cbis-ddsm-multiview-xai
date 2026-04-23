from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    CSV_PATH   = Path("data_processed/indexed_full_mammogram_images_with_labels.csv")
    SPLITS_DIR = Path("splits")
    SPLITS_DIR.mkdir(exist_ok=True)

    SEED        = 42
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

    df = pd.read_csv(CSV_PATH)

    # Extract the base case ID (P_XXXXX) to group all images belonging to a patient
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    # Derive a single label per patient: if any mammogram in the case is malignant,
    # the patient is considered malignant. This conservative rule prevents the
    # stratification from being skewed by the number of images per patient.
    case_labels = (
        df.groupby("case_id")["label"]
        .max()
        .reset_index()
    )

    # Stratified split ensures both splits have the same benign/malignant ratio.
    # With ~850 cases and roughly 60% malignant, unstratified splits could
    # easily produce a test set with a very different class balance.
    train_cases, temp_cases = train_test_split(
        case_labels,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=case_labels["label"],
        random_state=SEED,
    )

    # Split the remaining 30% equally between val and test
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_cases, test_cases = train_test_split(
        temp_cases,
        test_size=(1.0 - val_size),
        stratify=temp_cases["label"],
        random_state=SEED,
    )

    # Write one patient ID per line — these files are the ground truth for all
    # train/val/test splits used across every script in the project
    (SPLITS_DIR / "train_cases.txt").write_text(
        "\n".join(sorted(train_cases["case_id"])) + "\n"
    )
    (SPLITS_DIR / "val_cases.txt").write_text(
        "\n".join(sorted(val_cases["case_id"])) + "\n"
    )
    (SPLITS_DIR / "test_cases.txt").write_text(
        "\n".join(sorted(test_cases["case_id"])) + "\n"
    )

    print("Split sizes (patients):")
    print(f"Train: {len(train_cases)}")
    print(f"Val:   {len(val_cases)}")
    print(f"Test:  {len(test_cases)}")
    print(f"Total: {len(case_labels)}")


if __name__ == "__main__":
    main()
