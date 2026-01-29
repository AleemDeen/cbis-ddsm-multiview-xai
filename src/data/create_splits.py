from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.config import load_config

RANDOM_SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main():
    np.random.seed(RANDOM_SEED)

    cfg = load_config()

    # IMPORTANT: use the LABELLED CSV
    data_csv = Path(cfg["PROCESSED_DATA_DIR"]) / "indexed_full_mammogram_images_with_labels.csv"

    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_csv)

    # --------------------------------------------------
    # FORCE patient-level IDs (P_XXXXX)
    # --------------------------------------------------
    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.extract(r"(P_\d+)")
    )

    # --------------------------------------------------
    # Patient-level splitting (NOT lesion-level)
    # --------------------------------------------------
    cases = df["patient_id"].unique()
    np.random.shuffle(cases)

    n_total = len(cases)
    n_train = int(TRAIN_FRAC * n_total)
    n_val = int(VAL_FRAC * n_total)

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    # --------------------------------------------------
    # Write split files
    # --------------------------------------------------
    (splits_dir / "train_cases.txt").write_text("\n".join(train_cases))
    (splits_dir / "val_cases.txt").write_text("\n".join(val_cases))
    (splits_dir / "test_cases.txt").write_text("\n".join(test_cases))

    print("Split sizes (patients):")
    print("Train:", len(train_cases))
    print("Val:", len(val_cases))
    print("Test:", len(test_cases))
    print("Total:", n_total)


if __name__ == "__main__":
    main()
