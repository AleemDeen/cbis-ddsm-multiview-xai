from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.config import load_config

RANDOM_SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def extract_case_id(patient_id: str) -> str:
    """
    Strip view from CBIS-DDSM patient_id
    e.g. Calc-Training_P_00182_LEFT_CC -> Calc-Training_P_00182_LEFT
    """
    if patient_id.endswith("_CC"):
        return patient_id[:-3]
    if patient_id.endswith("_MLO"):
        return patient_id[:-4]
    return patient_id


def main():
    np.random.seed(RANDOM_SEED)

    cfg = load_config()
    data_csv = Path(cfg["PROCESSED_DATA_DIR"]) / "indexed_full_mammogram_images.csv"
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_csv)
    df["case_id"] = df["patient_id"].apply(extract_case_id)

    cases = df["case_id"].unique()
    np.random.shuffle(cases)

    n_total = len(cases)
    n_train = int(TRAIN_FRAC * n_total)
    n_val = int(VAL_FRAC * n_total)

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    (splits_dir / "train_cases.txt").write_text("\n".join(train_cases))
    (splits_dir / "val_cases.txt").write_text("\n".join(val_cases))
    (splits_dir / "test_cases.txt").write_text("\n".join(test_cases))

    print("Split sizes (cases):")
    print("Train:", len(train_cases))
    print("Val:", len(val_cases))
    print("Test:", len(test_cases))
    print("Total:", n_total)


if __name__ == "__main__":
    main()
