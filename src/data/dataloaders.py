from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from src.data.mammogram_dataset import CBISDDSMImageDataset


def load_case_ids(path: Path):
    """Load case-level patient IDs (P_XXXXX) from split file."""
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def build_dataloaders(
    csv_path,
    splits_dir,
    batch_size=8,
    num_workers=2,
    pin_memory=True,
):
    csv_path = Path(csv_path)
    splits_dir = Path(splits_dir)

    # ------------------------------
    # Load CSV
    # ------------------------------
    df = pd.read_csv(csv_path)

    print("CSV rows before split:", len(df))
    print("CSV patient_id examples:", df["patient_id"].head(5).tolist())

    # ------------------------------
    # Extract case-level ID (P_XXXXX)
    # ------------------------------
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    if df["case_id"].isna().any():
        raise ValueError("Some rows could not extract case_id from patient_id")

    # ------------------------------
    # Load split case IDs
    # ------------------------------
    train_cases = load_case_ids(splits_dir / "train_cases.txt")
    val_cases   = load_case_ids(splits_dir / "val_cases.txt")
    test_cases  = load_case_ids(splits_dir / "test_cases.txt")

    print("First 5 train case IDs:", list(train_cases)[:5])

    # ------------------------------
    # Apply patient-level split
    # ------------------------------
    train_df = df[df["case_id"].isin(train_cases)]
    val_df   = df[df["case_id"].isin(val_cases)]
    test_df  = df[df["case_id"].isin(test_cases)]

    print("CSV rows after train split:", len(train_df))
    print("CSV rows after val split:", len(val_df))
    print("CSV rows after test split:", len(test_df))

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One or more dataset splits are empty")

    # ------------------------------
    # Build datasets
    # ------------------------------
    train_dataset = CBISDDSMImageDataset(train_df.reset_index(drop=True))
    val_dataset   = CBISDDSMImageDataset(val_df.reset_index(drop=True))
    test_dataset  = CBISDDSMImageDataset(test_df.reset_index(drop=True))

    # ------------------------------
    # Build loaders
    # ------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
