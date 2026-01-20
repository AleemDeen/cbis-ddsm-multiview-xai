from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.mammogram_dataset import CBISDDSMImageDataset


def load_case_ids(path):
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def extract_case_id(patient_id: str) -> str:
    if patient_id.endswith("_CC"):
        return patient_id[:-3]
    if patient_id.endswith("_MLO"):
        return patient_id[:-4]
    return patient_id


def filter_df_by_cases(df, case_ids):
    df = df.copy()
    df["case_id"] = df["patient_id"].apply(extract_case_id)
    return df[df["case_id"].isin(case_ids)]


def build_dataloaders(
    csv_path,
    splits_dir,
    batch_size=8,
    num_workers=2,
    pin_memory=True,
):
    csv_path = Path(csv_path)
    splits_dir = Path(splits_dir)

    df = pd.read_csv(csv_path)

    train_cases = load_case_ids(splits_dir / "train_cases.txt")
    val_cases = load_case_ids(splits_dir / "val_cases.txt")
    test_cases = load_case_ids(splits_dir / "test_cases.txt")

    train_df = filter_df_by_cases(df, train_cases)
    val_df = filter_df_by_cases(df, val_cases)
    test_df = filter_df_by_cases(df, test_cases)

    train_dataset = CBISDDSMImageDataset(
        csv_path=train_df,
    )
    val_dataset = CBISDDSMImageDataset(
        csv_path=val_df,
    )
    test_dataset = CBISDDSMImageDataset(
        csv_path=test_df,
    )

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
