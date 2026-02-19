from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2  # Modern V2 transforms
from src.data.mammogram_dataset import CBISDDSMImageDataset

def load_case_ids(path: Path):
    """Load case-level patient IDs (P_XXXXX) from split file."""
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def get_transforms(train=True):
    """
    Returns medical-grade augmentations for training.
    Validation/Test only get basic tensor conversion.
    """
    if train:
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            # Tiny brightness/contrast nudge helps generalize across different machines
            v2.ColorJitter(brightness=0.05, contrast=0.05),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        # Validation and Testing should NEVER be augmented
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])

def build_dataloaders(
    csv_path,
    splits_dir,
    batch_size=8,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
):
    csv_path = Path(csv_path)
    splits_dir = Path(splits_dir)

    # ------------------------------
    # Load and Split Logic
    # ------------------------------
    df = pd.read_csv(csv_path)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    if df["case_id"].isna().any():
        raise ValueError("Some rows could not extract case_id from patient_id")

    train_cases = load_case_ids(splits_dir / "train_cases.txt")
    val_cases   = load_case_ids(splits_dir / "val_cases.txt")
    test_cases  = load_case_ids(splits_dir / "test_cases.txt")

    train_df = df[df["case_id"].isin(train_cases)]
    val_df   = df[df["case_id"].isin(val_cases)]
    test_df  = df[df["case_id"].isin(test_cases)]

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One or more dataset splits are empty")

    # ------------------------------
    # Apply Transforms
    # ------------------------------
    train_dataset = CBISDDSMImageDataset(
        train_df.reset_index(drop=True), 
        transform=get_transforms(train=True)
    )
    val_dataset = CBISDDSMImageDataset(
        val_df.reset_index(drop=True), 
        transform=get_transforms(train=False)
    )
    test_dataset = CBISDDSMImageDataset(
        test_df.reset_index(drop=True), 
        transform=get_transforms(train=False)
    )

    # Safety check: prefetch_factor requires num_workers > 0
    pf = prefetch_factor if num_workers > 0 else None
    pw = persistent_workers if num_workers > 0 else False

    # ------------------------------
    # Build loaders
    # ------------------------------
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": pf,
        "persistent_workers": pw
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader