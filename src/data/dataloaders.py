from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from src.data.mammogram_dataset import CBISDDSMImageDataset


def load_case_ids(path: Path) -> set:
    """Read a plain-text split file and return a set of patient case IDs (P_XXXXX)."""
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def get_transforms(train: bool = True):
    """
    Return data augmentation transforms appropriate for the given split.

    Training augmentations are deliberately conservative — flips, small rotations,
    and subtle brightness/contrast jitter. More aggressive transforms (e.g. elastic
    deformations) risk distorting the fine structures that distinguish benign from
    malignant lesions in mammograms.

    Validation and test sets receive no augmentation so evaluation metrics are
    reproducible and unaffected by random variation.
    """
    if train:
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            # Very small photometric jitter to simulate scanner variability across sites
            v2.ColorJitter(brightness=0.05, contrast=0.05),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
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
    persistent_workers=True,
):
    """
    Build train, validation, and test DataLoaders for the single-view pipeline.

    Splits are performed at patient (case) level using the pre-generated split
    files in splits_dir. Patient-level splitting is critical — without it,
    different mammograms from the same patient could appear in both train and
    test sets, causing data leakage and inflated evaluation metrics.
    """
    csv_path   = Path(csv_path)
    splits_dir = Path(splits_dir)

    df = pd.read_csv(csv_path)
    # Extract the base patient ID (P_XXXXX) used as the split key
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

    train_dataset = CBISDDSMImageDataset(train_df.reset_index(drop=True), transform=get_transforms(train=True))
    val_dataset   = CBISDDSMImageDataset(val_df.reset_index(drop=True),   transform=get_transforms(train=False))
    test_dataset  = CBISDDSMImageDataset(test_df.reset_index(drop=True),  transform=get_transforms(train=False))

    # prefetch_factor and persistent_workers require at least one worker process
    pf = prefetch_factor if num_workers > 0 else None
    pw = persistent_workers if num_workers > 0 else False

    loader_args = {
        "batch_size":         batch_size,
        "num_workers":        num_workers,
        "pin_memory":         pin_memory,
        "prefetch_factor":    pf,
        "persistent_workers": pw,
    }

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_args)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
