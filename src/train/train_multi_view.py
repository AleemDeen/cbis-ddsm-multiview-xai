import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from src.models.resnet18_multi_view import ResNet18MultiView
from src.data.multi_view_dataset import CBISDDSMMultiViewDataset


def get_transforms(train=True):
    if train:
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.05, contrast=0.05),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])


def load_case_ids(path):
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def dice_loss(pred, target):
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return 1.0 - (2.0 * intersection / (pred.sum(dim=1) + target.sum(dim=1) + 1e-8)).mean()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, lambda_loc):
    model.train()
    running_loss = running_cls_loss = running_loc_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        cc        = batch["cc_image"].to(device, non_blocking=True)
        mlo       = batch["mlo_image"].to(device, non_blocking=True)
        labels    = batch["label"].float().to(device, non_blocking=True)
        cc_masks  = batch["cc_mask"].to(device, non_blocking=True)
        mlo_masks = batch["mlo_mask"].to(device, non_blocking=True)

        malignant_mask = labels.bool()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            if lambda_loc > 0.0:
                logits, cc_features, mlo_features = model(cc, mlo, return_features=True)
            else:
                logits = model(cc, mlo)

            classification_loss = criterion(logits.view(-1), labels)

        if lambda_loc > 0.0 and malignant_mask.any():
            # GradCAM for CC branch (retain graph for mlo + final backward)
            cc_grads = torch.autograd.grad(
                classification_loss, cc_features,
                retain_graph=True,
                create_graph=False,
            )[0]
            cc_weights = cc_grads.mean(dim=(2, 3), keepdim=True)
            cc_cam = F.relu((cc_weights * cc_features).sum(dim=1, keepdim=True))
            cc_cam = cc_cam / (cc_cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
            cc_cam_up = F.interpolate(cc_cam, size=(512, 512), mode="bilinear", align_corners=False)

            # GradCAM for MLO branch (retain graph for final backward)
            mlo_grads = torch.autograd.grad(
                classification_loss, mlo_features,
                retain_graph=True,
                create_graph=False,
            )[0]
            mlo_weights = mlo_grads.mean(dim=(2, 3), keepdim=True)
            mlo_cam = F.relu((mlo_weights * mlo_features).sum(dim=1, keepdim=True))
            mlo_cam = mlo_cam / (mlo_cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
            mlo_cam_up = F.interpolate(mlo_cam, size=(512, 512), mode="bilinear", align_corners=False)

            cc_loc_loss  = dice_loss(cc_cam_up[malignant_mask],  cc_masks[malignant_mask])
            mlo_loc_loss = dice_loss(mlo_cam_up[malignant_mask], mlo_masks[malignant_mask])
            loc_loss = (cc_loc_loss + mlo_loc_loss) / 2.0
        else:
            loc_loss = torch.tensor(0.0, device=device)

        loss = classification_loss + lambda_loc * loc_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss     += loss.item()
        running_cls_loss += classification_loss.item()
        running_loc_loss += loc_loss.item()
        n += 1

    return running_loss / n, running_cls_loss / n, running_loc_loss / n


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            cc     = batch["cc_image"].to(device, non_blocking=True)
            mlo    = batch["mlo_image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(cc, mlo)
                loss   = criterion(logits.view(-1), labels)

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-loc", type=float, default=0.1,
                        help="Weight for localization loss (0.0 = classification only)")
    args = parser.parse_args()
    lambda_loc = args.lambda_loc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:  {device}")
    print(f"Lambda-loc:    {lambda_loc}")

    PT_CSV   = "data_processed/indexed_multi_view_cases_pt.csv"
    CSV_PATH = PT_CSV if Path(PT_CSV).exists() else "data_processed/indexed_multi_view_cases.csv"
    print(f"CSV:           {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    splits_dir  = Path("splits")
    train_cases = load_case_ids(splits_dir / "train_cases.txt")
    val_cases   = load_case_ids(splits_dir / "val_cases.txt")

    train_df = df[df["case_id"].isin(train_cases)].reset_index(drop=True)
    val_df   = df[df["case_id"].isin(val_cases)].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            "One or more splits are empty — verify patient_id format in the multi-view CSV "
            "matches the P_XXXXX pattern in splits/train_cases.txt and splits/val_cases.txt"
        )

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset = CBISDDSMMultiViewDataset(train_df, transform=get_transforms(train=True))
    val_dataset   = CBISDDSMMultiViewDataset(val_df,   transform=get_transforms(train=False))

    loader_args = dict(
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_args)

    num_pos    = (train_df["label"] == 1).sum()
    num_neg    = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    model     = ResNet18MultiView().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    save_path = f"models/resnet18_multi_view_best_loc{lambda_loc}.pt"

    num_epochs     = 40
    patience       = 10
    best_val_loss  = float("inf")
    epochs_no_improve = 0
    start_time     = time.time()

    print(f"\nStarting Multi-View Training  (save → {save_path})")
    print("-" * 50)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, cls_loss, loc_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, lambda_loc
        )
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_dur  = time.time() - epoch_start
        total_min  = (time.time() - start_time) / 60

        print(f"Epoch {epoch+1:02d}/{num_epochs} | LR: {current_lr:.2e} | "
              f"Time: {epoch_dur:.1f}s ({total_min:.1f} min)")
        print(f"  Train Loss: {train_loss:.4f}  (cls {cls_loss:.4f}  loc {loc_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model -> {save_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

        print("-" * 50)

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training Complete! Total Time: {total_time/60:.2f} minutes.")


if __name__ == "__main__":
    main()
