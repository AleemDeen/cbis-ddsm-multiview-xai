"""
Stage-2 localization fine-tuning for the multi-view model.

Loads a pre-trained classification model, freezes the classifier head,
then fine-tunes only the branch feature extractors with a small GradCAM
localization loss.  Classification performance is protected because the
head weights cannot change.

Usage:
    python -m src.train.finetune_multi_view_loc \
        --base-model models/resnet18_multi_view_best_loc0.0.pt \
        --lambda-loc 0.01
"""

import argparse
import torch
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


def train_one_epoch(model, loader, optimizer, device, lambda_loc):
    model.train()
    # Keep classifier in eval mode so its BatchNorm/Dropout stats don't shift
    model.classifier.eval()

    running_loc = 0.0
    n = 0

    for batch in tqdm(loader, desc="Fine-tuning", leave=False):
        cc        = batch["cc_image"].to(device, non_blocking=True)
        mlo       = batch["mlo_image"].to(device, non_blocking=True)
        labels    = batch["label"].float().to(device, non_blocking=True)
        cc_masks  = batch["cc_mask"].to(device, non_blocking=True)
        mlo_masks = batch["mlo_mask"].to(device, non_blocking=True)

        malignant_mask = labels.bool()
        if not malignant_mask.any():
            continue

        optimizer.zero_grad(set_to_none=True)

        # No AMP here — autograd.grad + fp16 gradients overflow without a GradScaler
        logits, cc_features, mlo_features = model(cc, mlo, return_features=True)
        # Proxy scalar: mean sigmoid over batch drives gradients into both branches
        proxy_loss = torch.sigmoid(logits).view(-1).mean()

        # GradCAM for CC (retain graph for MLO + backward)
        cc_grads = torch.autograd.grad(
            proxy_loss, cc_features,
            retain_graph=True, create_graph=False,
        )[0]
        cc_weights = cc_grads.mean(dim=(2, 3), keepdim=True)
        cc_cam = F.relu((cc_weights * cc_features).sum(dim=1, keepdim=True))
        cc_cam = cc_cam / (cc_cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        cc_cam_up = F.interpolate(cc_cam, size=(512, 512), mode="bilinear", align_corners=False)

        # GradCAM for MLO (retain graph for backward)
        mlo_grads = torch.autograd.grad(
            proxy_loss, mlo_features,
            retain_graph=True, create_graph=False,
        )[0]
        mlo_weights = mlo_grads.mean(dim=(2, 3), keepdim=True)
        mlo_cam = F.relu((mlo_weights * mlo_features).sum(dim=1, keepdim=True))
        mlo_cam = mlo_cam / (mlo_cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        mlo_cam_up = F.interpolate(mlo_cam, size=(512, 512), mode="bilinear", align_corners=False)

        cc_loc  = dice_loss(cc_cam_up[malignant_mask],  cc_masks[malignant_mask])
        mlo_loc = dice_loss(mlo_cam_up[malignant_mask], mlo_masks[malignant_mask])
        loc_loss = (cc_loc + mlo_loc) / 2.0

        (lambda_loc * loc_loss).backward()
        optimizer.step()

        running_loc += loc_loss.item()
        n += 1

    return running_loc / n if n > 0 else 0.0


def eval_val_loss(model, loader, criterion, device):
    """Evaluate classification val loss to detect regression."""
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch in loader:
            cc     = batch["cc_image"].to(device, non_blocking=True)
            mlo    = batch["mlo_image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)
            logits = model(cc, mlo)
            loss   = criterion(logits.view(-1), labels)
            running += loss.item()
    return running / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/resnet18_multi_view_best_loc0.0.pt",
                        help="Pre-trained classification model to start from")
    parser.add_argument("--lambda-loc", type=float, default=0.01,
                        help="Localization loss weight (keep small, e.g. 0.01–0.02)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:  {device}")
    print(f"Base model:    {args.base_model}")
    print(f"Lambda-loc:    {args.lambda_loc}")

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
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset = CBISDDSMMultiViewDataset(train_df, transform=get_transforms(train=True))
    val_dataset   = CBISDDSMMultiViewDataset(val_df,   transform=get_transforms(train=False))

    loader_args = dict(batch_size=16, num_workers=4, pin_memory=True,
                       prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_args)

    # Load pre-trained weights
    model = ResNet18MultiView().to(device)
    state_dict = torch.load(args.base_model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {args.base_model}")

    # Freeze classifier — only branch parameters will update
    for p in model.classifier.parameters():
        p.requires_grad_(False)

    branch_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in branch_params):,}  "
          f"(classifier frozen)")

    import torch.nn as nn
    num_pos    = (train_df["label"] == 1).sum()
    num_neg    = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Low LR to avoid disturbing learned features
    optimizer = Adam(branch_params, lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    save_path = f"models/resnet18_multi_view_finetuned_loc{args.lambda_loc}.pt"

    num_epochs        = 20
    patience          = 8
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    start_time        = time.time()

    print(f"\nStarting loc fine-tuning  (save → {save_path})")
    print("-" * 50)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        loc_loss = train_one_epoch(model, train_loader, optimizer, device, args.lambda_loc)
        val_loss = eval_val_loss(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_dur  = time.time() - epoch_start
        total_min  = (time.time() - start_time) / 60

        print(f"Epoch {epoch+1:02d}/{num_epochs} | LR: {current_lr:.2e} | "
              f"Time: {epoch_dur:.1f}s ({total_min:.1f} min)")
        print(f"  Loc Loss (train): {loc_loss:.4f}")
        print(f"  Val Cls Loss:     {val_loss:.4f}")

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
    print(f"Fine-tuning Complete! Total Time: {total_time/60:.2f} minutes.")
    print(f"\nEvaluate with:")
    print(f"  python -m src.eval.evaluate_multi_view --model-path {save_path}")


if __name__ == "__main__":
    main()
