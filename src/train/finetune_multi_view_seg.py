"""
Stage-2 segmentation fine-tuning with U-Net decoder and partial backbone unfreeze.

Strategy:
  - Load pre-trained classification backbone (strict=False for new seg heads).
  - Freeze conv1, layer1, layer2, classifier — keeps low-level / semantic features stable.
  - Unfreeze layer3 + layer4 of both branches so features become spatially discriminative.
  - Train seg heads (U-Net decoders with skip connections) + unfrozen backbone layers.
  - Combined loss: classification BCE (prevents forgetting) + weighted seg BCE + Dice.

Usage:
    python -m src.train.finetune_multi_view_seg \
        --base-model models/resnet18_multi_view_best_loc0.0.pt \
        --epochs 40 \
        --lambda-seg 1.0
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from src.models.resnet18_multi_view import ResNet18MultiViewSeg
from src.data.multi_view_dataset import CBISDDSMMultiViewDataset


def get_transforms(train=True):
    if train:
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
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


def seg_loss(pred, target):
    """Weighted BCE + Dice for sparse ROI masks."""
    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + 1e-8)).clamp(max=50.0)
    bce = -(pos_weight * target * torch.log(pred + 1e-8)
            + (1.0 - target) * torch.log(1.0 - pred + 1e-8)).mean()
    return bce + dice_loss(pred, target)


def train_one_epoch(model, loader, optimizer, criterion, device, lambda_seg):
    model.train()
    # Frozen layers stay in eval mode (correct BN/Dropout stats)
    model.classifier.eval()
    for branch in [model.cc_branch, model.mlo_branch]:
        branch.conv1.eval()
        branch.bn1.eval()
        branch.layer1.eval()
        branch.layer2.eval()

    running_cls = running_seg = 0.0
    n = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        cc        = batch["cc_image"].to(device, non_blocking=True)
        mlo       = batch["mlo_image"].to(device, non_blocking=True)
        labels    = batch["label"].float().to(device, non_blocking=True)
        cc_masks  = batch["cc_mask"].to(device, non_blocking=True)
        mlo_masks = batch["mlo_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, cc_pred, mlo_pred = model(cc, mlo, return_masks=True)

        # Classification loss — prevents catastrophic forgetting
        cls = criterion(logits.view(-1), labels)

        # Seg loss only on malignant samples (masks are defined only there)
        malignant_mask = labels.bool()
        if malignant_mask.any():
            seg = (seg_loss(cc_pred[malignant_mask],  cc_masks[malignant_mask]) +
                   seg_loss(mlo_pred[malignant_mask], mlo_masks[malignant_mask])) / 2.0
        else:
            seg = torch.zeros(1, device=device)

        loss = cls + lambda_seg * seg
        loss.backward()
        optimizer.step()

        running_cls += cls.item()
        running_seg += seg.item()
        n += 1

    return (running_cls / n, running_seg / n) if n > 0 else (0.0, 0.0)


def eval_metrics(model, loader, criterion, device):
    """Returns (val_cls_loss, val_cc_dice, val_mlo_dice)."""
    model.eval()
    cls_losses, cc_scores, mlo_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            cc        = batch["cc_image"].to(device, non_blocking=True)
            mlo       = batch["mlo_image"].to(device, non_blocking=True)
            labels    = batch["label"].float().to(device, non_blocking=True)
            cc_masks  = batch["cc_mask"].to(device, non_blocking=True)
            mlo_masks = batch["mlo_mask"].to(device, non_blocking=True)

            logits, cc_pred, mlo_pred = model(cc, mlo, return_masks=True)
            cls_losses.append(criterion(logits.view(-1), labels).item())

            malignant_mask = labels.bool()
            for i in range(cc.size(0)):
                if not malignant_mask[i]:
                    continue
                p = cc_pred[i].view(-1);  t = cc_masks[i].view(-1)
                cc_scores.append((2 * (p * t).sum() / (p.sum() + t.sum() + 1e-8)).item())
                p = mlo_pred[i].view(-1); t = mlo_masks[i].view(-1)
                mlo_scores.append((2 * (p * t).sum() / (p.sum() + t.sum() + 1e-8)).item())

    return (
        float(np.mean(cls_losses)),
        float(np.mean(cc_scores))  if cc_scores  else 0.0,
        float(np.mean(mlo_scores)) if mlo_scores else 0.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/resnet18_multi_view_best_loc0.0.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lambda-seg", type=float, default=1.0,
                        help="Weight for segmentation loss relative to classification loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:    {device}")
    print(f"Base model:      {args.base_model}")
    print(f"Lambda-seg:      {args.lambda_seg}")

    PT_CSV   = "data_processed/indexed_multi_view_cases_pt.csv"
    CSV_PATH = PT_CSV if Path(PT_CSV).exists() else "data_processed/indexed_multi_view_cases.csv"
    print(f"CSV:             {CSV_PATH}")

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

    loader_args = dict(batch_size=8, num_workers=4, pin_memory=True,
                       prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_args)

    # Build model and load backbone weights (strict=False — new SegDecoder heads)
    model = ResNet18MultiViewSeg().to(device)
    base_weights = torch.load(args.base_model, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(base_weights, strict=False)
    seg_keys = [k for k in missing if "seg_head" in k]
    print(f"Loaded backbone from {args.base_model}")
    print(f"  Seg decoder keys initialised from scratch: {len(seg_keys)}")

    # ── Freeze conv1, layer1, layer2, classifier ──────────────────────────────
    # Unfreeze layer3 + layer4 so features become spatially discriminative
    for branch in [model.cc_branch, model.mlo_branch]:
        for name, p in branch.named_parameters():
            frozen = any(name.startswith(k) for k in ("conv1", "bn1", "layer1", "layer2"))
            p.requires_grad_(not frozen)
    for p in model.classifier.parameters():
        p.requires_grad_(False)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "seg_head" not in n]
    seg_params      = [p for n, p in model.named_parameters()
                       if p.requires_grad and "seg_head" in n]
    print(f"Trainable — backbone (layer3+4): {sum(p.numel() for p in backbone_params):,}")
    print(f"Trainable — seg decoders:        {sum(p.numel() for p in seg_params):,}")

    # Weighted BCE for classification (handles class imbalance)
    num_pos    = (train_df["label"] == 1).sum()
    num_neg    = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Backbone gets lower LR (pre-trained), seg decoders get higher LR (random init)
    optimizer = Adam([
        {"params": backbone_params, "lr": 1e-5},
        {"params": seg_params,      "lr": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    save_path         = "models/resnet18_multi_view_seg.pt"
    num_epochs        = args.epochs
    patience          = 12
    best_val_dice     = 0.0
    epochs_no_improve = 0
    start_time        = time.time()

    print(f"\nTraining  (save → {save_path})")
    print("-" * 55)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        cls_loss, seg_loss_val = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.lambda_seg
        )
        val_cls, val_cc_dice, val_mlo_dice = eval_metrics(model, val_loader, criterion, device)
        mean_val_dice = (val_cc_dice + val_mlo_dice) / 2.0

        scheduler.step(mean_val_dice)
        lr_bb  = optimizer.param_groups[0]["lr"]
        lr_seg = optimizer.param_groups[1]["lr"]
        epoch_dur = time.time() - epoch_start
        total_min = (time.time() - start_time) / 60

        print(f"Epoch {epoch+1:02d}/{num_epochs} | LR bb:{lr_bb:.1e} seg:{lr_seg:.1e} | "
              f"Time: {epoch_dur:.1f}s ({total_min:.1f} min)")
        print(f"  Train — cls: {cls_loss:.4f}  seg: {seg_loss_val:.4f}")
        print(f"  Val   — cls: {val_cls:.4f}  "
              f"Dice CC: {val_cc_dice:.4f}  MLO: {val_mlo_dice:.4f}  "
              f"Mean: {mean_val_dice:.4f}")

        if mean_val_dice > best_val_dice:
            best_val_dice     = mean_val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model -> {save_path}  (val dice {best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

        print("-" * 55)

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training Complete! Total Time: {total_time/60:.2f} minutes.")
    print(f"\nEvaluate with:")
    print(f"  python -m src.eval.evaluate_multi_view --model-path {save_path} --seg-head")


if __name__ == "__main__":
    main()
