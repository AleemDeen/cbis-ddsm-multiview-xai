"""
Stage-2 segmentation fine-tuning — improved localisation (v2).

Key changes over v1:
  - lambda-seg default raised to 2.5 (forces the decoder to localise harder).
  - Sparsity penalty (L1 on mask output) discourages broad whole-breast activation.
  - layer2 also unfrozen (finer spatial resolution for the skip connections).
  - Focal-BCE for the segmentation head — down-weights easy background pixels.
  - Gradient clipping to stabilise the higher-lr decoder.
  - Saves to models/resnet18_multi_view_seg_v2.pt so the original is untouched.

Usage:
    python -m src.train.finetune_multi_view_seg_v2 \
        --base-model models/resnet18_multi_view_seg.pt \
        --epochs 30 \
        --lambda-seg 2.5 \
        --lambda-sparse 0.05

Start from the existing seg model (already learned rough localisation) rather
than the raw classification backbone, so it converges faster.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from src.models.resnet18_multi_view import ResNet18MultiViewSeg
from src.data.multi_view_dataset import CBISDDSMMultiViewDataset


# ── Transforms ──────────────────────────────────────────────────────────────

def get_transforms(train=True):
    if train:
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=10),
            v2.ToDtype(torch.float32, scale=True),
        ])
    return v2.Compose([v2.ToDtype(torch.float32, scale=True)])


# ── Loss functions ───────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    return 1.0 - (2.0 * inter / (pred.sum(dim=1) + target.sum(dim=1) + 1e-8)).mean()


def focal_bce(pred: torch.Tensor, target: torch.Tensor,
              gamma: float = 2.0, pos_weight: float = 50.0) -> torch.Tensor:
    """
    Focal binary cross-entropy for sparse ROI masks.
    - pos_weight: up-weight positive (lesion) pixels to compensate for sparsity.
    - gamma: focusing parameter — down-weights easy negatives so the model
      concentrates on the hard (boundary) pixels rather than learning
      'activate everywhere safely'.
    """
    eps  = 1e-8
    bce  = -(pos_weight * target * torch.log(pred + eps)
             + (1.0 - target) * torch.log(1.0 - pred + eps))
    pt   = torch.where(target > 0.5, pred, 1.0 - pred)
    loss = ((1.0 - pt) ** gamma * bce).mean()
    return loss


def seg_loss_v2(pred: torch.Tensor, target: torch.Tensor,
                lambda_sparse: float = 0.05) -> torch.Tensor:
    """
    Focal BCE + Dice + L1 sparsity.
    The sparsity term penalises the total mask area, pushing the decoder
    to activate only where it's most confident rather than blanketing the
    whole breast.
    """
    pos = target.sum()
    neg = target.numel() - pos
    pw  = float((neg / (pos + 1e-8)).clamp(max=50.0))

    focal = focal_bce(pred, target, gamma=2.0, pos_weight=pw)
    dice  = dice_loss(pred, target)
    # L1 sparsity: penalise mean output magnitude to fight whole-breast diffusion
    sparse = pred.mean()
    return focal + dice + lambda_sparse * sparse


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_case_ids(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


# ── Training step ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    lambda_seg: float, lambda_sparse: float):
    model.train()
    # Keep frozen layers in eval mode (correct BN / Dropout behaviour)
    model.classifier.eval()
    for branch in [model.cc_branch, model.mlo_branch]:
        branch.conv1.eval()
        branch.bn1.eval()
        branch.layer1.eval()

    running_cls = running_seg = 0.0
    n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        cc        = batch["cc_image"].to(device, non_blocking=True)
        mlo       = batch["mlo_image"].to(device, non_blocking=True)
        labels    = batch["label"].float().to(device, non_blocking=True)
        cc_masks  = batch["cc_mask"].to(device, non_blocking=True)
        mlo_masks = batch["mlo_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, cc_pred, mlo_pred = model(cc, mlo, return_masks=True)

        cls = criterion(logits.view(-1), labels)

        mal = labels.bool()
        if mal.any():
            seg = (seg_loss_v2(cc_pred[mal],  cc_masks[mal],  lambda_sparse) +
                   seg_loss_v2(mlo_pred[mal], mlo_masks[mal], lambda_sparse)) / 2.0
        else:
            seg = torch.zeros(1, device=device)

        loss = cls + lambda_seg * seg
        loss.backward()

        # Clip gradients — the higher lambda-seg can cause instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_cls += cls.item()
        running_seg += seg.item()
        n += 1

    return (running_cls / n, running_seg / n) if n > 0 else (0.0, 0.0)


# ── Validation ───────────────────────────────────────────────────────────────

def eval_metrics(model, loader, criterion, device):
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

            mal = labels.bool()
            for i in range(cc.size(0)):
                if not mal[i]:
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",    type=str,   default="models/resnet18_multi_view_seg.pt",
                        help="Checkpoint to start from (use existing seg model for faster convergence)")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--lambda-seg",    type=float, default=2.5,
                        help="Weight of segmentation loss vs classification loss")
    parser.add_argument("--lambda-sparse", type=float, default=0.05,
                        help="Weight of L1 sparsity penalty on mask outputs")
    parser.add_argument("--save-path",     type=str,   default="models/resnet18_multi_view_seg_v2.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:         {device}")
    print(f"Base model:     {args.base_model}")
    print(f"lambda-seg:     {args.lambda_seg}  (v1 used 1.0)")
    print(f"lambda-sparse:  {args.lambda_sparse}")
    print(f"Save path:      {args.save_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    pt_csv  = "data_processed/indexed_multi_view_cases_pt.csv"
    csv_raw = "data_processed/indexed_multi_view_cases.csv"
    csv     = pt_csv if Path(pt_csv).exists() else csv_raw

    df = pd.read_csv(csv)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    train_cases = load_case_ids("splits/train_cases.txt")
    val_cases   = load_case_ids("splits/val_cases.txt")

    train_df = df[df["case_id"].isin(train_cases)].reset_index(drop=True)
    val_df   = df[df["case_id"].isin(val_cases)].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    loader_args = dict(batch_size=8, num_workers=4, pin_memory=True,
                       prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(CBISDDSMMultiViewDataset(train_df, transform=get_transforms(True)),
                              shuffle=True,  **loader_args)
    val_loader   = DataLoader(CBISDDSMMultiViewDataset(val_df,   transform=get_transforms(False)),
                              shuffle=False, **loader_args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ResNet18MultiViewSeg().to(device)
    weights = torch.load(args.base_model, map_location=device, weights_only=True)
    missing, _ = model.load_state_dict(weights, strict=False)
    print(f"Loaded {args.base_model}  ({len(missing)} missing keys)")

    # Freeze: conv1, bn1, layer1  (keep low-level features stable)
    # Unfreeze: layer2, layer3, layer4  (more spatial resolution for skip connections)
    # Freeze: classifier  (classification head — prevents forgetting)
    for branch in [model.cc_branch, model.mlo_branch]:
        for name, p in branch.named_parameters():
            frozen = any(name.startswith(k) for k in ("conv1", "bn1", "layer1"))
            p.requires_grad_(not frozen)
    for p in model.classifier.parameters():
        p.requires_grad_(False)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "seg_head" not in n]
    seg_params      = [p for n, p in model.named_parameters()
                       if p.requires_grad and "seg_head" in n]
    print(f"Trainable backbone (layer2-4): {sum(p.numel() for p in backbone_params):,}")
    print(f"Trainable seg decoders:        {sum(p.numel() for p in seg_params):,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    num_pos    = (train_df["label"] == 1).sum()
    num_neg    = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam([
        {"params": backbone_params, "lr": 5e-6},   # even lower — we're fine-tuning a fine-tune
        {"params": seg_params,      "lr": 5e-4},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=False
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_dice     = 0.0
    epochs_no_improve = 0
    patience          = 10
    start_time        = time.time()

    print(f"\nTraining for up to {args.epochs} epochs  (patience={patience})")
    print("-" * 60)

    for epoch in range(args.epochs):
        t0 = time.time()
        cls_l, seg_l = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, args.lambda_seg, args.lambda_sparse,
        )
        val_cls, val_cc, val_mlo = eval_metrics(model, val_loader, criterion, device)
        mean_dice = (val_cc + val_mlo) / 2.0

        scheduler.step(mean_dice)
        lr_bb  = optimizer.param_groups[0]["lr"]
        lr_seg = optimizer.param_groups[1]["lr"]

        print(f"Epoch {epoch+1:02d}/{args.epochs}  "
              f"lr bb:{lr_bb:.1e} seg:{lr_seg:.1e}  [{time.time()-t0:.0f}s]")
        print(f"  Train  cls:{cls_l:.4f}  seg:{seg_l:.4f}")
        print(f"  Val    cls:{val_cls:.4f}  Dice CC:{val_cc:.4f}  MLO:{val_mlo:.4f}  "
              f"mean:{mean_dice:.4f}")

        if mean_dice > best_val_dice:
            best_val_dice     = mean_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  *** Saved best -> {args.save_path}  (dice {best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

        print("-" * 60)

        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

    total = (time.time() - start_time) / 60
    print(f"\nDone in {total:.1f} min.  Best val mean Dice: {best_val_dice:.4f}")
    print(f"\nEvaluate with:")
    print(f"  python -m src.eval.evaluate_multi_view --model-path {args.save_path} --seg-head")


if __name__ == "__main__":
    main()
