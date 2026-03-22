"""
Stage-2 segmentation fine-tuning — v3: benign suppression loss.

Root cause of v2's mislocalisation:
  The seg head was only trained with a positive signal on malignant cases.
  Benign cases contributed zero seg gradient, so the decoder learned to
  highlight salient mammographic features (nipple/skin-line) rather than
  actual lesion tissue — because it was never penalised for doing so.

Fix — two-sided seg loss:
  Malignant cases: Focal-BCE + Dice toward the ground-truth ROI mask (as before).
  Benign cases:    L1 suppression — penalise *any* non-zero output, teaching
                   the decoder "don't activate here, nothing is malignant."

This forces the model to learn the contrast: activate near a real lesion,
stay silent everywhere else.

Other changes vs v2:
  - Lower lambda-sparse (0.02) — the nipple-focus showed sparsity was too
    aggressive; we want the model selective, not single-spot.
  - lambda-benign-suppress = 0.3 for the new benign suppression term.
  - Start from v2 weights (already learned some spatial selectivity).
  - Saves to models/resnet18_multi_view_seg_v3.pt.

Usage:
    python -m src.train.finetune_multi_view_seg_v3 \
        --base-model models/resnet18_multi_view_seg_v2.pt \
        --epochs 30
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    eps = 1e-8
    bce = -(pos_weight * target * torch.log(pred + eps)
            + (1.0 - target) * torch.log(1.0 - pred + eps))
    pt  = torch.where(target > 0.5, pred, 1.0 - pred)
    return ((1.0 - pt) ** gamma * bce).mean()


def malignant_seg_loss(pred: torch.Tensor, target: torch.Tensor,
                       lambda_sparse: float) -> torch.Tensor:
    """Focal-BCE + Dice + mild sparsity for malignant samples."""
    pos = target.sum()
    neg = target.numel() - pos
    pw  = float((neg / (pos + 1e-8)).clamp(max=50.0))
    return focal_bce(pred, target, gamma=2.0, pos_weight=pw) \
           + dice_loss(pred, target) \
           + lambda_sparse * pred.mean()


def benign_suppress_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    L1 suppression for benign cases — the seg head should output near-zero
    everywhere. This is the core new term that stops the model from activating
    on salient non-lesion features (nipple, skin-line) in benign scans.
    """
    return pred.mean()


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_case_ids(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


# ── Training step ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    lambda_seg, lambda_sparse, lambda_suppress):
    model.train()
    model.classifier.eval()
    for branch in [model.cc_branch, model.mlo_branch]:
        branch.conv1.eval()
        branch.bn1.eval()
        branch.layer1.eval()

    running_cls = running_mal = running_sup = 0.0
    n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        cc        = batch["cc_image"].to(device, non_blocking=True)
        mlo       = batch["mlo_image"].to(device, non_blocking=True)
        labels    = batch["label"].float().to(device, non_blocking=True)
        cc_masks  = batch["cc_mask"].to(device, non_blocking=True)
        mlo_masks = batch["mlo_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, cc_pred, mlo_pred = model(cc, mlo, return_masks=True)

        # Classification loss
        cls = criterion(logits.view(-1), labels)

        mal = labels.bool()
        ben = ~mal

        # Malignant: pull prediction toward ground-truth ROI
        if mal.any():
            mal_seg = (malignant_seg_loss(cc_pred[mal],  cc_masks[mal],  lambda_sparse) +
                       malignant_seg_loss(mlo_pred[mal], mlo_masks[mal], lambda_sparse)) / 2.0
        else:
            mal_seg = torch.zeros(1, device=device)

        # Benign: push prediction toward zero everywhere (the key new term)
        if ben.any():
            sup = (benign_suppress_loss(cc_pred[ben]) +
                   benign_suppress_loss(mlo_pred[ben])) / 2.0
        else:
            sup = torch.zeros(1, device=device)

        loss = cls + lambda_seg * mal_seg + lambda_suppress * sup
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_cls += cls.item()
        running_mal += mal_seg.item()
        running_sup += sup.item()
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0
    return running_cls / n, running_mal / n, running_sup / n


# ── Validation ───────────────────────────────────────────────────────────────

def eval_metrics(model, loader, criterion, device):
    model.eval()
    cls_losses, cc_scores, mlo_scores = [], [], []
    cc_sup_scores, mlo_sup_scores = [], []

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
            ben = ~mal

            for i in range(cc.size(0)):
                if mal[i]:
                    p = cc_pred[i].view(-1);  t = cc_masks[i].view(-1)
                    cc_scores.append((2*(p*t).sum()/(p.sum()+t.sum()+1e-8)).item())
                    p = mlo_pred[i].view(-1); t = mlo_masks[i].view(-1)
                    mlo_scores.append((2*(p*t).sum()/(p.sum()+t.sum()+1e-8)).item())
                else:
                    # Track mean suppression on benign (lower = better)
                    cc_sup_scores.append(cc_pred[i].mean().item())
                    mlo_sup_scores.append(mlo_pred[i].mean().item())

    return (
        float(np.mean(cls_losses)),
        float(np.mean(cc_scores))      if cc_scores      else 0.0,
        float(np.mean(mlo_scores))     if mlo_scores     else 0.0,
        float(np.mean(cc_sup_scores))  if cc_sup_scores  else 0.0,
        float(np.mean(mlo_sup_scores)) if mlo_sup_scores else 0.0,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",      type=str,   default="models/resnet18_multi_view_seg_v2.pt")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--lambda-seg",      type=float, default=2.5)
    parser.add_argument("--lambda-sparse",   type=float, default=0.02,
                        help="Sparsity on malignant masks (lower than v2 — suppression handles benign)")
    parser.add_argument("--lambda-suppress", type=float, default=0.30,
                        help="Benign suppression weight — penalises non-zero output on benign cases")
    parser.add_argument("--save-path",       type=str,   default="models/resnet18_multi_view_seg_v3.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:            {device}")
    print(f"Base model:        {args.base_model}")
    print(f"lambda-seg:        {args.lambda_seg}")
    print(f"lambda-sparse:     {args.lambda_sparse}  (v2: 0.05)")
    print(f"lambda-suppress:   {args.lambda_suppress}  (NEW — benign suppression)")
    print(f"Save path:         {args.save_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    pt_csv = "data_processed/indexed_multi_view_cases_pt.csv"
    csv    = pt_csv if Path(pt_csv).exists() else "data_processed/indexed_multi_view_cases.csv"

    df = pd.read_csv(csv)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    train_df = df[df["case_id"].isin(load_case_ids("splits/train_cases.txt"))].reset_index(drop=True)
    val_df   = df[df["case_id"].isin(load_case_ids("splits/val_cases.txt"))].reset_index(drop=True)

    n_mal = (train_df["label"] == 1).sum()
    n_ben = (train_df["label"] == 0).sum()
    print(f"Train: {len(train_df)} ({n_mal} malignant / {n_ben} benign) | Val: {len(val_df)}")

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

    # Freeze: conv1, bn1, layer1 | Unfreeze: layer2, layer3, layer4 | Freeze: classifier
    for branch in [model.cc_branch, model.mlo_branch]:
        for name, p in branch.named_parameters():
            p.requires_grad_(not any(name.startswith(k) for k in ("conv1", "bn1", "layer1")))
    for p in model.classifier.parameters():
        p.requires_grad_(False)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "seg_head" not in n]
    seg_params      = [p for n, p in model.named_parameters() if p.requires_grad and "seg_head" in n]
    print(f"Trainable backbone: {sum(p.numel() for p in backbone_params):,}")
    print(f"Trainable decoders: {sum(p.numel() for p in seg_params):,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    pos_weight = torch.tensor([n_ben / n_mal], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam([
        {"params": backbone_params, "lr": 2e-6},
        {"params": seg_params,      "lr": 2e-4},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_dice     = 0.0
    epochs_no_improve = 0
    patience          = 10
    start             = time.time()

    print(f"\nTraining for up to {args.epochs} epochs  (patience={patience})")
    print("-" * 68)

    for epoch in range(args.epochs):
        t0 = time.time()
        cls_l, mal_l, sup_l = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.lambda_seg, args.lambda_sparse, args.lambda_suppress,
        )
        val_cls, val_cc, val_mlo, cc_sup, mlo_sup = eval_metrics(
            model, val_loader, criterion, device
        )
        mean_dice = (val_cc + val_mlo) / 2.0
        scheduler.step(mean_dice)

        lr_bb  = optimizer.param_groups[0]["lr"]
        lr_seg = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs}  lr bb:{lr_bb:.1e} seg:{lr_seg:.1e}  [{time.time()-t0:.0f}s]")
        print(f"  Train  cls:{cls_l:.4f}  mal_seg:{mal_l:.4f}  benign_sup:{sup_l:.4f}")
        print(f"  Val    cls:{val_cls:.4f}  Dice CC:{val_cc:.4f}  MLO:{val_mlo:.4f}  mean:{mean_dice:.4f}")
        print(f"  Val benign mean activation  CC:{cc_sup:.4f}  MLO:{mlo_sup:.4f}  (target: ~0)")

        if mean_dice > best_val_dice:
            best_val_dice     = mean_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  *** Saved best -> {args.save_path}  (dice {best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

        print("-" * 68)

        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

    print(f"\nDone in {(time.time()-start)/60:.1f} min.  Best val mean Dice: {best_val_dice:.4f}")
    print(f"\nEvaluate with:")
    print(f"  python -m src.eval.evaluate_multi_view --model-path {args.save_path} --seg-head")


if __name__ == "__main__":
    main()
