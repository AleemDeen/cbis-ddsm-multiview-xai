import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import time
import warnings

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, lambda_loc=0.1):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_loc_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)
        roi_masks = batch["roi_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        malignant_mask = (labels == 1)

        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            if lambda_loc > 0.0:
                logits, features = model(images, return_features=True)
            else:
                logits = model(images)

            classification_loss = criterion(logits.view(-1), labels)

            if lambda_loc > 0.0 and malignant_mask.any():
                # Compute GradCAM inline: weight layer4 channels by gradient of
                # the classification loss, then compare against the ROI mask.
                # retain_graph=True keeps the graph alive for the main backward.
                # create_graph=False means grad weights are treated as constants,
                # so loc_loss still backprops through features but not through
                # the gradient computation itself.
                grads = torch.autograd.grad(
                    classification_loss, features,
                    retain_graph=True,
                    create_graph=False
                )[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * features).sum(dim=1, keepdim=True))
                cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)

                cam_upsampled = F.interpolate(
                    cam, size=(512, 512), mode="bilinear", align_corners=False
                )

                loc_loss = dice_loss(
                    cam_upsampled[malignant_mask],
                    roi_masks[malignant_mask]
                )
            else:
                loc_loss = torch.tensor(0.0, device=device)

            loss = classification_loss + lambda_loc * loc_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss     += loss.item()
        running_cls_loss += classification_loss.item()
        running_loc_loss += loc_loss.item()

    n = len(loader)
    return running_loss / n, running_cls_loss / n, running_loc_loss / n


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits.view(-1), labels)

            running_loss += loss.item()

    return running_loss / len(loader)


def dice_loss(pred, target, eps=1e-8):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-loc", type=float, default=0.1,
                        help="Weight for GradCAM localization loss (0 = classification only)")
    args = parser.parse_args()

    lambda_loc = args.lambda_loc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Lambda loc:   {lambda_loc}")

    # Use preprocessed .pt CSV if available, otherwise fall back to raw DICOMs
    PT_CSV = "data_processed/indexed_full_mammogram_images_with_labels_pt.csv"
    CSV_PATH = PT_CSV if Path(PT_CSV).exists() else "data_processed/indexed_full_mammogram_images_with_labels.csv"
    SPLITS_DIR = "splits"

    train_loader, val_loader, _ = build_dataloaders(
        csv_path=CSV_PATH,
        splits_dir=SPLITS_DIR,
        batch_size=64,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    df = pd.read_csv(CSV_PATH)
    num_pos = (df["label"] == 1).sum()
    num_neg = (df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    model = ResNet18SingleView().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    num_epochs = 50
    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0

    start_time = time.time()

    print(f"\nStarting Training (Patience: {patience} epochs)")
    print("-" * 30)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, cls_loss, loc_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, lambda_loc
        )
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f} (Cls: {cls_loss:.4f} | Loc: {loc_loss:.4f}) | Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_duration:.2f}s (Total: {total_elapsed/60:.2f} min)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = f"models/resnet18_single_view_best_loc{lambda_loc}.pt"
            torch.save(model.state_dict(), save_path)
            print("✓ Saved new best model")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        print("-" * 30)

        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training Complete! Total Time: {total_time/60:.2f} minutes.")


if __name__ == "__main__":
    main()