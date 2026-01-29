import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders


# --------------------------------------------------
# Training / evaluation helpers
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item()

    return running_loss / len(loader)


# --------------------------------------------------
# Main training script
# --------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # Paths
    # ------------------------------
    CSV_PATH = "data_processed/indexed_full_mammogram_images_with_labels.csv"
    SPLITS_DIR = "splits"

    # ------------------------------
    # Load data (GPU-friendly defaults)
    # ------------------------------
    train_loader, val_loader, _ = build_dataloaders(
        csv_path=CSV_PATH,
        splits_dir=SPLITS_DIR,
        batch_size=64,          # ↑ increase if VRAM allows
        num_workers=4,          # parallel data loading
        pin_memory=True,        # faster CPU → GPU transfer
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ------------------------------
    # Compute class weighting (from full CSV)
    # ------------------------------
    df = pd.read_csv(CSV_PATH)

    num_pos = (df["label"] == 1).sum()
    num_neg = (df["label"] == 0).sum()

    if num_pos == 0 or num_neg == 0:
        raise ValueError("Invalid label distribution – check CSV")

    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    print(f"Positive (malignant) samples: {num_pos}")
    print(f"Negative (benign) samples: {num_neg}")
    print(f"pos_weight: {pos_weight.item():.2f}")

    # ------------------------------
    # Model, loss, optimiser
    # ------------------------------
    model = ResNet18SingleView().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=1e-4)

   scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")
with torch.amp.autocast('cuda', enabled=device.type == "cuda"):

    # ------------------------------
    # Training loop with early stopping
    # ------------------------------
    num_epochs = 50
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "resnet18_single_view_best.pt")
            print("✓ Saved new best model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
