import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import time
import warnings

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders

# Silence specific torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# --------------------------------------------------
# Training / evaluation helpers
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits = model(images)
            # FIX: Use .view(-1) to change shape from [64, 1] to [64] to match labels
            loss = criterion(logits.view(-1), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(images)
                # FIX: Use .view(-1) to change shape from [64, 1] to [64] to match labels
                loss = criterion(logits.view(-1), labels)

            running_loss += loss.item()

    return running_loss / len(loader)


# --------------------------------------------------
# Main training script
# --------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CSV_PATH = "data_processed/indexed_full_mammogram_images_with_labels.csv"
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

    # Class Weighting
    df = pd.read_csv(CSV_PATH)
    num_pos = (df["label"] == 1).sum()
    num_neg = (df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    # Model Initialization
    model = ResNet18SingleView().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer and Scheduler setup
    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Modern GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    # Early Stopping & Timing parameters
    num_epochs = 50
    patience = 10  # Room for the scheduler to work
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    start_time = time.time()

    print(f"\nStarting Training (Patience: {patience} epochs)")
    print("-" * 30)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 1. Core training and validation
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        # 2. Update the scheduler with the new val_loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Timing and status
        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_duration:.2f}s (Total: {total_elapsed/60:.2f} min)")

        # 3. Early Stopping and Saving Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "resnet18_single_view_best.pt")
            print("✓ Saved new best model")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        print("-" * 30)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training Complete! Total Time: {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()