import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import time

from src.models.resnet18_multi_view import ResNet18MultiView
from src.data.multi_view_dataset import CBISDDSMMultiViewDataset


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        cc = batch["cc_image"].to(device, non_blocking=True)
        mlo = batch["mlo_image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits = model(cc, mlo)
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
            cc = batch["cc_image"].to(device, non_blocking=True)
            mlo = batch["mlo_image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(cc, mlo)
                loss = criterion(logits.view(-1), labels)

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CSV_PATH = "data_processed/indexed_multi_view_cases.csv"

    full_df = pd.read_csv(CSV_PATH)

    # 80/20 split
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)

    train_dataset = CBISDDSMMultiViewDataset(train_df)
    val_dataset = CBISDDSMMultiViewDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Class weighting
    num_pos = (train_df["label"] == 1).sum()
    num_neg = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    model = ResNet18MultiView().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    num_epochs = 40
    patience = 8
    best_val_loss = float("inf")
    epochs_no_improve = 0

    start_time = time.time()

    print("\nStarting Multi-View Training")
    print("-" * 40)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_duration:.2f}s (Total: {total_elapsed/60:.2f} min)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "resnet18_multi_view_best.pt")
            print("✓ Saved new best model")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        print("-" * 40)

        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training Complete! Total Time: {total_time/60:.2f} minutes.")


if __name__ == "__main__":
    main()