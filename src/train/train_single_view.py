import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = build_dataloaders(
        csv_path="data_processed/indexed_full_mammogram_images.csv",
        splits_dir="splits",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    model = ResNet18SingleView().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    num_epochs = 5  # short run for sanity

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

    torch.save(model.state_dict(), "resnet18_single_view_baseline.pt")
    print("Saved model checkpoint.")


if __name__ == "__main__":
    main()
