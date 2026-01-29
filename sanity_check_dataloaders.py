from src.data.dataloaders import build_dataloaders
train_loader, val_loader, test_loader = build_dataloaders(
    csv_path="data_processed/indexed_full_mammogram_images.csv",
    splits_dir="splits",
    batch_size=4,
    num_workers=0,
    pin_memory=False,
)


batch = next(iter(train_loader))

print("Train batch image shape:", batch["image"].shape)
print("Views:", batch["view"])
print("Laterality:", batch["laterality"])
