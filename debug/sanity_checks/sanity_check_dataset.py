from src.data.mammogram_dataset import CBISDDSMImageDataset

ds = CBISDDSMImageDataset(
    csv_path="data_processed/indexed_full_mammogram_images.csv"
)

print("Dataset size:", len(ds))

sample = ds[0]
print("Image shape:", sample["image"].shape)
print("Min/max:", sample["image"].min().item(), sample["image"].max().item())
print("View:", sample["view"])
print("Laterality:", sample["laterality"])
