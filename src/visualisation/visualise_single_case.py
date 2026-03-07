import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pandas as pd
from pathlib import Path

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.mammogram_dataset import CBISDDSMImageDataset


MODEL_PATH = "resnet18_single_view_best.pt"
CSV_PATH = "data_processed/indexed_full_mammogram_images_with_labels.csv"
OUTPUT_DIR = "results/single_case_visualisation"
TARGET_SIZE = 512


def save_image(array, path):
    array = array.astype(np.float32)
    array -= array.min()
    if array.max() > 0:
        array /= array.max()
    array = (array * 255).astype(np.uint8)
    cv2.imwrite(path, array)


def overlay_heatmap(image, heatmap):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    return overlay


def main(patient_id_to_visualise):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    case_df = df[df["patient_id"].str.contains(patient_id_to_visualise)]

    if len(case_df) == 0:
        print("Patient not found.")
        return

    row = case_df.iloc[0]

    dataset = CBISDDSMImageDataset(case_df.iloc[[0]])
    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)
    label = sample["label"]

    # Save input
    input_np = image.squeeze().cpu().numpy()
    save_image(input_np, f"{OUTPUT_DIR}/1_input.png")

    # Load model
    model = ResNet18SingleView().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits, features = model(image, return_features=True)
        prob = torch.sigmoid(logits).item()

    print(f"Prediction probability: {prob:.4f}")
    print(f"True label: {label}")

    # Feature map mean
    activation_map = features.mean(dim=1, keepdim=True)
    activation_map = F.interpolate(
        activation_map,
        size=(TARGET_SIZE, TARGET_SIZE),
        mode="bilinear",
        align_corners=False
    )

    activation_np = activation_map.squeeze().cpu().numpy()

    save_image(activation_np, f"{OUTPUT_DIR}/2_activation_map.png")

    # ROI mask
    roi_mask = dataset._load_roi_mask(row["file_path"])
    save_image(roi_mask, f"{OUTPUT_DIR}/3_roi_mask.png")

    # Overlay
    input_vis = (input_np * 255).astype(np.uint8)
    overlay = overlay_heatmap(input_vis, activation_np)
    cv2.imwrite(f"{OUTPUT_DIR}/4_overlay.png", overlay)

    print("Saved visualisations to:", OUTPUT_DIR)


if __name__ == "__main__":
    # Replace with an example like "P_00209"
    main("P_00209")