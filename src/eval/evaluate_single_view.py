import torch
import pandas as pd
import numpy as np
import warnings
import os
import cv2
import pydicom
from pathlib import Path

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders
from src.xai.gradcam import GradCAM


def load_roi_mask(scan_path, target_size=512):
    """
    Given original scan file_path, locate and load ROI mask.
    """
    scan_path = Path(scan_path)

    # Example:
    # .../CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29.../1.000.../1-1.dcm
    scan_folder = scan_path.parents[2].name  # Calc-Test_P_00038_LEFT_CC
    roi_folder_name = scan_folder + "_1"

    roi_root = scan_path.parents[3] / roi_folder_name

    if not roi_root.exists():
        return None

    # Navigate nested folders
    subfolders = list(roi_root.glob("*/*/*.dcm"))

    if len(subfolders) == 0:
        return None

    # Load first ROI mask (ignore _2 for now)
    ds = pydicom.dcmread(str(subfolders[0]))
    mask = ds.pixel_array.astype(np.float32)

    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    mask -= mask.min()
    if mask.max() > 0:
        mask /= mask.max()

    return mask


def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)

    intersection = (pred_bin * target_bin).sum()
    return (2.0 * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = build_dataloaders(
        csv_path="data_processed/indexed_full_mammogram_images_with_labels.csv",
        splits_dir="splits",
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    )

    model = ResNet18SingleView().to(device)
    state_dict = torch.load("resnet18_single_view_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(True)

    cam = GradCAM(model, model.model.layer4)

    all_probs, all_labels = [], []
    dice_scores = []
    dice_malignant = []
    dice_true_positive = []
    print("Running Inference with Grad-CAM + Dice...")



    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        file_paths = batch["file_path"]

        logits = model(images)
        probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.view(-1).cpu().numpy().tolist())

        

        for i in range(images.size(0)):
            image = images[i].unsqueeze(0)
            heatmap = cam.generate(image)
            heatmap = heatmap.squeeze().detach().cpu().numpy()

            img_np = image.squeeze().detach().cpu().numpy()
            heatmap = cv2.resize(
                heatmap,
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            mask = load_roi_mask(file_paths[i])

            if mask is not None:
                d = dice_score(heatmap, mask)
                dice_scores.append(d)

                # Malignant only
                if labels[i].item() == 1:
                    dice_malignant.append(d)

                    # True positive only
                    if probs[i] >= 0.5:
                        dice_true_positive.append(d)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_probs >= 0.5)
    rec = recall_score(all_labels, all_probs >= 0.5)
    mean_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0.0
    std_dice = np.std(dice_scores) if len(dice_scores) > 0 else 0.0

    mean_dice_malignant = np.mean(dice_malignant) if len(dice_malignant) > 0 else 0.0
    mean_dice_tp = np.mean(dice_true_positive) if len(dice_true_positive) > 0 else 0.0

    print(f"\n{'='*20} EVALUATION COMPLETE {'='*20}")
    print(f"Test Samples: {len(all_labels)}")
    print(f"AUC Score:    {auc:.4f}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"Mean Dice (All):            {mean_dice:.4f}")
    print(f"Std Dice:                   {std_dice:.4f}")
    print(f"Mean Dice (Malignant):      {mean_dice_malignant:.4f}")
    print(f"Mean Dice (True Positives): {mean_dice_tp:.4f}")
    print(f"{'='*51}")


if __name__ == "__main__":
    main()
