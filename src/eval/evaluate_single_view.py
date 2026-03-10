import argparse
import torch
import pandas as pd
import numpy as np
import warnings
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


def dice_score_hard(pred, target, threshold=0.5):
    """Binarized Dice at a fixed threshold."""
    pred_bin   = (pred   > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    return (2.0 * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-8)


def dice_score_soft(pred, target):
    """Soft Dice on continuous values — matches the training loss."""
    pred   = pred.flatten().astype(np.float32)
    target = target.flatten().astype(np.float32)
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/resnet18_single_view_best.pt",
                        help="Path to model .pt file to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:  {device}")
    print(f"Model:         {args.model_path}")

    PT_CSV   = "data_processed/indexed_full_mammogram_images_with_labels_pt.csv"
    CSV_PATH = PT_CSV if Path(PT_CSV).exists() else "data_processed/indexed_full_mammogram_images_with_labels.csv"

    _, _, test_loader = build_dataloaders(
        csv_path=CSV_PATH,
        splits_dir="splits",
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    )

    model = ResNet18SingleView().to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(True)

    cam = GradCAM(model, model.model.layer4)

    all_probs, all_labels = [], []
    soft_dice_scores, hard_dice_scores = [], []
    soft_dice_malignant, hard_dice_malignant = [], []
    soft_dice_tp, hard_dice_tp = [], []
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
                sd = dice_score_soft(heatmap, mask)
                hd = dice_score_hard(heatmap, mask)
                soft_dice_scores.append(sd)
                hard_dice_scores.append(hd)

                if labels[i].item() == 1:
                    soft_dice_malignant.append(sd)
                    hard_dice_malignant.append(hd)

                    if probs[i] >= 0.5:
                        soft_dice_tp.append(sd)
                        hard_dice_tp.append(hd)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_probs >= 0.5)
    rec = recall_score(all_labels, all_probs >= 0.5)

    def _mean(lst): return np.mean(lst) if lst else 0.0
    def _std(lst):  return np.std(lst)  if lst else 0.0

    print(f"\n{'='*20} EVALUATION COMPLETE {'='*20}")
    print(f"Test Samples: {len(all_labels)}")
    print(f"AUC Score:    {auc:.4f}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"\n--- Soft Dice (matches training loss) ---")
    print(f"Mean Soft Dice (All):            {_mean(soft_dice_scores):.4f} ± {_std(soft_dice_scores):.4f}")
    print(f"Mean Soft Dice (Malignant):      {_mean(soft_dice_malignant):.4f}")
    print(f"Mean Soft Dice (True Positives): {_mean(soft_dice_tp):.4f}")
    print(f"\n--- Hard Dice (binarized at 0.5) ---")
    print(f"Mean Hard Dice (All):            {_mean(hard_dice_scores):.4f} ± {_std(hard_dice_scores):.4f}")
    print(f"Mean Hard Dice (Malignant):      {_mean(hard_dice_malignant):.4f}")
    print(f"Mean Hard Dice (True Positives): {_mean(hard_dice_tp):.4f}")
    print(f"{'='*51}")


if __name__ == "__main__":
    main()
