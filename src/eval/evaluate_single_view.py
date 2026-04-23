import argparse
import sys
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
    Locate and load the ground-truth ROI mask for a given scan.

    CBIS-DDSM stores masks in a sibling folder named <scan_folder>_1 at the
    study level. Returns None when no mask is found so the calling code can
    skip Dice computation for that sample rather than raising an exception.
    """
    scan_path = Path(scan_path)

    # Navigate the CBIS-DDSM folder hierarchy:
    # .../CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29.../1.000.../1-1.dcm
    scan_folder     = scan_path.parents[2].name   # e.g. Calc-Test_P_00038_LEFT_CC
    roi_folder_name = scan_folder + "_1"
    roi_root        = scan_path.parents[3] / roi_folder_name

    if not roi_root.exists():
        return None

    subfolders = list(roi_root.glob("*/*/*.dcm"))
    if len(subfolders) == 0:
        return None

    ds   = pydicom.dcmread(str(subfolders[0]))
    mask = ds.pixel_array.astype(np.float32)

    # INTER_NEAREST preserves the hard binary boundary of the annotation
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask -= mask.min()
    if mask.max() > 0:
        mask /= mask.max()

    return mask


def dice_score_hard(pred, target, threshold=0.5):
    """
    Compute Dice coefficient after binarising predictions at a fixed threshold.

    Hard Dice is reported alongside soft Dice because it reflects the overlap
    you would actually see if you drew a binary contour around the predicted region.
    """
    pred_bin   = (pred   > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    return (2.0 * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-8)


def dice_score_soft(pred, target):
    """
    Compute soft Dice on continuous prediction values.

    Soft Dice matches the training loss formulation, so it gives a sense of how
    close the Grad-CAM activations are to the ground-truth mask during training.
    """
    pred   = pred.flatten().astype(np.float32)
    target = target.flatten().astype(np.float32)
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)


def pick_model(default: str) -> str:
    """Interactively choose a .pt file from models/ when running in a terminal."""
    if not sys.stdin.isatty():
        return default
    candidates = sorted(Path("models").glob("*.pt"))
    if not candidates:
        return default
    print("\nAvailable models:")
    for i, p in enumerate(candidates, 1):
        marker = " (default)" if p.name == Path(default).name else ""
        print(f"  [{i}] {p.name}{marker}")
    print(f"  [Enter] use default ({Path(default).name})")
    choice = input("Select model: ").strip()
    if not choice:
        return default
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        return str(candidates[int(choice) - 1])
    print(f"Invalid choice, using default: {Path(default).name}")
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model .pt file to evaluate (prompts if omitted)")
    args = parser.parse_args()

    model_path = args.model_path or pick_model("models/sv_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:  {device}")
    print(f"Model:         {model_path}")

    # Prefer pre-cached .pt tensors for faster test-set loading
    PT_CSV   = "data_processed/indexed_full_mammogram_images_with_labels_pt.csv"
    CSV_PATH = PT_CSV if Path(PT_CSV).exists() else "data_processed/indexed_full_mammogram_images_with_labels.csv"

    # Only load the test split — validation data should never influence evaluation
    _, _, test_loader = build_dataloaders(
        csv_path=CSV_PATH,
        splits_dir="splits",
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    )

    model = ResNet18SingleView().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Grad-CAM requires gradients to be enabled even during evaluation
    for p in model.parameters():
        p.requires_grad_(True)

    # Attach Grad-CAM to layer4 — the deepest spatial feature map
    cam = GradCAM(model, model.model.layer4)

    all_probs, all_labels = [], []

    # Dice scores are tracked in three groups to understand localisation quality
    # across different subsets: all test cases, malignant cases only, and true
    # positives (correctly classified malignant cases)
    soft_dice_scores, hard_dice_scores     = [], []
    soft_dice_malignant, hard_dice_malignant = [], []
    soft_dice_tp, hard_dice_tp             = [], []

    print("Running inference with Grad-CAM + Dice scoring...")

    for batch in test_loader:
        images     = batch["image"].to(device)
        labels     = batch["label"].to(device)
        file_paths = batch["file_path"]

        logits = model(images)
        probs  = torch.sigmoid(logits).view(-1).detach().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.view(-1).cpu().numpy().tolist())

        # Compute Grad-CAM and Dice individually per image (batch_size=1 equivalent)
        for i in range(images.size(0)):
            image   = images[i].unsqueeze(0)
            heatmap = cam.generate(image)
            heatmap = heatmap.squeeze().detach().cpu().numpy()

            img_np  = image.squeeze().detach().cpu().numpy()
            # Resize the 16×16 Grad-CAM output to match the full image resolution
            heatmap = cv2.resize(
                heatmap,
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            mask = load_roi_mask(file_paths[i])
            if mask is None:
                continue

            sd = dice_score_soft(heatmap, mask)
            hd = dice_score_hard(heatmap, mask)

            soft_dice_scores.append(sd)
            hard_dice_scores.append(hd)

            if labels[i].item() == 1:
                soft_dice_malignant.append(sd)
                hard_dice_malignant.append(hd)

                # True positive: malignant and correctly predicted as malignant
                if probs[i] >= 0.5:
                    soft_dice_tp.append(sd)
                    hard_dice_tp.append(hd)

    all_probs  = np.array(all_probs)
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
    print(f"\n--- Hard Dice (binarised at 0.5) ---")
    print(f"Mean Hard Dice (All):            {_mean(hard_dice_scores):.4f} ± {_std(hard_dice_scores):.4f}")
    print(f"Mean Hard Dice (Malignant):      {_mean(hard_dice_malignant):.4f}")
    print(f"Mean Hard Dice (True Positives): {_mean(hard_dice_tp):.4f}")
    print(f"{'='*51}")


if __name__ == "__main__":
    main()
