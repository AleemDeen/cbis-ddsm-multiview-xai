import argparse
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
import cv2
from pathlib import Path

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from src.models.resnet18_multi_view import ResNet18MultiView, ResNet18MultiViewSeg
from src.data.multi_view_dataset import CBISDDSMMultiViewDataset


def load_case_ids(path) -> set:
    """Read a plain-text split file and return a set of patient case IDs."""
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def gradcam_heatmap(features, loss, size=512):
    """
    Compute a Grad-CAM heatmap from a stored feature map and a scalar loss.

    retain_graph=True is used here because this function is called twice in
    sequence (once for CC, once for MLO) against the same computation graph.
    The graph is released after the second call.
    """
    grads   = torch.autograd.grad(loss, features, retain_graph=True, create_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam     = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
    cam_up  = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
    return cam_up  # (B, 1, size, size)


def dice_score_hard(pred, target, threshold=0.5):
    """
    Compute Dice after binarising predictions at a fixed threshold.

    Hard Dice gives a concrete measure of spatial overlap, equivalent to
    drawing a binary boundary around the predicted region and comparing it
    to the annotated ROI.
    """
    pred_bin   = (pred   > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    return (2.0 * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-8)


def dice_score_soft(pred, target):
    """
    Compute soft Dice on continuous prediction values.

    Mirrors the training loss formulation to give a comparable measure
    of how well predictions aligned with ground-truth masks during training.
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
                        help="Path to trained multi-view model .pt file (prompts if omitted)")
    parser.add_argument("--seg-head", action="store_true",
                        help="Load as ResNet18MultiViewSeg and evaluate seg head masks instead of GradCAM")
    args = parser.parse_args()

    model_path = args.model_path or pick_model("models/mv_baseline.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:  {device}")
    print(f"Model:         {model_path}")

    # Prefer pre-cached .pt tensors for faster data loading
    PT_CSV   = "data_processed/indexed_multi_view_cases_pt.csv"
    CSV_PATH = PT_CSV if Path(PT_CSV).exists() else "data_processed/indexed_multi_view_cases.csv"
    print(f"CSV:           {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    # Evaluate on the held-out test split only
    test_cases = load_case_ids("splits/test_cases.txt")
    test_df    = df[df["case_id"].isin(test_cases)].reset_index(drop=True)
    print(f"Test samples:  {len(test_df)}")

    test_dataset = CBISDDSMMultiViewDataset(test_df)
    test_loader  = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    if args.seg_head:
        # Load the segmentation model — strict=False because a mv_baseline
        # checkpoint can be passed here without the seg head keys
        model = ResNet18MultiViewSeg().to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print("Mode: segmentation head masks")
    else:
        model = ResNet18MultiView().to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        # Grad-CAM requires gradients even during eval
        for p in model.parameters():
            p.requires_grad_(True)
        print("Mode: GradCAM heatmaps")

    all_probs, all_labels = [], []

    # Dice lists are split by view and subset (all / malignant only / true positives)
    # to understand where localisation quality varies most
    cc_soft_all,  cc_hard_all  = [], []
    mlo_soft_all, mlo_hard_all = [], []
    cc_soft_mal,  cc_hard_mal  = [], []
    mlo_soft_mal, mlo_hard_mal = [], []
    cc_soft_tp,   cc_hard_tp   = [], []
    mlo_soft_tp,  mlo_hard_tp  = [], []

    print("Running inference...")

    for batch in test_loader:
        cc        = batch["cc_image"].to(device)
        mlo       = batch["mlo_image"].to(device)
        labels    = batch["label"].float().to(device)
        cc_masks  = batch["cc_mask"].cpu().numpy()    # (B, 1, 512, 512)
        mlo_masks = batch["mlo_mask"].cpu().numpy()

        if args.seg_head:
            with torch.no_grad():
                logits, cc_cams, mlo_cams = model(cc, mlo, return_masks=True)
            probs    = torch.sigmoid(logits).view(-1).cpu().numpy()
            cc_cams  = cc_cams.cpu().numpy()
            mlo_cams = mlo_cams.cpu().numpy()
        else:
            logits, cc_features, mlo_features = model(cc, mlo, return_features=True)
            loss     = logits.view(-1).sigmoid().mean()
            probs    = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
            cc_cams  = gradcam_heatmap(cc_features,  loss).detach().cpu().numpy()
            mlo_cams = gradcam_heatmap(mlo_features, loss).detach().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        for i in range(len(probs)):
            lbl  = int(labels[i].item())
            prob = probs[i]

            cc_cam_np   = cc_cams[i, 0]    # (512, 512)
            mlo_cam_np  = mlo_cams[i, 0]
            cc_mask_np  = cc_masks[i, 0]
            mlo_mask_np = mlo_masks[i, 0]

            cc_sd  = dice_score_soft(cc_cam_np,  cc_mask_np)
            cc_hd  = dice_score_hard(cc_cam_np,  cc_mask_np)
            mlo_sd = dice_score_soft(mlo_cam_np, mlo_mask_np)
            mlo_hd = dice_score_hard(mlo_cam_np, mlo_mask_np)

            cc_soft_all.append(cc_sd);   cc_hard_all.append(cc_hd)
            mlo_soft_all.append(mlo_sd); mlo_hard_all.append(mlo_hd)

            if lbl == 1:
                cc_soft_mal.append(cc_sd);   cc_hard_mal.append(cc_hd)
                mlo_soft_mal.append(mlo_sd); mlo_hard_mal.append(mlo_hd)

                # True positives: malignant cases correctly classified as malignant
                if prob >= 0.5:
                    cc_soft_tp.append(cc_sd);   cc_hard_tp.append(cc_hd)
                    mlo_soft_tp.append(mlo_sd); mlo_hard_tp.append(mlo_hd)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_probs >= 0.5)
    rec = recall_score(all_labels, all_probs >= 0.5)

    def _mean(lst): return np.mean(lst) if lst else 0.0
    def _std(lst):  return np.std(lst)  if lst else 0.0

    print(f"\n{'='*22} EVALUATION COMPLETE {'='*22}")
    print(f"Test Samples: {len(all_labels)}")
    print(f"AUC Score:    {auc:.4f}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Recall:       {rec:.4f}")

    print(f"\n--- Soft Dice (CC view) ---")
    print(f"  All:            {_mean(cc_soft_all):.4f} ± {_std(cc_soft_all):.4f}")
    print(f"  Malignant:      {_mean(cc_soft_mal):.4f}")
    print(f"  True Positives: {_mean(cc_soft_tp):.4f}")

    print(f"\n--- Soft Dice (MLO view) ---")
    print(f"  All:            {_mean(mlo_soft_all):.4f} ± {_std(mlo_soft_all):.4f}")
    print(f"  Malignant:      {_mean(mlo_soft_mal):.4f}")
    print(f"  True Positives: {_mean(mlo_soft_tp):.4f}")

    print(f"\n--- Hard Dice (CC view, threshold = 0.5) ---")
    print(f"  All:            {_mean(cc_hard_all):.4f} ± {_std(cc_hard_all):.4f}")
    print(f"  Malignant:      {_mean(cc_hard_mal):.4f}")
    print(f"  True Positives: {_mean(cc_hard_tp):.4f}")

    print(f"\n--- Hard Dice (MLO view, threshold = 0.5) ---")
    print(f"  All:            {_mean(mlo_hard_all):.4f} ± {_std(mlo_hard_all):.4f}")
    print(f"  Malignant:      {_mean(mlo_hard_mal):.4f}")
    print(f"  True Positives: {_mean(mlo_hard_tp):.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
