import torch
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

# Silence the specific UserWarnings from torchvision internally
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build Dataloader
    _, _, test_loader = build_dataloaders(
        csv_path="data_processed/indexed_full_mammogram_images_with_labels.csv",
        splits_dir="splits",
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    )

    # Load Model
    model = ResNet18SingleView().to(device)

    state_dict = torch.load(
        "resnet18_single_view_best.pt",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    all_probs, all_labels = [], []

    print("Running Inference...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.view(-1).cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_probs >= 0.5)
    rec = recall_score(all_labels, all_probs >= 0.5)

    print(f"\n{'='*20} EVALUATION COMPLETE {'='*20}")
    print(f"Test Samples: {len(all_labels)}")
    print(f"AUC Score:    {auc:.4f}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"{'='*51}")

    # -----------------------------
    # NEW: Save / update CSV files
    # -----------------------------

    os.makedirs("results", exist_ok=True)

    # Save predictions
    predictions_df = pd.DataFrame({
        "true_label": all_labels,
        "predicted_probability": all_probs,
        "predicted_label": (all_probs >= 0.5).astype(int)
    })

    predictions_df.to_csv(
        "results/single_view_test_predictions.csv",
        index=False
    )

    # Save metrics
    metrics_df = pd.DataFrame([{
        "num_test_samples": len(all_labels),
        "auc": auc,
        "accuracy": acc,
        "recall": rec
    }])

    metrics_df.to_csv(
        "results/single_view_test_metrics.csv",
        index=False
    )

    print("Saved:")
    print(" - results/single_view_test_predictions.csv")
    print(" - results/single_view_test_metrics.csv")


if __name__ == "__main__":
    main()
