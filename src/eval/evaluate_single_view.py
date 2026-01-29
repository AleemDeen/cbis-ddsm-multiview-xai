import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score
)

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Load test data ONLY
    # -------------------------
    _, _, test_loader = build_dataloaders(
        csv_path="data_processed/indexed_full_mammogram_images_with_labels.csv",
        splits_dir="splits",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    # -------------------------
    # Load trained model
    # -------------------------
    model = ResNet18SingleView().to(device)
    model.load_state_dict(
        torch.load("resnet18_single_view_best.pt", map_location=device)
    )
    model.eval()

    all_probs = []
    all_labels = []
    all_patient_ids = []

    # -------------------------
    # Inference
    # -------------------------
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            patient_ids = batch["patient_id"]

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)
            all_patient_ids.extend(patient_ids)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    # -------------------------
    # Metrics
    # -------------------------
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, preds)
    sensitivity = recall_score(all_labels, preds)
    specificity = recall_score(all_labels, preds, pos_label=0)

    print("\nTEST SET RESULTS")
    print(f"AUC:          {auc:.3f}")
    print(f"Accuracy:     {accuracy:.3f}")
    print(f"Sensitivity:  {sensitivity:.3f}")
    print(f"Specificity:  {specificity:.3f}")

    # -------------------------
    # Save metrics
    # -------------------------
    metrics_df = pd.DataFrame([{
        "model": "single_view_cc",
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity
    }])

    metrics_df.to_csv(
        "results/single_view_test_metrics.csv",
        index=False
    )

    # -------------------------
    # Save per-sample predictions
    # -------------------------
    preds_df = pd.DataFrame({
        "patient_id": all_patient_ids,
        "true_label": all_labels,
        "predicted_prob": all_probs,
        "predicted_label": preds
    })

    preds_df.to_csv(
        "results/single_view_test_predictions.csv",
        index=False
    )

    print("\nSaved results to results/")


if __name__ == "__main__":
    main()
