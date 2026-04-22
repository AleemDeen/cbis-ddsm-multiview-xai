"""
Model comparison graph generator.

Run from project root:
    python generate_graphs.py

Evaluates all four models on the test split and saves comparison PNGs to graphs/.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch

from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
GRAPHS_DIR = ROOT / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
COLOURS = {
    "SV Baseline": "#4C72B0",
    "SV Best":     "#55A868",
    "MV Baseline": "#C44E52",
    "MV Best":     "#DD8452",
}
MODEL_ORDER = ["SV Baseline", "SV Best", "MV Baseline", "MV Best"]

# Hard Dice scores for the best seg model (malignant subset, from eval script)
SEG_DICE = {
    "CC Hard Dice\n(malignant)":       0.4719,
    "MLO Hard Dice\n(malignant)":      0.5129,
    "MLO Hard Dice\n(true positives)": 0.5545,
}


# ── Data helpers ───────────────────────────────────────────────────────────

def load_test_cases():
    with open(ROOT / "splits" / "test_cases.txt") as f:
        return set(line.strip() for line in f if line.strip())


def get_single_view_loader():
    from src.data.dataloaders import build_dataloaders
    pt_csv  = ROOT / "data_processed" / "indexed_full_mammogram_images_with_labels_pt.csv"
    raw_csv = ROOT / "data_processed" / "indexed_full_mammogram_images_with_labels.csv"
    csv     = str(pt_csv if pt_csv.exists() else raw_csv)
    _, _, loader = build_dataloaders(
        csv_path=csv, splits_dir=str(ROOT / "splits"),
        batch_size=32, num_workers=0, pin_memory=False,
    )
    return loader


def get_multi_view_loader():
    from torch.utils.data import DataLoader
    from src.data.multi_view_dataset import CBISDDSMMultiViewDataset
    pt_csv  = ROOT / "data_processed" / "indexed_multi_view_cases_pt.csv"
    raw_csv = ROOT / "data_processed" / "indexed_multi_view_cases.csv"
    csv     = str(pt_csv if pt_csv.exists() else raw_csv)
    df      = pd.read_csv(csv)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")
    test_df = df[df["case_id"].isin(load_test_cases())].reset_index(drop=True)
    return DataLoader(CBISDDSMMultiViewDataset(test_df),
                      batch_size=8, shuffle=False, num_workers=0)


# ── Inference ──────────────────────────────────────────────────────────────

def run_single_view(model_path):
    from src.models.resnet18_single_view import ResNet18SingleView
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet18SingleView().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in get_single_view_loader():
            imgs   = batch["image"].to(device)
            labels = batch["label"].view(-1).cpu().numpy()
            probs  = torch.sigmoid(model(imgs)).view(-1).cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.tolist())
    return np.array(labels_all), np.array(probs_all)


def run_multi_view(model_path, seg=False):
    from src.models.resnet18_multi_view import ResNet18MultiView, ResNet18MultiViewSeg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = (ResNet18MultiViewSeg() if seg else ResNet18MultiView()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in get_multi_view_loader():
            cc     = batch["cc_image"].to(device)
            mlo    = batch["mlo_image"].to(device)
            labels = batch["label"].float().cpu().numpy()
            logits = model(cc, mlo)                        # plain logits only
            probs  = torch.sigmoid(logits).view(-1).cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.tolist())
    return np.array(labels_all), np.array(probs_all)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs, threshold=0.5):
    preds              = (probs >= threshold).astype(int)
    tn, fp, fn, tp     = confusion_matrix(labels, preds).ravel()
    return dict(
        AUC         = roc_auc_score(labels, probs),
        Accuracy    = accuracy_score(labels, preds),
        Precision   = precision_score(labels, preds, zero_division=0),
        Recall      = recall_score(labels, preds, zero_division=0),
        F1          = f1_score(labels, preds, zero_division=0),
        Specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        _labels=labels, _probs=probs, _preds=preds,
        _cm=np.array([[tn, fp], [fn, tp]]),
        _n=len(labels),
    )


# ── Plotting helpers ───────────────────────────────────────────────────────

def _bar_style(ax):
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    path = GRAPHS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ── Graph 1: Metric comparison bar chart ──────────────────────────────────

def plot_metric_bars(results):
    metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1", "Specificity"]
    n_m, n_mod = len(metrics), len(MODEL_ORDER)
    x     = np.arange(n_m)
    width = 0.18
    offsets = np.linspace(-(n_mod - 1) / 2, (n_mod - 1) / 2, n_mod) * width

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(MODEL_ORDER):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=name,
                      color=COLOURS[name], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Classification Metrics — All Models", fontsize=12, fontweight="bold", pad=10)
    _bar_style(ax)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "metric_comparison_bars.png")


# ── Graph 2: ROC curves ────────────────────────────────────────────────────

def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random (AUC = 0.50)")

    for name in MODEL_ORDER:
        r   = results[name]
        fpr, tpr, _ = roc_curve(r["_labels"], r["_probs"])
        ax.plot(fpr, tpr, color=COLOURS[name], linewidth=2,
                label=f"{name}  (AUC = {r['AUC']:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — All Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "roc_curves.png")


# ── Graph 3: Confusion matrices ────────────────────────────────────────────

def plot_confusion_matrices(results):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.suptitle("Confusion Matrices (threshold = 0.5)", fontsize=13, fontweight="bold", y=1.01)

    for ax, name in zip(axes.flat, MODEL_ORDER):
        cm   = results[name]["_cm"]           # [[TN, FP],[FN, TP]]
        norm = cm.astype(float) / cm.sum()

        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nBenign", "Predicted\nMalignant"], fontsize=8)
        ax.set_yticklabels(["Actual\nBenign", "Actual\nMalignant"], fontsize=8)

        labels_cm = [["TN", "FP"], ["FN", "TP"]]
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{labels_cm[r][c]}\n{cm[r, c]}\n({norm[r,c]:.1%})",
                        ha="center", va="center", fontsize=9,
                        color="white" if norm[r, c] > 0.5 else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save(fig, "confusion_matrices.png")


# ── Graph 4: AUC + Recall focus bar chart ─────────────────────────────────

def plot_auc_recall(results):
    metrics = ["AUC", "Recall", "Specificity"]
    n_m, n_mod = len(metrics), len(MODEL_ORDER)
    x       = np.arange(n_m)
    width   = 0.18
    offsets = np.linspace(-(n_mod - 1) / 2, (n_mod - 1) / 2, n_mod) * width

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, name in enumerate(MODEL_ORDER):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=name,
                      color=COLOURS[name], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("AUC, Recall & Specificity — All Models", fontsize=12, fontweight="bold", pad=10)
    _bar_style(ax)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "auc_recall_specificity.png")


# ── Graph 5: Seg model Dice scores ─────────────────────────────────────────

def plot_dice_scores():
    labels = list(SEG_DICE.keys())
    values = list(SEG_DICE.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=COLOURS["MV Best"],
                  alpha=0.88, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Hard Dice Score", fontsize=10)
    ax.set_title("MV Best — Localisation Hard Dice Scores", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    save(fig, "seg_dice_scores.png")


# ── Graph 6: Summary table ─────────────────────────────────────────────────

def plot_summary_table(results):
    metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1", "Specificity"]
    rows    = []
    for name in MODEL_ORDER:
        r    = results[name]
        row  = [name, r["_n"]] + [f"{r[m]:.4f}" for m in metrics]
        rows.append(row)

    col_labels = ["Model", "N"] + metrics

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")

    tbl = ax.table(
        cellText  = rows,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.0)

    # Header row styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Model colour stripe
    for i, name in enumerate(MODEL_ORDER, start=1):
        tbl[i, 0].set_facecolor(COLOURS[name])
        tbl[i, 0].set_text_props(color="white", fontweight="bold")
        for j in range(1, len(col_labels)):
            tbl[i, j].set_facecolor("#F7F9FA" if i % 2 == 0 else "white")

    fig.suptitle("Classification Metrics Summary", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    save(fig, "summary_table.png")


# ── Graph 7: Single vs Multi-view AUC comparison ──────────────────────────

def plot_sv_vs_mv(results):
    """Side-by-side AUC bar chart highlighting the single→multi gain."""
    sv_names = ["SV Baseline", "SV Best"]
    mv_names = ["MV Baseline", "MV Best"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle("Single-View vs Multi-View: AUC Comparison", fontsize=12,
                 fontweight="bold", y=1.01)

    for ax, group, title in zip(axes, [sv_names, mv_names],
                                 ["Single-View Models", "Multi-View Models"]):
        names = group
        vals  = [results[n]["AUC"] for n in names]
        bars  = ax.bar(names, vals, color=[COLOURS[n] for n in names],
                       alpha=0.88, edgecolor="white", linewidth=0.5, width=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylim(0.6, 0.95)
        ax.set_ylabel("AUC" if ax == axes[0] else "", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    save(fig, "sv_vs_mv_auc.png")


# ── Main ───────────────────────────────────────────────────────────────────

EVAL_CONFIG = {
    "SV Baseline": (run_single_view, "models/sv_baseline.pt",   {}),
    "SV Best":     (run_single_view, "models/sv_best.pt",       {}),
    "MV Baseline": (run_multi_view,  "models/mv_baseline.pt",   {"seg": False}),
    "MV Best":     (run_multi_view,  "models/mv_best.pt",       {"seg": True}),
}


def main():
    print("=" * 60)
    print("CBIS-DDSM Model Comparison — Graph Generator")
    print("=" * 60)

    results = {}
    for name in MODEL_ORDER:
        fn, path, kwargs = EVAL_CONFIG[name]
        full_path = ROOT / path
        if not full_path.exists():
            print(f"  [SKIP] {name}: model file not found at {path}")
            continue
        print(f"\n  Evaluating {name}  ({path}) ...")
        labels, probs = fn(str(full_path), **kwargs)
        results[name] = compute_metrics(labels, probs)
        r = results[name]
        print(f"    n={r['_n']}  AUC={r['AUC']:.4f}  Acc={r['Accuracy']:.4f}"
              f"  Prec={r['Precision']:.4f}  Rec={r['Recall']:.4f}"
              f"  F1={r['F1']:.4f}  Spec={r['Specificity']:.4f}")

    if len(results) < 2:
        print("\nNot enough models evaluated to produce comparison graphs.")
        return

    print(f"\nGenerating graphs -> {GRAPHS_DIR}/")
    plot_metric_bars(results)
    plot_roc_curves(results)
    plot_confusion_matrices(results)
    plot_auc_recall(results)
    plot_dice_scores()
    plot_summary_table(results)
    plot_sv_vs_mv(results)

    print(f"\nDone - {len(list(GRAPHS_DIR.glob('*.png')))} PNG(s) saved to {GRAPHS_DIR}/")


if __name__ == "__main__":
    main()
