"""
Visualization Module for Research Paper Figures
=================================================
Generates publication-quality plots: ROC curves, PR curves,
confusion matrices, and training history.

Usage:
    python -m src.evaluation.visualize --results_dir results
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# Use a clean style for papers
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def plot_roc_curves(results_dir: str = "results"):
    """
    Plot ROC curves for all evaluated models on a single figure.
    Includes AUC in the legend.
    """
    results_path = Path(results_dir)
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    color_idx = 0

    for model_dir in sorted(results_path.iterdir()):
        curves_file = model_dir / "curves.json"
        metrics_file = model_dir / "metrics.json"
        if not curves_file.exists():
            continue

        with open(curves_file) as f:
            curves = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)

        fpr = curves["roc"]["fpr"]
        tpr = curves["roc"]["tpr"]
        auc_val = metrics["auc_roc"]
        name = metrics["model_name"]

        ax.plot(fpr, tpr, color=colors[color_idx % len(colors)],
                linewidth=2, label=f"{name} (AUC = {auc_val:.4f})")
        color_idx += 1

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    save_path = results_path / "roc_curves.png"
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curves saved to {save_path}")


def plot_pr_curves(results_dir: str = "results"):
    """Plot Precision-Recall curves for all models."""
    results_path = Path(results_dir)
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    color_idx = 0

    for model_dir in sorted(results_path.iterdir()):
        curves_file = model_dir / "curves.json"
        metrics_file = model_dir / "metrics.json"
        if not curves_file.exists():
            continue

        with open(curves_file) as f:
            curves = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)

        precision = curves["pr"]["precision"]
        recall = curves["pr"]["recall"]
        ap = metrics["auc_pr"]
        name = metrics["model_name"]

        ax.plot(recall, precision, color=colors[color_idx % len(colors)],
                linewidth=2, label=f"{name} (AP = {ap:.4f})")
        color_idx += 1

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    save_path = results_path / "pr_curves.png"
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PR curves saved to {save_path}")


def plot_confusion_matrices(results_dir: str = "results"):
    """Plot confusion matrices for all evaluated models side by side."""
    results_path = Path(results_dir)
    model_data = []

    for model_dir in sorted(results_path.iterdir()):
        metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            m = json.load(f)

        cm = np.array([
            [m["true_negatives"], m["false_positives"]],
            [m["false_negatives"], m["true_positives"]],
        ])
        model_data.append((m["model_name"], cm))

    if not model_data:
        print("No evaluation results found.")
        return

    n = len(model_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    class_names = ["No Tumor", "Tumor"]

    for ax, (name, cm) in zip(axes, model_data):
        # Normalize for percentages
        cm_pct = cm.astype(float) / cm.sum() * 100

        # Plot with both counts and percentages
        labels = np.array([
            [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
            for i in range(2)
        ])

        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=True, square=True,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"{name}")

    plt.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = results_path / "confusion_matrices.png"
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrices saved to {save_path}")


def plot_training_history(
    model_name: str,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
):
    """Plot training and validation loss/accuracy/AUC curves."""
    history_path = Path(checkpoint_dir) / model_name / "training_history.json"
    if not history_path.exists():
        print(f"No training history found at {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, [h["loss"] for h in history["train"]], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, [h["loss"] for h in history["val"]], "r-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [h["accuracy"] for h in history["train"]], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, [h["accuracy"] for h in history["val"]], "r-", label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(epochs, [h["auc"] for h in history["train"]], "b-", label="Train", linewidth=2)
    axes[2].plot(epochs, [h["auc"] for h in history["val"]], "r-", label="Validation", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC-ROC")
    axes[2].set_title("Training & Validation AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Training History — {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_dir = Path(results_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "training_history.png"
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training history saved to {save_path}")


def generate_all_figures(results_dir: str = "results", checkpoint_dir: str = "checkpoints"):
    """Generate all visualization figures for the paper."""
    print("Generating paper figures...")
    plot_roc_curves(results_dir)
    plot_pr_curves(results_dir)
    plot_confusion_matrices(results_dir)

    # Training history for each model
    results_path = Path(results_dir)
    for model_dir in sorted(results_path.iterdir()):
        if model_dir.is_dir():
            plot_training_history(model_dir.name, checkpoint_dir, results_dir)

    print("\nAll figures generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate paper visualizations")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
