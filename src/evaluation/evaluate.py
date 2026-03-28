"""
Comprehensive Model Evaluation
================================
Computes all classification metrics on the test set for the research paper.
Generates metrics tables, predictions, and raw data for visualization.

Usage:
    python -m src.evaluation.evaluate --model_name resnet50
    python -m src.evaluation.evaluate --model_name all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.classifiers import get_model, count_parameters


def load_trained_model(
    model_name: str,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / model_name / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Best val metric: {checkpoint.get('best_metric', '?'):.4f}")
    return model


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on all samples, return (true_labels, predicted_probabilities)."""
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute comprehensive classification metrics."""
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),  # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(labels, preds, zero_division=0),
        "auc_roc": roc_auc_score(labels, probs),
        "auc_pr": average_precision_score(labels, probs),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(labels),
        "positive_samples": int(labels.sum()),
        "negative_samples": int(len(labels) - labels.sum()),
        "threshold": threshold,
    }

    return metrics


def compute_curve_data(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compute ROC and PR curve data points."""
    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(labels, probs)

    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr": {
            "precision": precision_arr.tolist(),
            "recall": recall_arr.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def evaluate_model(
    model_name: str,
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    device: str = "cpu",
    batch_size: int = 64,
) -> dict:
    """
    Full evaluation pipeline for a single model.

    Returns dict with all metrics, curve data, and predictions.
    """
    # Load model
    model = load_trained_model(model_name, checkpoint_dir, device)
    params = count_parameters(model)

    # Create test loader
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_channels=3,
        num_workers=2,
    )

    # Collect predictions
    labels, probs = collect_predictions(model, loaders["test"], device)

    # Compute metrics
    metrics = compute_metrics(labels, probs)
    metrics["model_name"] = model_name
    metrics["parameters_millions"] = params["total_millions"]

    # Compute curves
    curves = compute_curve_data(labels, probs)

    # Measure inference time
    inference_times = measure_inference_time(model, device)
    metrics.update(inference_times)

    # Print results
    print(f"\n{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:      {metrics['auc_pr']:.4f}")
    print(f"  Params:      {params['total_millions']:.1f}M")
    print(f"  Avg Infer:   {metrics['avg_inference_ms']:.2f} ms")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['true_positives']} FN={metrics['false_negatives']}")
    print(f"    FP={metrics['false_positives']} TN={metrics['true_negatives']}")

    # Save results
    results_path = Path(results_dir) / model_name
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_path / "curves.json", "w") as f:
        json.dump(curves, f, indent=2)

    np.save(str(results_path / "predictions.npy"), {
        "labels": labels, "probs": probs
    })

    print(f"\n  Results saved to {results_path}/")
    return {"metrics": metrics, "curves": curves, "labels": labels, "probs": probs}


def measure_inference_time(
    model: nn.Module,
    device: str,
    num_runs: int = 100,
    image_size: int = 224,
) -> dict:
    """Measure average inference time over multiple runs."""
    import time

    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    # Warmup
    for _ in range(10):
        _ = model(dummy)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "avg_inference_ms": np.mean(times),
        "std_inference_ms": np.std(times),
        "min_inference_ms": np.min(times),
        "max_inference_ms": np.max(times),
    }


def generate_comparison_table(results_dir: str = "results") -> str:
    """Generate a LaTeX/Markdown comparison table across all evaluated models."""
    results_path = Path(results_dir)
    rows = []

    for model_dir in sorted(results_path.iterdir()):
        metrics_file = model_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
            rows.append(m)

    if not rows:
        return "No evaluation results found."

    # Markdown table
    header = "| Model | Params (M) | Accuracy | Precision | Recall | F1 | AUC-ROC | Inference (ms) |"
    separator = "|---|---|---|---|---|---|---|---|"
    lines = [header, separator]

    for m in rows:
        line = (
            f"| {m['model_name']} "
            f"| {m.get('parameters_millions', 0):.1f} "
            f"| {m['accuracy']:.4f} "
            f"| {m['precision']:.4f} "
            f"| {m['recall']:.4f} "
            f"| {m['f1']:.4f} "
            f"| {m['auc_roc']:.4f} "
            f"| {m.get('avg_inference_ms', 0):.1f} |"
        )
        lines.append(line)

    table = "\n".join(lines)
    print("\nModel Comparison Table:")
    print(table)

    # Save
    with open(results_path / "comparison_table.md", "w") as f:
        f.write(table)

    return table


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Model to evaluate, or 'all' for all available")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if args.model_name == "all":
        model_names = ["resnet50", "efficientnet_b0", "mobilenetv3_small"]
    else:
        model_names = [args.model_name]

    for name in model_names:
        try:
            evaluate_model(
                model_name=name,
                data_dir=args.data_dir,
                checkpoint_dir=args.checkpoint_dir,
                results_dir=args.results_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")

    # Generate comparison table if evaluating multiple models
    if len(model_names) > 1:
        generate_comparison_table(args.results_dir)


if __name__ == "__main__":
    main()
