"""
Training Pipeline
=================
Full training loop with validation, early stopping, checkpointing,
and TensorBoard logging.

Usage:
    python -m src.training.train --model_name resnet50 --num_epochs 50
    python -m src.training.train --model_name efficientnet_b0
    python -m src.training.train --model_name mobilenetv3_small
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.classifiers import get_model, count_parameters
from src.training.config import TrainConfig


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    """Create learning rate scheduler from config."""
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.t_max
        )
    elif config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=config.patience, factor=0.5
        )
    elif config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    writer: SummaryWriter,
    epoch: int,
    log_interval: int,
) -> dict:
    """Train for one epoch. Returns dict with loss and metrics."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        # Collect predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Log to TensorBoard
        global_step = epoch * len(loader) + batch_idx
        if batch_idx % log_interval == 0:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)

    # Compute epoch metrics
    avg_loss = running_loss / num_batches
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    preds = (all_probs >= 0.5).astype(int)
    accuracy = (preds == all_labels).mean()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    f1 = f1_score(all_labels, preds, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> dict:
    """Validate the model. Returns dict with loss and metrics."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / num_batches
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    preds = (all_probs >= 0.5).astype(int)
    accuracy = (preds == all_labels).mean()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    f1 = f1_score(all_labels, preds, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
    }


def train(config: TrainConfig):
    """Full training pipeline."""
    set_seed(config.seed)

    device = config.get_device()
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nLoading datasets...")
    loaders = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_channels=config.num_channels,
        num_workers=config.num_workers,
    )

    # Create model
    print(f"\nCreating model: {config.model_name}")
    model = get_model(config.model_name, pretrained=config.pretrained)
    model = model.to(device)
    params = count_parameters(model)
    print(f"  Parameters: {params['total_millions']:.1f}M total, {params['trainable_millions']:.1f}M trainable")

    # Loss function with class weights
    pos_weight = loaders["class_weights"].to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Positive class weight: {pos_weight.item():.2f}")

    # Optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Checkpointing
    checkpoint_dir = Path(config.checkpoint_dir) / config.model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=f"{config.log_dir}/{config.model_name}")

    # Training loop
    best_metric = 0.0
    patience_counter = 0
    history = {"train": [], "val": []}

    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("=" * 70)

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer,
            device, writer, epoch, config.log_interval,
        )

        # Validate
        val_metrics = validate(
            model, loaders["val"], criterion, device, epoch,
        )

        epoch_time = time.time() - epoch_start

        # Log metrics
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("train/auc", train_metrics["auc"], epoch)
        writer.add_scalar("train/f1", train_metrics["f1"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/auc", val_metrics["auc"], epoch)
        writer.add_scalar("val/f1", val_metrics["f1"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Print epoch summary
        print(
            f"Epoch {epoch+1:3d}/{config.num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['auc']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"{epoch_time:.1f}s"
        )

        # Scheduler step
        if config.scheduler == "plateau":
            scheduler.step(val_metrics["auc"])
        else:
            scheduler.step()

        # Early stopping check
        metric_key = config.early_stopping_metric.replace("val_", "")
        current_metric = val_metrics[metric_key]

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            # Save best model
            if config.save_best_only:
                best_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config.__dict__,
                    "val_metrics": val_metrics,
                }, str(best_path))
                print(f"  ✓ Saved best model (val_{metric_key}={best_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={config.early_stopping_patience})")
                break

    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, str(final_path))

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    writer.close()

    print("=" * 70)
    print(f"Training complete. Best val_{metric_key}: {best_metric:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"History saved to: {history_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "mobilenetv3_small"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "step"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        device=args.device,
        seed=args.seed,
        t_max=args.num_epochs,
    )

    print("Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()

    train(config)


if __name__ == "__main__":
    main()
