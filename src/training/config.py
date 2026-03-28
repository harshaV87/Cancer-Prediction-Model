"""
Training Configuration
======================
Hyperparameters and settings for model training.
"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/processed"
    image_size: int = 224
    num_channels: int = 3  # 3 for pretrained models (grayscale repeated)
    batch_size: int = 32
    num_workers: int = 4

    # Model
    model_name: str = "resnet50"  # resnet50, efficientnet_b0, mobilenetv3_small
    pretrained: bool = True

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # adam or sgd

    # SGD-specific
    momentum: float = 0.9

    # Scheduler
    scheduler: str = "cosine"  # cosine, plateau, step
    step_size: int = 10  # for StepLR
    gamma: float = 0.1  # for StepLR
    patience: int = 5  # for ReduceLROnPlateau
    t_max: int = 50  # for CosineAnnealingLR (typically set to num_epochs)

    # Training
    num_epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_auc"  # val_auc, val_loss, val_f1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Logging
    log_dir: str = "runs"  # TensorBoard log dir
    log_interval: int = 10  # Log every N batches

    # Device
    device: str = "auto"  # auto, cpu, mps, cuda

    # Reproducibility
    seed: int = 42

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
