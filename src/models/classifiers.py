"""
Model Architectures for Brain Tumor Classification
====================================================
Provides ResNet-50, EfficientNet-B0, and MobileNetV3-Small
adapted for binary classification on single-channel MRI slices.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_resnet50(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """
    ResNet-50 adapted for binary classification.

    Uses ImageNet pretrained weights, replaces the final FC layer
    with a single output neuron for binary classification (BCEWithLogitsLoss).
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_efficientnet_b0(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """
    EfficientNet-B0 adapted for binary classification.

    Smaller and more efficient than ResNet-50 while maintaining
    competitive accuracy. Good balance for mobile deployment.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def get_mobilenetv3_small(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """
    MobileNetV3-Small adapted for binary classification.

    Extremely lightweight — optimized for mobile/edge deployment.
    Smallest model size and fastest inference on Apple Neural Engine.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


# Registry for easy model selection by name
MODEL_REGISTRY = {
    "resnet50": get_resnet50,
    "efficientnet_b0": get_efficientnet_b0,
    "mobilenetv3_small": get_mobilenetv3_small,
}


def get_model(name: str, pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """
    Get a model by name from the registry.

    Args:
        name: One of 'resnet50', 'efficientnet_b0', 'mobilenetv3_small'
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of output classes (1 for binary with BCEWithLogitsLoss)

    Returns:
        PyTorch model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](pretrained=pretrained, num_classes=num_classes)


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def print_model_summary():
    """Print a comparison table of all available models."""
    print(f"{'Model':<20} {'Total Params':>15} {'Size (MB, est)':>15}")
    print("-" * 55)
    for name in MODEL_REGISTRY:
        model = get_model(name, pretrained=False)
        params = count_parameters(model)
        size_mb = params["total"] * 4 / (1024 * 1024)  # FP32
        print(f"{name:<20} {params['total_millions']:>12.1f}M {size_mb:>12.1f}")


if __name__ == "__main__":
    print_model_summary()

    # Verify forward pass for each model
    print("\nForward pass verification:")
    dummy_input = torch.randn(2, 3, 224, 224)
    for name in MODEL_REGISTRY:
        model = get_model(name, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  {name}: input={list(dummy_input.shape)} → output={list(output.shape)}")
