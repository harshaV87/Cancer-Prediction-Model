"""
Grad-CAM Visualization
=======================
Generates Gradient-weighted Class Activation Maps to visualize
which regions of the MRI slice the model focuses on for its prediction.
Essential for interpretability in the research paper.

Usage:
    python -m src.evaluation.grad_cam --model_name resnet50 --num_samples 20
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.data.dataset import BraTSSliceDataset
from src.evaluation.evaluate import load_trained_model


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.

    Computes gradient of the target class score w.r.t. the feature maps
    of a target convolutional layer, producing a heatmap highlighting
    discriminative regions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)

        Returns:
            Heatmap as numpy array (H, W) in range [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # For binary classification, use the single output logit
        target = output.squeeze()

        # Backward pass
        self.model.zero_grad()
        target.backward()

        # Compute weights: global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Get the appropriate target layer for Grad-CAM based on model architecture."""
    if model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    elif model_name == "mobilenetv3_small":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    # Overlay
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def generate_gradcam_grid(
    model_name: str,
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    device: str = "cpu",
    num_samples: int = 16,
):
    """
    Generate Grad-CAM visualizations for a grid of samples.

    Creates a figure with columns:
    [Original Image | Grad-CAM Heatmap | Overlay] for each sample.
    Includes both true positive and true negative examples.
    """
    model = load_trained_model(model_name, checkpoint_dir, device)
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    # Load test dataset (no augmentation)
    dataset = BraTSSliceDataset(
        data_dir, split="test", image_size=224, augment=False, num_channels=3
    )

    # Collect samples: balanced between positive and negative
    half = num_samples // 2
    pos_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
    neg_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]

    np.random.seed(42)
    selected_pos = np.random.choice(pos_indices, min(half, len(pos_indices)), replace=False)
    selected_neg = np.random.choice(neg_indices, min(half, len(neg_indices)), replace=False)
    selected = list(selected_pos) + list(selected_neg)

    # Generate figure
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Original", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("Grad-CAM", fontsize=14, fontweight="bold")
    axes[0, 2].set_title("Overlay", fontsize=14, fontweight="bold")

    for row, idx in enumerate(selected):
        tensor, label = dataset[idx]
        input_tensor = tensor.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob >= 0.5 else 0

        # Generate Grad-CAM
        input_for_cam = tensor.unsqueeze(0).to(device)
        heatmap = grad_cam.generate(input_for_cam)

        # Original image (take first channel since channels are repeated)
        original = tensor[0].numpy()

        # Overlay
        overlay = overlay_heatmap(original, heatmap, alpha=0.4)

        # Plot
        label_str = "Tumor" if label == 1 else "No Tumor"
        pred_str = f"Pred: {prob:.2f}"
        correct = "✓" if pred == label else "✗"
        color = "green" if pred == label else "red"

        axes[row, 0].imshow(original, cmap="gray")
        axes[row, 0].set_ylabel(f"{label_str}\n{pred_str} {correct}", fontsize=11, color=color)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

    plt.suptitle(f"Grad-CAM Visualization — {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save
    output_dir = Path(results_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "gradcam_grid.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM grid saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    generate_gradcam_grid(
        model_name=args.model_name,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        device=args.device,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
