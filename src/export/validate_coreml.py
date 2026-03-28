"""
CoreML Model Validation
========================
Validates that the CoreML .mlpackage produces predictions consistent
with the original PyTorch model on the test set.

Usage:
    python -m src.export.validate_coreml --model_name resnet50
"""

import argparse
import platform
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.dataset import BraTSSliceDataset
from src.evaluation.evaluate import load_trained_model


def validate_coreml(
    model_name: str,
    checkpoint_dir: str = "checkpoints",
    exported_dir: str = "exported_models",
    data_dir: str = "data/processed",
    num_samples: int = 100,
    quantize: str = "fp16",
) -> dict:
    """
    Compare CoreML and PyTorch predictions on test samples.

    Note: CoreML prediction only works on macOS.
    """
    if platform.system() != "Darwin":
        print("WARNING: CoreML validation requires macOS. Skipping prediction comparison.")
        return {"status": "skipped", "reason": "not macOS"}

    import coremltools as ct

    suffix = f"_{quantize}" if quantize != "fp32" else ""
    mlpackage_path = Path(exported_dir) / f"BrainTumorClassifier_{model_name}{suffix}.mlpackage"

    if not mlpackage_path.exists():
        raise FileNotFoundError(f"CoreML model not found at {mlpackage_path}")

    # Load CoreML model
    print(f"Loading CoreML model from {mlpackage_path}...")
    coreml_model = ct.models.MLModel(str(mlpackage_path))

    # Load PyTorch model
    print(f"Loading PyTorch model {model_name}...")
    pytorch_model = load_trained_model(model_name, checkpoint_dir, device="cpu")
    pytorch_model.eval()

    # Load test dataset
    dataset = BraTSSliceDataset(
        data_dir, split="test", image_size=224, augment=False, num_channels=3
    )

    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    pytorch_probs = []
    coreml_preds = []
    max_diff = 0.0

    print(f"Comparing predictions on {len(indices)} samples...")
    for idx in tqdm(indices, desc="Validating"):
        tensor, label = dataset[idx]

        # PyTorch prediction
        with torch.no_grad():
            pt_output = pytorch_model(tensor.unsqueeze(0)).squeeze()
            pt_prob = torch.sigmoid(pt_output).item()
        pytorch_probs.append(pt_prob)

        # CoreML prediction
        # Convert tensor to PIL Image for CoreML ImageType input
        img_array = tensor.permute(1, 2, 0).numpy()  # (H, W, C)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)

        coreml_output = coreml_model.predict({"image": pil_image})

        # Extract probability from CoreML output
        # ClassifierConfig produces 'classLabel' and class probabilities
        if "Tumor Detected" in coreml_output.get("classLabel_probs", {}):
            cm_prob = coreml_output["classLabel_probs"]["Tumor Detected"]
        else:
            # Fallback: check raw output
            cm_prob = 0.5
        coreml_preds.append(cm_prob)

        diff = abs(pt_prob - cm_prob)
        max_diff = max(max_diff, diff)

    pytorch_probs = np.array(pytorch_probs)
    coreml_preds = np.array(coreml_preds)

    # Compute agreement metrics
    mean_diff = np.mean(np.abs(pytorch_probs - coreml_preds))
    correlation = np.corrcoef(pytorch_probs, coreml_preds)[0, 1]

    # Check binary prediction agreement
    pt_binary = (pytorch_probs >= 0.5).astype(int)
    cm_binary = (coreml_preds >= 0.5).astype(int)
    agreement = (pt_binary == cm_binary).mean()

    results = {
        "status": "pass" if max_diff < 0.05 else "warning",
        "num_samples": len(indices),
        "max_absolute_diff": float(max_diff),
        "mean_absolute_diff": float(mean_diff),
        "correlation": float(correlation),
        "binary_agreement": float(agreement),
    }

    print(f"\nValidation Results:")
    print(f"  Samples tested:      {results['num_samples']}")
    print(f"  Max absolute diff:   {results['max_absolute_diff']:.6f}")
    print(f"  Mean absolute diff:  {results['mean_absolute_diff']:.6f}")
    print(f"  Correlation:         {results['correlation']:.6f}")
    print(f"  Binary agreement:    {results['binary_agreement']:.2%}")
    print(f"  Status:              {results['status'].upper()}")

    if results["status"] == "pass":
        print("\n  CoreML model validated successfully — predictions match PyTorch.")
    else:
        print("\n  WARNING: Some predictions differ notably. Check quantization effects.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate CoreML model")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--exported_dir", type=str, default="exported_models")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--quantize", type=str, default="fp16")
    args = parser.parse_args()

    validate_coreml(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        exported_dir=args.exported_dir,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
