"""
PyTorch to CoreML Conversion
==============================
Converts trained PyTorch models to .mlpackage format for iOS deployment.
Supports Float32 and Float16 quantization.

Usage:
    python -m src.export.convert_coreml --model_name resnet50
    python -m src.export.convert_coreml --model_name resnet50 --quantize fp16
"""

import argparse
from pathlib import Path

import coremltools as ct
import torch

from src.models.classifiers import get_model
from src.evaluation.evaluate import load_trained_model


def convert_to_coreml(
    model_name: str,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "exported_models",
    image_size: int = 224,
    quantize: str = "fp16",  # fp32, fp16
    minimum_ios: str = "16",
) -> str:
    """
    Convert a trained PyTorch model to CoreML .mlpackage format.

    Args:
        model_name: Name of the model to convert
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save the .mlpackage
        image_size: Input image size
        quantize: Quantization mode ('fp32' or 'fp16')
        minimum_ios: Minimum iOS deployment target

    Returns:
        Path to the saved .mlpackage
    """
    device = "cpu"  # CoreML conversion must be done on CPU

    # Load trained model
    print(f"Loading trained {model_name}...")
    model = load_trained_model(model_name, checkpoint_dir, device)
    model.eval()

    # Trace the model with a dummy input
    print(f"Tracing model with input shape (1, 3, {image_size}, {image_size})...")
    dummy_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(model, dummy_input)

    # Convert to CoreML
    print("Converting to CoreML...")

    # Map iOS version string to target
    ios_target_map = {
        "15": ct.target.iOS15,
        "16": ct.target.iOS16,
        "17": ct.target.iOS17,
    }
    deployment_target = ios_target_map.get(minimum_ios, ct.target.iOS16)

    # Set compute precision based on quantization
    compute_precision = ct.precision.FLOAT16 if quantize == "fp16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1.0 / 255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        convert_to="mlprogram",
        minimum_deployment_target=deployment_target,
        compute_precision=compute_precision,
    )

    # Set model metadata
    mlmodel.author = "Brain Tumor Classification Research"
    mlmodel.short_description = (
        f"Binary brain tumor classifier ({model_name}) trained on BraTS 2023 MRI dataset. "
        "Predicts whether an MRI slice contains a tumor. NOT for clinical use."
    )
    mlmodel.version = "1.0.0"
    mlmodel.license = "Research Use Only"

    # Add input/output descriptions
    mlmodel.input_description["image"] = "Brain MRI axial slice (224x224 RGB)"
    if "var_840" in mlmodel.output_description:
        mlmodel.output_description["var_840"] = "Tumor probability (0.0-1.0, after sigmoid). >0.5 = Tumor"
    elif len(mlmodel.output_description) > 0:
        # Set description for the first output
        first_output = list(mlmodel.output_description.keys())[0]
        mlmodel.output_description[first_output] = "Tumor probability (0.0-1.0, after sigmoid). >0.5 = Tumor"

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_{quantize}" if quantize != "fp32" else ""
    filename = f"BrainTumorClassifier_{model_name}{suffix}.mlpackage"
    save_path = output_path / filename

    mlmodel.save(str(save_path))

    # Print model info
    file_size_mb = sum(
        f.stat().st_size for f in save_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    print(f"\nConversion complete!")
    print(f"  Model:       {model_name}")
    print(f"  Quantize:    {quantize}")
    print(f"  iOS target:  {minimum_ios}+")
    print(f"  File size:   {file_size_mb:.1f} MB")
    print(f"  Saved to:    {save_path}")

    return str(save_path)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to CoreML")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "mobilenetv3_small"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="exported_models")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--quantize", type=str, default="fp16",
                        choices=["fp32", "fp16"])
    parser.add_argument("--minimum_ios", type=str, default="16",
                        choices=["15", "16", "17"])
    args = parser.parse_args()

    convert_to_coreml(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        quantize=args.quantize,
        minimum_ios=args.minimum_ios,
    )


if __name__ == "__main__":
    main()
