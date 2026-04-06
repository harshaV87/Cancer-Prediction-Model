"""
Convert BraTS .nii.gz MRI volumes to 2D PNG slices for iOS app testing.

Loads the FLAIR modality (best for tumor visibility), applies z-score
normalization, extracts all valid axial slices, and saves them as PNG.

Usage:
    # Only test-split patients (model has NEVER seen these — recommended):
    python scripts/nii_to_png.py --input data/raw/BraTS2023 --split test

    # Specific split (train / val / test):
    python scripts/nii_to_png.py --input data/raw/BraTS2023 --split val

    # Single patient folder:
    python scripts/nii_to_png.py --input data/raw/BraTS2023/BraTS-GLI-00000-000

    # Entire dataset folder (processes all patients):
    python scripts/nii_to_png.py --input data/raw/BraTS2023

    # Custom output directory:
    python scripts/nii_to_png.py --input data/raw/BraTS2023 --split test --output my_pngs

    # Extract only the middle 20% of slices (most informative for tumor):
    python scripts/nii_to_png.py --input data/raw/BraTS2023 --split test --middle_only
"""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def parse_split_info(split_info_path: Path, split: str) -> list[str]:
    """
    Parse split_info.txt and return the list of patient IDs for the given split.
    split must be one of: 'train', 'val', 'test'.
    """
    if not split_info_path.exists():
        raise FileNotFoundError(f"split_info.txt not found at: {split_info_path}")

    text = split_info_path.read_text()
    # Match the section header, e.g. "=== test (31 patients) ==="
    pattern = rf"=== {re.escape(split)} \(\d+ patients\) ===(.*?)(?===|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Split '{split}' not found in {split_info_path}")

    patient_ids = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    return patient_ids


def load_and_normalize(filepath: Path) -> np.ndarray:
    """Load a NIfTI volume, z-score normalize, then scale to [0, 255] uint8."""
    img = nib.load(str(filepath))
    volume = img.get_fdata().astype(np.float32)

    # Z-score normalization
    mean, std = volume.mean(), volume.std()
    if std < 1e-8:
        return np.zeros_like(volume, dtype=np.uint8)
    volume = (volume - mean) / std

    # Min-max rescale to [0, 255]
    vmin, vmax = volume.min(), volume.max()
    if (vmax - vmin) < 1e-8:
        return np.zeros_like(volume, dtype=np.uint8)
    volume = (volume - vmin) / (vmax - vmin) * 255.0
    return volume.astype(np.uint8)


def find_modality_file(patient_dir: Path, *suffixes: str) -> Path | None:
    """Find the first matching NIfTI file for any of the given modality suffixes."""
    for suffix in suffixes:
        matches = list(patient_dir.glob(f"*{suffix}.nii.gz"))
        if not matches:
            matches = list(patient_dir.glob(f"*{suffix}.nii"))
        if matches:
            return matches[0]
    return None


def extract_slices_to_png(
    patient_dir: Path,
    output_dir: Path,
    middle_only: bool,
    min_brain_fraction: float = 0.01,
) -> dict:
    """
    Extract 2D axial slices from a patient's FLAIR volume and save as PNG.

    Slices are saved into two subfolders based on ground-truth label:
      <output_dir>/<patient_id>/positive/  — tumor present in that slice
      <output_dir>/<patient_id>/negative/  — no tumor in that slice

    Returns a dict with keys 'positive', 'negative', 'skipped'.
    """
    flair_path = find_modality_file(patient_dir, "-t2f", "-flair", "-t2w")
    if flair_path is None:
        print(f"  WARNING: No FLAIR file found in {patient_dir.name}, skipping.")
        return {"positive": 0, "negative": 0, "skipped": True}

    seg_path = find_modality_file(patient_dir, "-seg")
    if seg_path is None:
        print(f"  WARNING: No segmentation mask found in {patient_dir.name}; "
              "all slices will be saved as 'unknown'.")

    volume = load_and_normalize(flair_path)
    seg_vol = nib.load(str(seg_path)).get_fdata() if seg_path else None
    num_slices = volume.shape[2]

    # Optionally restrict to middle 20% of slices (where tumors appear most often)
    if middle_only:
        lo = int(num_slices * 0.40)
        hi = int(num_slices * 0.60)
        slice_range = range(lo, hi)
    else:
        slice_range = range(num_slices)

    # Create positive / negative output dirs
    pos_dir = output_dir / patient_dir.name / "positive"
    neg_dir = output_dir / patient_dir.name / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    pos_count = neg_count = 0

    for i in slice_range:
        slice_2d = volume[:, :, i]

        # Skip near-empty slices (top/bottom of skull)
        if np.count_nonzero(slice_2d) / slice_2d.size < min_brain_fraction:
            continue

        # Ground-truth label from segmentation mask
        if seg_vol is not None:
            is_positive = bool(np.any(seg_vol[:, :, i] > 0))
        else:
            is_positive = False  # no mask → treat as unknown / negative

        img = Image.fromarray(slice_2d, mode="L")  # 8-bit grayscale
        img = img.convert("RGB")                   # iOS UIImage expects RGB

        out_dir = pos_dir if is_positive else neg_dir
        filename = f"{patient_dir.name}_slice{i:03d}_{'pos' if is_positive else 'neg'}.png"
        img.save(str(out_dir / filename), format="PNG")

        if is_positive:
            pos_count += 1
        else:
            neg_count += 1

    return {"positive": pos_count, "negative": neg_count, "skipped": False}


def main():
    parser = argparse.ArgumentParser(
        description="Convert BraTS .nii.gz volumes to 2D PNG slices"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to a single patient folder (e.g. data/raw/BraTS2023/BraTS-GLI-00000-000) "
            "OR the dataset root folder containing multiple patient folders."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help=(
            "Which split to convert (train / val / test). Reads patient IDs from "
            "data/processed/split_info.txt. Use 'test' to get samples the model "
            "has never seen. Ignored when --input points to a single patient folder."
        ),
    )
    parser.add_argument(
        "--split_info",
        type=str,
        default="data/processed/split_info.txt",
        help="Path to split_info.txt (default: data/processed/split_info.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exported_pngs",
        help="Output directory for PNG files (default: exported_pngs/)",
    )
    parser.add_argument(
        "--middle_only",
        action="store_true",
        help="Only extract the middle 20%% of axial slices (most tumor-relevant).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}")
        return

    # Determine if input is a single patient folder or a dataset root
    has_nii = any(input_path.glob("*.nii.gz")) or any(input_path.glob("*.nii"))
    if has_nii:
        # Single patient folder — ignore --split
        patient_dirs = [input_path]
    elif args.split:
        # Filter by split using split_info.txt
        split_info_path = Path(args.split_info)
        try:
            patient_ids = parse_split_info(split_info_path, args.split)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            return
        patient_dirs = sorted([
            input_path / pid for pid in patient_ids
            if (input_path / pid).is_dir()
        ])
        missing = [pid for pid in patient_ids if not (input_path / pid).is_dir()]
        if missing:
            print(f"  NOTE: {len(missing)} patient folder(s) from split_info.txt not found in --input and will be skipped.")
    else:
        # No split specified — process all subdirectories
        patient_dirs = sorted([
            d for d in input_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    if not patient_dirs:
        print(f"ERROR: No patient directories found under {input_path}")
        return

    print(f"Found {len(patient_dirs)} patient(s) to process.")
    if args.split:
        print(f"Split: {args.split} (model has {'NEVER seen' if args.split == 'test' else 'already seen'} these patients)")
    print(f"Output directory: {output_path.resolve()}")
    print(f"Middle slices only: {args.middle_only}\n")

    total_pos = total_neg = 0
    for patient_dir in patient_dirs:
        counts = extract_slices_to_png(patient_dir, output_path, args.middle_only)
        if not counts["skipped"]:
            p, n = counts["positive"], counts["negative"]
            print(f"  {patient_dir.name}: {p} positive, {n} negative  ({p + n} total)")
            total_pos += p
            total_neg += n

    print(f"\nDone.")
    print(f"  Total positive slices (tumor present): {total_pos}")
    print(f"  Total negative slices (no tumor):      {total_neg}")
    print(f"  Grand total:                           {total_pos + total_neg}")
    print(f"\nLocation: {output_path.resolve()}")
    print("\nFolder structure:")
    print("  exported_pngs/")
    print("  └── <patient_id>/")
    print("      ├── positive/   ← slices where tumor IS present (ground truth)")
    print("      └── negative/   ← slices where NO tumor is present (ground truth)")


if __name__ == "__main__":
    main()
