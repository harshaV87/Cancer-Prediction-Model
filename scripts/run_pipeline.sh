#!/bin/bash
# =============================================================================
# Quick Start: Run Complete Pipeline
# =============================================================================
# Step 1: Preprocess BraTS data
# Step 2: Train all models
# Step 3: Evaluate, visualize, export
#
# Prerequisites:
#   - Download BraTS 2023 dataset to data/raw/BraTS2023/
#   - pip install -r requirements.txt
#
# Usage: ./scripts/run_pipeline.sh
# =============================================================================

set -e

BRATS_DIR="data/raw/BraTS2023"
PROCESSED_DIR="data/processed"

echo "============================================="
echo "  Brain Tumor Classification Pipeline"
echo "============================================="
echo ""

# Check if raw data exists
if [ ! -d "$BRATS_DIR" ]; then
    echo "ERROR: BraTS dataset not found at $BRATS_DIR"
    echo ""
    echo "Please download BraTS 2023 GLI dataset:"
    echo "  1. Register at https://www.synapse.org/"
    echo "  2. Go to: https://www.synapse.org/Synapse:syn51156910/wiki/622351"
    echo "  3. Download and extract to: $BRATS_DIR"
    echo ""
    echo "Expected structure:"
    echo "  $BRATS_DIR/"
    echo "    BraTS-GLI-00000-000/"
    echo "      BraTS-GLI-00000-000-t2f.nii.gz   (FLAIR)"
    echo "      BraTS-GLI-00000-000-seg.nii.gz    (Segmentation mask)"
    echo "      ..."
    exit 1
fi

# Step 1: Preprocess
if [ ! -d "$PROCESSED_DIR/train" ]; then
    echo ">>> Step 1: Preprocessing BraTS data..."
    python -m src.data.preprocess --data_dir "$BRATS_DIR" --output_dir "$PROCESSED_DIR"
else
    echo ">>> Step 1: Preprocessed data already exists, skipping."
fi

echo ""

# Step 2-8: Train, evaluate, export
echo ">>> Steps 2-8: Training, evaluation, and export..."
bash scripts/train_all.sh
