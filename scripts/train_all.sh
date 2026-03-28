#!/bin/bash
# =============================================================================
# Train All Models
# =============================================================================
# Trains ResNet-50, EfficientNet-B0, and MobileNetV3-Small sequentially.
# Usage: ./scripts/train_all.sh
# =============================================================================

set -e

DATA_DIR="data/processed"
EPOCHS=50
BATCH_SIZE=32
LR=0.0001

echo "============================================="
echo "  Brain Tumor Classification — Train All"
echo "============================================="
echo ""

# 1. ResNet-50
echo ">>> Training ResNet-50..."
python -m src.training.train \
    --model_name resnet50 \
    --data_dir "$DATA_DIR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --optimizer adam \
    --scheduler cosine

echo ""

# 2. EfficientNet-B0
echo ">>> Training EfficientNet-B0..."
python -m src.training.train \
    --model_name efficientnet_b0 \
    --data_dir "$DATA_DIR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --optimizer adam \
    --scheduler cosine

echo ""

# 3. MobileNetV3-Small
echo ">>> Training MobileNetV3-Small..."
python -m src.training.train \
    --model_name mobilenetv3_small \
    --data_dir "$DATA_DIR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --optimizer adam \
    --scheduler cosine

echo ""
echo "============================================="
echo "  All models trained!"
echo "============================================="
echo ""

# 4. Evaluate all models
echo ">>> Evaluating all models on test set..."
python -m src.evaluation.evaluate --model_name all --device cpu

echo ""

# 5. Generate Grad-CAM for each model
echo ">>> Generating Grad-CAM visualizations..."
for MODEL in resnet50 efficientnet_b0 mobilenetv3_small; do
    python -m src.evaluation.grad_cam --model_name "$MODEL" --num_samples 8
done

echo ""

# 6. Generate paper figures
echo ">>> Generating paper figures..."
python -m src.evaluation.visualize

echo ""

# 7. Convert best model to CoreML
echo ">>> Converting ResNet-50 to CoreML (.mlpackage)..."
python -m src.export.convert_coreml --model_name resnet50 --quantize fp16

echo ""

# 8. Validate CoreML model
echo ">>> Validating CoreML model..."
python -m src.export.validate_coreml --model_name resnet50

echo ""
echo "============================================="
echo "  Pipeline complete!"
echo "  Checkpoints:  checkpoints/"
echo "  Results:      results/"
echo "  CoreML:       exported_models/"
echo "============================================="
