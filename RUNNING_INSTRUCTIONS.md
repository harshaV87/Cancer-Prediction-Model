# Running Instructions

Complete guide to set up, train, evaluate, and deploy the Brain Tumor Classification model.

---

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.11+**
- **Xcode 15+** (for iOS app)
- **BraTS 2023 GLI dataset** (download instructions below)

---

## Step 1: Python Environment Setup

```bash
cd Cancer-Prediction-Model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Verify PyTorch MPS support:**
```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## Step 2: Download BraTS 2023 Dataset

1. Register at [Synapse](https://www.synapse.org/)
2. Navigate to the [BraTS 2023 GLI Challenge](https://www.synapse.org/Synapse:syn51156910/wiki/622351)
3. Download the training dataset
4. Extract into `data/raw/BraTS2023/`

**Expected directory structure:**
```
data/raw/BraTS2023/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t1c.nii.gz    (T1 contrast-enhanced)
│   ├── BraTS-GLI-00000-000-t1n.nii.gz    (T1 native)
│   ├── BraTS-GLI-00000-000-t2f.nii.gz    (FLAIR — used for training)
│   ├── BraTS-GLI-00000-000-t2w.nii.gz    (T2 weighted)
│   └── BraTS-GLI-00000-000-seg.nii.gz    (Segmentation mask)
├── BraTS-GLI-00002-000/
│   └── ...
└── ... (~1,250 patient folders)
```

---

## Step 3: Preprocess Data

Extracts 2D axial slices from 3D MRI volumes and creates patient-level train/val/test split.

```bash
python -m src.data.preprocess \
    --data_dir data/raw/BraTS2023 \
    --output_dir data/processed
```

**Output structure:**
```
data/processed/
├── train/
│   ├── positive/   (slices with tumor)
│   └── negative/   (slices without tumor)
├── val/
│   ├── positive/
│   └── negative/
├── test/
│   ├── positive/
│   └── negative/
└── split_info.txt  (patient-level split record)
```

---

## Step 4: Train Models

### Train a single model:
```bash
python -m src.training.train \
    --model_name resnet50 \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --optimizer adam \
    --scheduler cosine
```

### Train all three models:
```bash
./scripts/train_all.sh
```

**Available models:** `resnet50`, `efficientnet_b0`, `mobilenetv3_small`

### Monitor training with TensorBoard:
```bash
tensorboard --logdir runs/
```

**Checkpoints are saved to:** `checkpoints/<model_name>/best_model.pth`

---

## Step 5: Evaluate Models

### Evaluate a single model:
```bash
python -m src.evaluation.evaluate --model_name resnet50 --device cpu
```

### Evaluate all trained models + generate comparison table:
```bash
python -m src.evaluation.evaluate --model_name all --device cpu
```

**Outputs:** `results/<model_name>/metrics.json`, `results/comparison_table.md`

---

## Step 6: Generate Visualizations (for Paper)

### Grad-CAM attention maps:
```bash
python -m src.evaluation.grad_cam \
    --model_name resnet50 \
    --num_samples 8
```

### ROC curves, PR curves, confusion matrices, training history:
```bash
python -m src.evaluation.visualize
```

**All figures saved to:** `results/`

---

## Step 7: Convert to CoreML (.mlpackage)

### Convert with Float16 quantization:
```bash
python -m src.export.convert_coreml \
    --model_name resnet50 \
    --quantize fp16 \
    --minimum_ios 16
```

### Validate CoreML vs PyTorch output consistency:
```bash
python -m src.export.validate_coreml --model_name resnet50
```

**Output:** `exported_models/BrainTumorClassifier_resnet50_fp16.mlpackage`

---

## Step 8: iOS App Setup

1. Open **Xcode** → Create new project → **App** → SwiftUI → Name: `CancerPredictor`
2. Copy all `.swift` files from `ios/CancerPredictor/CancerPredictor/` into the Xcode project
3. Drag the `.mlpackage` file from `exported_models/` into the Xcode project navigator
   - Xcode will auto-generate a Swift class for the model
4. Update `MLService.swift` line with `forResource:` to match the auto-generated model class name
5. Set minimum deployment target to **iOS 16.0**
6. Add `NSCameraUsageDescription` to `Info.plist`: `"Camera access is needed to capture MRI images for analysis."`
7. Build and run on a real iPhone (CoreML + camera don't work on Simulator)

---

## One-Command Full Pipeline

If you have the BraTS dataset downloaded, run everything at once:

```bash
./scripts/run_pipeline.sh
```

This will: preprocess → train all models → evaluate → generate figures → convert to CoreML → validate.

---

## Command Reference

| Task | Command |
|---|---|
| Preprocess data | `python -m src.data.preprocess --data_dir data/raw/BraTS2023` |
| Train ResNet-50 | `python -m src.training.train --model_name resnet50` |
| Train EfficientNet-B0 | `python -m src.training.train --model_name efficientnet_b0` |
| Train MobileNetV3 | `python -m src.training.train --model_name mobilenetv3_small` |
| Train all models | `./scripts/train_all.sh` |
| Evaluate one model | `python -m src.evaluation.evaluate --model_name resnet50` |
| Evaluate all models | `python -m src.evaluation.evaluate --model_name all` |
| Grad-CAM | `python -m src.evaluation.grad_cam --model_name resnet50` |
| Paper figures | `python -m src.evaluation.visualize` |
| Convert to CoreML | `python -m src.export.convert_coreml --model_name resnet50` |
| Validate CoreML | `python -m src.export.validate_coreml --model_name resnet50` |
| TensorBoard | `tensorboard --logdir runs/` |
| Full pipeline | `./scripts/run_pipeline.sh` |
| Print model sizes | `python -m src.models.classifiers` |

---

## Troubleshooting

**"No .npy files found"** → Run preprocessing first (Step 3).

**"No checkpoint found"** → Run training first (Step 4).

**MPS out of memory** → Reduce `--batch_size` to 16 or 8.

**CoreML validation fails on Linux** → CoreML prediction only works on macOS.

**Xcode model class not generated** → Clean build folder (Cmd+Shift+K), then rebuild.
