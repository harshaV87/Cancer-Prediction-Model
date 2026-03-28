# Brain Tumor Classification — From MRI to Mobile

An end-to-end deep learning pipeline that classifies brain tumors from MRI scans and deploys the trained model as a native iOS app via CoreML. Built on the **BraTS 2023 GLI** dataset, trained with **PyTorch**, and optimized for real-time inference on **Apple Neural Engine**.

> **Disclaimer:** This project is for research and educational purposes only. It is **not** intended for clinical diagnosis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Model Architectures](#model-architectures)
- [Comparative Analysis](#comparative-analysis)
- [iOS App Architecture](#ios-app-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Usage](#setup--usage)
- [Design Decisions & Best Practices](#design-decisions--best-practices)
- [Limitations](#limitations)
- [License](#license)

---

## Project Overview

| Aspect | Detail |
|---|---|
| **Task** | Binary classification — Tumor Present vs. No Tumor |
| **Input** | 2D axial slices extracted from 3D FLAIR MRI volumes |
| **Dataset** | BraTS 2023 GLI (~1,250 patients, ~26,000+ slices after preprocessing) |
| **Models** | ResNet-50, EfficientNet-B0, MobileNetV3-Small |
| **Training** | Transfer learning from ImageNet, fine-tuned on BraTS |
| **Deployment** | CoreML `.mlpackage` → SwiftUI iOS app with camera + photo picker |
| **Hardware** | Optimized for Apple Silicon (MPS) training and Apple Neural Engine inference |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  BraTS 2023   │    │  Preprocess   │    │   PyTorch Training    │  │
│  │  NIfTI Files  │───▶│  3D → 2D      │───▶│   (MPS / CUDA)       │  │
│  │  (.nii.gz)    │    │  Z-score Norm  │    │                       │  │
│  └──────────────┘    │  Patient Split │    │  ResNet-50            │  │
│                       └──────────────┘    │  EfficientNet-B0      │  │
│                                            │  MobileNetV3-Small    │  │
│                                            └───────────┬───────────┘  │
│                                                        │              │
│  ┌──────────────┐    ┌──────────────┐                  │              │
│  │  Grad-CAM     │    │  Evaluation   │◀───────────────┘              │
│  │  Heatmaps     │    │  AUC, F1, PR  │                              │
│  └──────────────┘    └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   CoreML Conversion    │
                    │   PyTorch → .mlpackage │
                    │   FP16 Quantization    │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      iOS APPLICATION                                │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Camera /     │    │  MLService    │    │  ContentView          │  │
│  │  Photo Picker │───▶│  (Vision +    │───▶│  (SwiftUI)            │  │
│  │              │    │   CoreML)     │    │  Result Card           │  │
│  └──────────────┘    └──────────────┘    └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### 1. Preprocessing (`src/data/preprocess.py`)

- Loads 3D NIfTI FLAIR volumes (`.nii.gz`) and segmentation masks using **nibabel**
- Applies **Z-score normalization** followed by min-max rescaling to `[0, 1]`
- Extracts 2D axial slices, filtering out near-empty boundary slices (`< 1%` brain fraction)
- Labels each slice as **positive** (tumor pixels present in segmentation mask) or **negative**
- Performs **patient-level train/val/test split** (70/15/15) to prevent data leakage — no patient appears in more than one split
- Saves slices as `.npy` files organized into `positive/` and `negative/` directories

### 2. Dataset & Augmentation (`src/data/dataset.py`)

- Custom PyTorch `Dataset` loading `.npy` slices, converting to PIL images, then to tensors
- **Grayscale → 3-channel** repetition to leverage ImageNet-pretrained convolutional filters
- Training augmentations:
  - Random horizontal flip (p=0.5)
  - Random vertical flip (p=0.5)
  - Random rotation (±15°)
  - Random affine translation (±5%) and scale (95–105%)
- Validation/test: resize to 224×224 only (no augmentation)
- Computes **inverse-frequency class weights** for `BCEWithLogitsLoss` to handle class imbalance

### 3. Training (`src/training/train.py`)

- **Transfer learning**: loads ImageNet-pretrained weights, replaces the final classifier head with a single-output neuron for binary classification
- **Loss**: `BCEWithLogitsLoss` with positive class weighting (handles imbalance without oversampling)
- **Optimizers**: Adam (default, lr=1e-4, weight_decay=1e-4) or SGD with momentum
- **Schedulers**: Cosine Annealing (default), ReduceLROnPlateau, or StepLR
- **Early stopping**: monitors `val_auc` with patience of 10 epochs
- **Checkpointing**: saves best model (by val AUC) to `checkpoints/<model_name>/best_model.pth`
- **Logging**: TensorBoard integration — loss, accuracy, AUC, F1, learning rate per epoch
- **Reproducibility**: seeded random, numpy, torch, MPS/CUDA generators

### 4. Evaluation (`src/evaluation/evaluate.py`)

Comprehensive metrics computed on the held-out test set:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Of predicted tumors, how many are real |
| **Recall (Sensitivity)** | Of real tumors, how many are detected — **critical in medical AI** |
| Specificity | True negative rate |
| F1 Score | Harmonic mean of precision & recall |
| AUC-ROC | Discrimination ability across all thresholds |
| AUC-PR | Precision-recall tradeoff (robust under class imbalance) |
| Inference Time | Measured over 100 runs with device synchronization |

Also generates:
- ROC and Precision-Recall curve data (saved as JSON)
- Confusion matrix (TP, FP, TN, FN)
- Multi-model comparison table (`results/comparison_table.md`)

### 5. Interpretability (`src/evaluation/grad_cam.py`)

- **Grad-CAM** (Gradient-weighted Class Activation Mapping) produces heatmaps showing which MRI regions the model attends to
- Hooks into the last convolutional layer of each architecture
- Generates side-by-side grids: Original → Heatmap → Overlay
- Balanced sampling: equal positive and negative examples

### 6. Visualization (`src/evaluation/visualize.py`)

Generates publication-ready figures:
- ROC curves (per-model, overlaid)
- Precision-Recall curves
- Confusion matrices
- Training history (loss & AUC curves)

### 7. CoreML Export (`src/export/convert_coreml.py`)

- Traces the PyTorch model with `torch.jit.trace`
- Converts via `coremltools` to `.mlpackage` (ML Program format)
- Configures `ImageType` input with RGB color layout and scale
- **FP16 quantization** by default — halves model size with negligible accuracy loss
- Sets classifier labels: `["No Tumor", "Tumor Detected"]`
- Targets iOS 16+ deployment

### 8. CoreML Validation (`src/export/validate_coreml.py`)

- Compares PyTorch vs CoreML predictions on 100 random test samples
- Measures max/mean absolute probability difference, Pearson correlation, and binary agreement
- Pass threshold: max absolute difference < 0.05

---

## Model Architectures

### ResNet-50 — Accuracy Baseline

```
Input (3×224×224) → [Conv7×7, BN, ReLU, MaxPool]
  → Layer1 (3 Bottleneck blocks, 256ch)
  → Layer2 (4 Bottleneck blocks, 512ch)
  → Layer3 (6 Bottleneck blocks, 1024ch)
  → Layer4 (3 Bottleneck blocks, 2048ch)
  → AdaptiveAvgPool → FC(2048 → 1) → Sigmoid
```

- **23.5M parameters**
- Deep residual connections prevent vanishing gradients
- Highest capacity — expected best raw accuracy
- Skip connections enable training of very deep networks

### EfficientNet-B0 — Balanced

```
Input (3×224×224) → [Stem Conv3×3]
  → MBConv blocks (compound-scaled depth/width/resolution)
  → Dropout(0.2) → FC(1280 → 1) → Sigmoid
```

- **5.3M parameters**
- Compound scaling balances network depth, width, and input resolution
- Mobile Inverted Bottleneck (MBConv) blocks with squeeze-and-excitation
- Strong accuracy with ~4.4× fewer parameters than ResNet-50

### MobileNetV3-Small — Edge Deployment

```
Input (3×224×224) → [Conv3×3, BN, H-Swish]
  → Inverted Residual blocks with SE attention
  → FC(576 → 1024) → H-Swish → FC(1024 → 1) → Sigmoid
```

- **2.5M parameters**
- Hardware-aware NAS (Neural Architecture Search) optimized for mobile
- H-Swish activation, squeeze-and-excitation attention
- Smallest footprint, fastest inference on Apple Neural Engine

---

## Comparative Analysis

### Model Comparison

| Property | ResNet-50 | EfficientNet-B0 | MobileNetV3-Small |
|---|---|---|---|
| Parameters | 23.5M | 5.3M | 2.5M |
| FP32 Size (est.) | ~90 MB | ~20 MB | ~10 MB |
| FP16 CoreML Size | ~45 MB | ~10 MB | ~5 MB |
| ImageNet Top-1 | 76.1% | 77.1% | 67.7% |
| Architecture Era | 2015 (Microsoft) | 2019 (Google) | 2019 (Google) |
| Inference Speed | Slowest | Moderate | Fastest |
| Best For | Maximum accuracy | Accuracy-efficiency balance | Real-time mobile |

### Strengths & Weaknesses

| | Pros | Cons |
|---|---|---|
| **ResNet-50** | Highest capacity; well-studied; robust feature extraction | Large model size; slower inference; overkill for binary task |
| **EfficientNet-B0** | Best accuracy/size ratio; compound scaling is principled | Slightly more complex architecture; slower than MobileNet |
| **MobileNetV3-Small** | Tiny model; real-time on-device; optimized for Neural Engine | Lower baseline accuracy; may miss subtle features |

### Why Three Models?

Training all three enables a data-driven deployment decision:
1. **If accuracy is paramount** → pick the model with highest AUC/sensitivity
2. **If the iOS app needs real-time camera feed** → MobileNetV3 wins on latency
3. **For publication** → the comparison table strengthens the paper with empirical tradeoff analysis

---

## iOS App Architecture

The iOS app follows **MVVM (Model-View-ViewModel)** with a clean separation of concerns:

```
ios/CancerPredictor/CancerPredictor/
├── CancerPredictorApp.swift          # App entry point (@main)
├── Services/
│   └── MLService.swift               # CoreML inference (singleton)
├── ViewModels/
│   └── PredictionViewModel.swift     # UI state + business logic
└── Views/
    ├── ContentView.swift             # Main screen (image picker, results)
    ├── CameraView.swift              # UIKit camera wrapped for SwiftUI
    └── InfoView.swift                # About/information screen
```

### Key Design Patterns

- **Singleton `MLService`** — loads the CoreML model once, reused across predictions
- **Vision framework** — `VNCoreMLRequest` handles image preprocessing (resize, crop, pixel buffer conversion) automatically
- **Async/await** — inference runs off the main thread via `withCheckedThrowingContinuation`
- **`@Published` properties** — drive reactive SwiftUI updates on prediction completion
- **`PhotosPicker` + `UIImagePickerController`** — supports both photo library and live camera input
- **Neural Engine optimization** — `computeUnits = .all` lets iOS choose the best hardware (ANE > GPU > CPU)

### Inference Flow

```
User selects image (Camera / Photo Library)
  → PredictionViewModel receives UIImage
  → MLService.predict() called
  → VNCoreMLRequest processes image (auto-resizes to 224×224)
  → CoreML model runs on Neural Engine
  → VNClassificationObservation returned
  → PredictionResult displayed in ResultCard
     (Label, Confidence %, Inference time in ms)
```

---

## Project Structure

```
Cancer-Prediction-Model/
├── README.md                      # This file
├── RUNNING_INSTRUCTIONS.md        # Step-by-step setup guide
├── QA.md                          # Q&A documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Excludes data, checkpoints, venv
│
├── data/
│   ├── raw/BraTS2023/             # Raw NIfTI volumes (not committed)
│   └── processed/                 # Preprocessed .npy slices (not committed)
│       ├── train/{positive,negative}/
│       ├── val/{positive,negative}/
│       ├── test/{positive,negative}/
│       └── split_info.txt
│
├── src/                           # Python ML pipeline
│   ├── data/
│   │   ├── preprocess.py          # NIfTI → 2D slices + patient-level split
│   │   └── dataset.py             # PyTorch Dataset + augmentation
│   ├── models/
│   │   └── classifiers.py         # ResNet-50, EfficientNet-B0, MobileNetV3
│   ├── training/
│   │   ├── config.py              # TrainConfig dataclass (all hyperparameters)
│   │   └── train.py               # Training loop, early stopping, TensorBoard
│   ├── evaluation/
│   │   ├── evaluate.py            # Metrics, confusion matrix, comparison table
│   │   ├── grad_cam.py            # Grad-CAM interpretability heatmaps
│   │   └── visualize.py           # ROC, PR curves, paper figures
│   └── export/
│       ├── convert_coreml.py      # PyTorch → CoreML .mlpackage
│       └── validate_coreml.py     # CoreML vs PyTorch consistency check
│
├── ios/                           # SwiftUI iOS application
│   └── CancerPredictor/
│       └── CancerPredictor/
│           ├── CancerPredictorApp.swift
│           ├── Services/MLService.swift
│           ├── ViewModels/PredictionViewModel.swift
│           └── Views/{ContentView,CameraView,InfoView}.swift
│
├── scripts/
│   ├── train_all.sh               # Train + evaluate + export all models
│   └── run_pipeline.sh            # Full end-to-end pipeline
│
├── checkpoints/                   # Saved model weights (not committed)
├── results/                       # Metrics, figures, comparison tables (not committed)
├── exported_models/               # CoreML .mlpackage files (not committed)
└── runs/                          # TensorBoard logs (not committed)
```

---

## Tech Stack

### Python / ML

| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 2.1.0 | Model training & inference |
| torchvision | ≥ 0.16.0 | Pretrained models & transforms |
| nibabel | ≥ 5.2.0 | NIfTI medical image loading |
| scikit-learn | ≥ 1.3.0 | Metrics, train/test split |
| coremltools | ≥ 7.0 | PyTorch → CoreML conversion |
| matplotlib / seaborn | ≥ 3.7 / ≥ 0.13 | Visualization & paper figures |
| TensorBoard | ≥ 2.15.0 | Training monitoring |
| OpenCV | ≥ 4.8.0 | Grad-CAM heatmap overlay |

### iOS

| Technology | Purpose |
|---|---|
| SwiftUI | Declarative UI framework |
| CoreML | On-device ML inference |
| Vision | Image preprocessing for CoreML |
| PhotosUI | Photo library picker |
| AVFoundation | Camera capture (via UIKit bridge) |

---

## Setup & Usage

See [RUNNING_INSTRUCTIONS.md](RUNNING_INSTRUCTIONS.md) for the full step-by-step guide.

**Quick start:**
```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Preprocess (requires BraTS 2023 dataset)
python -m src.data.preprocess --data_dir data/raw/BraTS2023 --output_dir data/processed

# Train
python -m src.training.train --model_name resnet50 --num_epochs 50

# Evaluate
python -m src.evaluation.evaluate --model_name resnet50 --device cpu

# Export to CoreML
python -m src.export.convert_coreml --model_name resnet50 --quantize fp16

# Or run the full pipeline at once
./scripts/run_pipeline.sh
```

---

## Design Decisions & Best Practices

| Decision | Rationale |
|---|---|
| **Patient-level split** | Prevents data leakage — slices from the same patient never appear in both train and test |
| **BCEWithLogitsLoss** | Numerically stable binary classification (sigmoid fused into loss); avoids separate sigmoid layer |
| **Inverse-frequency class weighting** | Handles class imbalance without oversampling, which could distort augmentation statistics |
| **Transfer learning from ImageNet** | Pretrained features (edges, textures) transfer well to medical imaging, especially with limited data |
| **Grayscale → 3-channel** | Required to utilize ImageNet pretrained conv1 filters; empirically works better than retraining from scratch |
| **Cosine annealing LR** | Smooth decay avoids the sharp drops of StepLR; proven effective for fine-tuning |
| **Early stopping on AUC** | AUC is threshold-independent and more robust than accuracy for imbalanced medical datasets |
| **FP16 CoreML quantization** | Halves model size with < 0.05 probability difference; enables faster Neural Engine execution |
| **Grad-CAM interpretability** | Essential for medical AI — clinicians need to see *why* the model predicts tumor |
| **MVVM in iOS** | Clean separation of ML inference (Service), state management (ViewModel), and UI (Views) |
| **VNCoreMLRequest** | Handles image resizing/normalization automatically; more robust than manual preprocessing |
| **Seed everything** | Reproducible results across runs (random, numpy, torch, MPS seeds all set) |

---

## Limitations

- **2D slice-level classification** — does not leverage full 3D volumetric context; a 3D CNN or transformer could capture inter-slice relationships
- **Binary task only** — does not distinguish tumor subtypes (HGG vs LGG) or segment tumor boundaries
- **Single modality (FLAIR)** — uses only T2-FLAIR; multi-modal fusion (T1, T1ce, T2, FLAIR) could improve performance
- **Not clinically validated** — no regulatory approval; not suitable for real diagnostic use
- **iOS only** — CoreML limits deployment to Apple devices; ONNX export would enable cross-platform deployment
- **No uncertainty estimation** — provides point predictions without confidence calibration or out-of-distribution detection

---

## License

This project is for **research and educational purposes only**. The BraTS 2023 dataset is subject to its own [Synapse license terms](https://www.synapse.org/Synapse:syn51156910/wiki/622351).
