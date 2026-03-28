# Q&A: Brain Tumor Classification Project

A summary of design decisions and technical questions addressed during project planning.

---

## 1. What is the difference between classification and segmentation?

| | Classification | Segmentation |
|---|---|---|
| **Output** | Single label (tumor yes/no) | Pixel-level mask |
| **Architecture** | ResNet, EfficientNet, MobileNet | U-Net, ResU-Net, nnU-Net |
| **Key Metric** | AUC-ROC, Accuracy, F1 | Dice Score, Hausdorff Distance |
| **Complexity** | Simpler to train and deploy | Much more complex |
| **Training Data** | Just needs image + label | Needs expert-drawn pixel masks |
| **Clinical Value** | Screening / triage | Treatment planning, surgery guidance |
| **CoreML Conversion** | Straightforward | Harder (custom ops, larger model) |
| **Model Size** | ~50 MB (ResNet-50 FP16) | ~200–500 MB (3D U-Net) |
| **iPhone Inference** | ~15–30 ms | ~100–500 ms (if feasible at all) |
| **Paper Difficulty** | Moderate | High |

**Our choice:** Classification — simpler, faster to build and publish. Segmentation can be a follow-up paper.

---

## 2. Can we train ResNet-50 and convert it to .mlpackage?

Yes. ResNet-50 uses only standard operators (`Conv2d`, `BatchNorm`, `ReLU`, `AdaptiveAvgPool`, `Linear`) that `coremltools` handles natively. No custom layers needed.

| | ResNet-50 | MobileNetV3-Small |
|---|---|---|
| Parameters | ~25.6M | ~2.5M |
| `.mlpackage` size (FP16) | ~50 MB | ~5 MB |
| Accuracy | Higher (likely) | Slightly lower |
| iPhone inference | ~15–30 ms (ANE) | ~5–10 ms (ANE) |

**Decision:** Train and convert **both**. Use ResNet-50 as the primary model in the iOS app, and include MobileNetV3 numbers in the paper as a lightweight alternative.

---

## 3. Why and where is FastAPI required?

The Gemini PDF proposed a server-side inference backend:

```
iPhone App  ──uploads .nii.gz──▶  FastAPI Server (GPU)  ──returns mask──▶  iPhone App
```

**Why we DON'T need it for this project:**

| Concern | FastAPI (Server) | Our Plan (On-Device CoreML) |
|---|---|---|
| Model size | Unlimited | ~50 MB (ResNet-50 FP16) — fine |
| Input | 3D NIfTI volumes | Single 2D image (224×224) — fine |
| Compute | Multi-stage cascade | Single forward pass — fine |
| Inference time | ~1–5s + network latency | ~15–30 ms locally |
| Privacy | Image sent over network | Image never leaves device ✓ |
| Offline use | Requires internet | Works offline ✓ |
| Infrastructure cost | Server + GPU hosting $$ | Free (runs on iPhone) |

**When you WOULD need FastAPI:**
- Full 3D segmentation too heavy for iPhone
- Storing/aggregating results across users
- Model versioning without App Store updates
- Multi-platform service (Android, web, iOS)

---

## 4. What journals can this be published in?

Ranked by impact/prestige with realistic fit assessment:

| Rank | Journal | Impact Factor | Fit |
|---|---|---|---|
| 1 | IEEE Transactions on Medical Imaging (TMI) | ~10.6 | ⚠️ Stretch — demands architectural novelty |
| 2 | Medical Image Analysis (MedIA) | ~10.9 | ⚠️ Stretch — same tier as TMI |
| 3 | **Computers in Biology and Medicine** | ~7.7 | ✅ Good — accepts applied ML + deployment |
| 4 | Computerized Medical Imaging and Graphics | ~5.7 | ✅ Good — medical imaging applications |
| 5 | Artificial Intelligence in Medicine | ~7.5 | ✅ Good — values clinical applicability |
| 6 | **IEEE Access** | ~3.9 | ✅ Very good — fast review (~2–3 months) |
| 7 | **MDPI Diagnostics** | ~3.6 | ✅ Best fit — fast review, receptive to DL papers |
| 8 | BMC Medical Imaging | ~2.5 | ✅ Best fit — good for first paper |
| 9 | MDPI Applied Sciences | ~2.7 | ✅ Easy fit — broad scope |
| 10 | Journal of Imaging (MDPI) | ~3.2 | ✅ Easy fit — mobile deployment angle fits |

**Recommended targets:**
- **Primary:** Computers in Biology and Medicine — highest impact with real acceptance chance
- **Backup:** MDPI Diagnostics — fast review (4–6 weeks), high acceptance for well-executed work
- **Speed:** IEEE Access — fast, indexed everywhere, no page limits

**Conference alternatives** (faster publication, 3–4 months):
- MICCAI Workshop, IEEE EMBC, SPIE Medical Imaging

**To target Tier 1 journals later**, add: segmentation, novel architecture, multi-institutional validation, or clinical reader study.

---

## 5. Project Scope Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Task | Classification (not segmentation) | Simpler, publishable, leverages iOS expertise |
| Dataset | BraTS 2023 GLI (MRI) | Gold standard, publicly available, ~1,250 cases |
| Modality | FLAIR (primary) | Best tumor visibility for classification |
| Framework | PyTorch | Best ecosystem, MPS support for M4 Mac |
| iOS Scope | Minimal PoC | Photo picker → CoreML → result screen |
| Backend | None (no FastAPI) | On-device inference sufficient for classification |
| Data Split | Patient-level 70/15/15 | Prevents data leakage — mandatory for medical ML |
| Training Hardware | Apple M4 Mac (MPS) | PyTorch MPS backend, feasible in hours |
| Quantization | Float16 | ~50% size reduction, negligible accuracy loss |
