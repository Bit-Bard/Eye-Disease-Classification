<div align="center">

<!-- Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:2e75b6,100:1F4E79&height=200&section=header&text=Eye%20Disease%20Classification&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Deep%20Learning%20%7C%20Ensemble%20%7C%20Medical%20AI&descAlignY=58&descSize=18" width="100%"/>

<br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-P100%20GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Competition%20Ready-brightgreen?style=for-the-badge)

<br/>

<!-- Accuracy Badge -->
<img src="https://img.shields.io/badge/Test%20Accuracy-97.00%25-1F4E79?style=flat-square&labelColor=0d1117" height="28"/>
&nbsp;
<img src="https://img.shields.io/badge/Macro%20F1--Score-96.99%25-2E75B6?style=flat-square&labelColor=0d1117" height="28"/>
&nbsp;
<img src="https://img.shields.io/badge/Models%20in%20Ensemble-3-6f42c1?style=flat-square&labelColor=0d1117" height="28"/>

</div>

---

## 🖥️ OcuScan AI — Live Interface

<div align="center">
<img src="Screenshot 2026-04-29 150840.png" alt="OcuScan AI Interface" width="90%" style="border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);"/>
<br/><em>OcuScan AI — Production-grade eye disease classification interface powered by a 3-model ensemble</em>
</div>

---

## 📊 Model Performance

<div align="center">

<img src="./screenshots/results1.png" alt="Accuracy and F1 Scores" width="55%"/>
&nbsp;&nbsp;&nbsp;
<img src="./screenshots/results2.png" alt="Classification Report and Confusion Matrix" width="38%"/>

</div>

<div align="center">

| Metric | Score |
|:---|:---:|
| 🎯 Test Accuracy | **97.00%** |
| 📐 Macro F1-Score | **96.99%** |
| ⚖️ Weighted F1-Score | **97.00%** |
| 🔵 Class_0 F1 | 97.48% |
| 🟠 Class_1 F1 | 96.28% |
| 🟢 Class_2 F1 | 97.22% |
| 🔴 Class_3 F1 | 96.97% |

</div>

---

## 🧭 Table of Contents

- [🌟 Project Summary](#-project-summary)
- [🏗️ Architecture Overview](#%EF%B8%8F-architecture-overview)
- [🔬 Dataset](#-dataset)
- [⚙️ Preprocessing Pipeline](#%EF%B8%8F-preprocessing-pipeline)
- [🤖 Model Selection](#-model-selection)
- [🏋️ Training Strategy](#%EF%B8%8F-training-strategy)
- [🧪 Evaluation & Results](#-evaluation--results)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [👥 Authors](#-authors)

---

## 🌟 Project Summary

> **Eye Disease Classification** is a competition-grade deep learning pipeline that classifies **4 categories of ocular diseases** from retinal fundus images using a heterogeneous ensemble of three state-of-the-art architectures.

The pipeline integrates:
- 🧠 **3-Model Ensemble** — EfficientNetB5 + EfficientNetV2-M + ConvNeXt-Small
- 🔬 **CLAHE + Ben Graham** retinal-specific preprocessing
- 🔀 **MixUp Augmentation** for smoother decision boundaries
- ⏱️ **Test-Time Augmentation (TTA × 6)** for robust inference
- ⚡ **Mixed Precision (FP16)** for 2× GPU throughput on Kaggle P100
- 🔁 **2-Fold Stratified Cross-Validation** with checkpoint resume support

---

## 🏗️ Architecture Overview

```
Raw Retinal Image
       │
       ▼
┌─────────────────────────────┐
│  CLAHE + Ben Graham         │  ← Local contrast enhancement (LAB space)
│  Preprocessing              │    + Global lighting gradient removal
└────────────┬────────────────┘
             │
       ┌─────┴──────┐
       │   Resize   │  B5→380px  │  V2M→384px  │  ConvNeXt→224px
       └─────┬──────┘
             │
       ┌─────┴──────────────────┐
       │  Albumentations (×11)  │  ← Train only
       │  + MixUp (α=0.3)       │
       └─────┬──────────────────┘
             │
   ┌─────────┼──────────┐
   ▼         ▼          ▼
EfficientB5  V2-M   ConvNeXt-S
   │         │          │
   └─────────┴──────────┘
             │
    Custom Classifier Head
    GAP → BN → Drop(0.4) → Dense(256,ReLU) → Drop(0.3) → Softmax(4)
             │
     Two-Phase Training
    Phase 1: Frozen (6 ep, lr=1e-3)
    Phase 2: Fine-tune (15 ep, lr=1e-4, Cosine Annealing)
             │
      TTA × 6 passes
             │
      Mean Ensemble (5 checkpoints)
             │
    ┌────────▼────────┐
    │  Predicted Class │
    └─────────────────┘
```

---

## 🔬 Dataset

| Property | Details |
|:---|:---|
| **Source** | [Kaggle — dhruvdevaliya/eye-diseases](https://www.kaggle.com/datasets/dhruvdevaliya/eye-diseases) |
| **Type** | Color retinal fundus photographs |
| **Classes** | 4 disease categories |
| **Total Samples** | ~10,000 images |
| **Test Split** | 10% stratified hold-out (1,000 images) |
| **CV Strategy** | 2-Fold StratifiedKFold |

---

## ⚙️ Preprocessing Pipeline

### 1️⃣ CLAHE + Ben Graham (Retinal-Specific)

```python
def clahe_preprocess(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Ben Graham: remove global lighting gradient
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX=10), -4, 128)
    return img
```

### 2️⃣ Albumentations Augmentation (11 Transforms)

```python
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.4),
    A.GridDistortion(p=0.3),
    A.ElasticTransform(alpha=1, sigma=10, p=0.3),
    A.GaussNoise(var_limit=(5, 30), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.Sharpen(p=0.2),
])
```

### 3️⃣ MixUp Regularization

```python
def mixup(images, labels, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(images))
    mixed_imgs   = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_imgs, mixed_labels
```

---

## 🤖 Model Selection

### Ensemble Members

| Model | Input Size | Pretraining | Folds |
|:---|:---:|:---|:---:|
| **EfficientNetB5** | 380 × 380 | ImageNet-1k | 2 |
| **EfficientNetV2-M** | 384 × 384 | ImageNet-1k | 2 |
| **ConvNeXt-Small** *(TF-Hub)* | 224 × 224 | ImageNet-21k | 1 |

> **Why heterogeneous ensemble?** Each backbone has different inductive biases, receptive field characteristics, and pretraining data — their errors are uncorrelated, so mean-averaging systematically reduces prediction variance.

### Custom Classifier Head (All Models)

```
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dropout(0.4)
        ↓
Dense(256, activation='relu', kernel_regularizer=L2(1e-4))
        ↓
Dropout(0.3)
        ↓
Dense(4, activation='softmax', dtype='float32')
```

---

## 🏋️ Training Strategy

### Two-Phase Protocol

```
┌─────────────────────────────────────────────────────┐
│  PHASE 1 — Frozen Backbone (6 epochs)               │
│  • Optimizer: Adam(lr=1e-3)                         │
│  • Only classifier head trains                      │
│  • EarlyStopping(patience=4)                        │
│  • ReduceLROnPlateau(factor=0.3, patience=2)        │
├─────────────────────────────────────────────────────┤
│  PHASE 2 — Full Fine-Tune (≤15 epochs)              │
│  • Optimizer: Adam(lr=1e-4)                         │
│  • All layers unfrozen (except BatchNorm)           │
│  • Cosine Annealing LR Scheduler                    │
│  • EarlyStopping(patience=6)                        │
│  • ModelCheckpoint(monitor='val_accuracy')          │
└─────────────────────────────────────────────────────┘
```

### Loss Function — Focal Loss

```python
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return loss_fn
```

> Focal Loss down-weights easy examples and focuses training on hard, misclassified samples — critical for class-imbalanced medical datasets.

### Key Hyperparameters

| Parameter | Value |
|:---|:---:|
| Batch Size | 8 |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 1e-4 |
| Focal Loss γ | 2.0 |
| Focal Loss α | 0.25 |
| MixUp α | 0.3 |
| TTA passes | 6 |
| CV Folds | 2 |
| Precision | FP16 (mixed) |

---

## 🧪 Evaluation & Results

### Test-Time Augmentation (TTA × 6)

```python
def tta_predict(model, df_sub, img_size, n_tta=6):
    for path in paths:
        preds = []
        for _ in range(n_tta):
            aug = tta_aug(image=img)['image']
            p   = model(aug[np.newaxis], training=False).numpy()
            preds.append(p[0])
        all_preds.append(np.mean(preds, axis=0))
```

### Mean Ensemble (5 Checkpoints)

```python
# B5_fold1 + B5_fold2 + V2M_fold1 + V2M_fold2 + ConvNeXt_fold1
final_preds   = np.mean(test_preds_all, axis=0)   # shape: (n_test, 4)
final_classes = np.argmax(final_preds, axis=1)
```

### Final Results on Hold-Out Test Set (1,000 images)

```
══════════════════════════════════════════════════════
   Ensemble Test Accuracy      : 0.9700
   Macro   F1-Score            : 0.9699
   Weighted F1-Score           : 0.9700

  Per-class F1:
    Class_0              : 0.9748
    Class_1              : 0.9628
    Class_2              : 0.9722
    Class_3              : 0.9697
══════════════════════════════════════════════════════

Classification Report:
              precision    recall  f1-score   support
     Class_0       0.98      0.97      0.97       258
     Class_1       0.97      0.96      0.96       230
     Class_2       0.97      0.98      0.97       232
     Class_3       0.97      0.97      0.97       280

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000

Confusion Matrix:
[[251   0   4   3]
 [  2 220   2   6]
 [  1   4 227   0]
 [  3   3   2 272]]
```

---

## 📁 Project Structure

```
eye-disease-classification/
├── 📓 eye-disease.ipynb          # Main training notebook
├── 📄 README.md                  # This file
├── 📊 screenshots/
│   ├── ui_screenshot.png         # OcuScan AI interface
│   ├── results1.png              # Accuracy & F1 scores
│   └── results2.png              # Classification report
├── 📈 outputs/
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── f1_per_class.png          # Per-class F1 bar chart
│   ├── training_curves.png       # Loss & accuracy curves
│   └── test_predictions.csv      # Full prediction results
└── 💾 models/
    ├── best_EfficientNetB5_fold1.keras
    ├── best_EfficientNetB5_fold2.keras
    ├── best_EfficientNetV2M_fold1.keras
    ├── best_EfficientNetV2M_fold2.keras
    └── best_ConvNeXt_fold1.keras
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow tensorflow_hub albumentations opencv-python scikit-learn
```

### Training

```bash
# Run all cells in eye-disease.ipynb on Kaggle (P100 GPU recommended)
# Estimated training time: ~5-6.5 hours
# Checkpoint resume: auto-skips already-trained folds
```

### Inference on New Images

```python
import tensorflow as tf
import numpy as np

# Load ensemble models
models = [tf.keras.models.load_model(f'models/best_{name}.keras') for name in model_names]

# Run TTA inference
final_preds = np.mean([tta_predict(m, test_df, img_size) for m in models], axis=0)
predicted_class = CLASS_NAMES[np.argmax(final_preds)]
```

### Estimated Training Time (Kaggle P100)

| Component | Time |
|:---|:---:|
| EfficientNetB5 × 2 folds | ~120–160 min |
| EfficientNetV2-M × 2 folds | ~140–180 min |
| ConvNeXt-Small × 1 fold | ~35–45 min |
| **Total** | **~5–6.5 hours** |

> ⚠️ Kaggle P100 sessions allow 9 hours. Checkpoint resume (Cell 14) protects against timeout.

---

## 🔧 Technical Highlights

<table>
<tr>
<td width="50%">

**🏆 What Makes This Competition-Ready**
- Heterogeneous 3-model ensemble (uncorrelated errors)
- ImageNet-21k pretrained ConvNeXt via TF-Hub
- Retinal-specific CLAHE + Ben Graham preprocessing
- Focal Loss for class imbalance
- MixUp for smoother decision boundaries
- TTA × 6 at inference time
- FP16 mixed precision (2× throughput)

</td>
<td width="50%">

**🛡️ Overfitting Prevention**
- Dropout (0.4 + 0.3 in head)
- L2 weight decay (1e-4)
- Early Stopping (patience=6)
- Data augmentation (11 transforms)
- MixUp label smoothing
- BatchNorm frozen during fine-tune
- Stratified CV for unbiased evaluation

</td>
</tr>
</table>

---

## 🔮 Future Improvements

- [ ] 5-Fold CV instead of 2-Fold for more diverse ensemble members
- [ ] Vision Transformer (ViT / Swin) backbones as additional ensemble members
- [ ] Pseudo-labeling on unlabeled retinal images (semi-supervised)
- [ ] Grad-CAM visualization for clinical interpretability
- [ ] ONNX export for lightweight deployment
- [ ] Larger dataset via domain adaptation from external collections

---

## 📚 References

- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [EfficientNetV2: Smaller, Faster and Better](https://arxiv.org/abs/2104.00298)
- [A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [albumentations: Fast Image Augmentation Library](https://arxiv.org/abs/1809.06839)

---

<div align="center">

## 👥 Authors

<table>
<tr>
<td align="center" width="50%">
<br/>
<b>🧑‍💻 Dhruv Devaliya</b>
<br/>
<sub>Deep Learning Engineer · Dataset Curator · Model Architecture</sub>
<br/><br/>
<a href="https://www.kaggle.com/dhruvdevaliya">
<img src="https://img.shields.io/badge/Kaggle-Dhruv%20Devaliya-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</a>
</td>
<td align="center" width="50%">
<br/>
<b>👩‍💻 Subhulaxmi Panda</b>
<br/>
<sub>ML Engineer · Training Pipeline · Evaluation Framework</sub>
<br/><br/>
<img src="https://img.shields.io/badge/Collaborator-Subhulaxmi%20Panda-2E75B6?style=for-the-badge&logo=github&logoColor=white"/>
</td>
</tr>
</table>

<br/>

---

*Built with ❤️ for the advancement of accessible medical AI*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1F4E79,50:2e75b6,100:1a1a2e&height=100&section=footer" width="100%"/>

</div>
