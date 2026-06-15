# 👁️ Cloudy Eye — Cataract Detection System

A deep learning application for automated cataract detection from eye images, built with **EfficientNetB3** and a full-featured **PyQt5** graphical interface. The model achieves **96.69% accuracy** and **99.32% AUC** on the test set.

---

## ✨ Features

- **Two-phase transfer learning** — EfficientNetB3 backbone with frozen-head pretraining followed by fine-tuning of the top 30 layers
- **Grad-CAM heatmaps** — visual explanation of which regions of the eye influenced the prediction
- **Test-Time Augmentation (TTA)** — averages predictions over 10 augmented variants for more robust results
- **Optimal threshold search** — uses Youden's J statistic on the validation set to find the best decision boundary instead of a fixed 0.5
- **Single image analysis** — upload one image and get a detailed diagnosis with confidence score and Grad-CAM overlay
- **Batch processing** — analyze an entire folder of images and export results to CSV
- **Live webcam screening** — real-time detection through any connected camera
- **Class imbalance handling** — automatic class weight computation during training

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | 96.69% |
| Test AUC | 99.32% |
| Test F1 Score | 96.72% |
| Optimal Threshold | 0.38 |
| Precision (Normal) | 0.97 |
| Recall (Normal) | 0.97 |
| Precision (Cataract) | 0.97 |
| Recall (Cataract) | 0.97 |

---

## 🗂️ Dataset Structure

The model expects the dataset in this exact directory layout:

```
cataract_image_dataset/
└── processed_images/
    ├── train/
    │   ├── cataract/
    │   └── normal/
    └── test/
        ├── cataract/
        └── normal/
```

The dataset used for training is publicly available on [Kaggle](https://www.kaggle.com/datasets) — search for **Cataract Dataset**. It contains labelled eye fundus images in two classes: `cataract` and `normal`.

---

## ⚙️ Installation

### Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended) or CPU

### Clone and install

```bash
git clone https://github.com/your-username/cloudy-eye.git
cd cloudy-eye
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.12
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
opencv-python
PyQt5
```

---

## 🚀 Usage

### First run (train + launch UI)

If no saved model is found, training starts automatically:

```bash
python Cloudy-eye.py
```

The training pipeline runs in two phases:

1. **Phase 1** — backbone frozen, only the classification head is trained (`LR = 1e-3`, up to 20 epochs)
2. **Phase 2** — top 30 EfficientNetB3 layers unfrozen, full fine-tuning (`LR = 1e-5`, up to 15 epochs)

After training, the optimal decision threshold is computed on the validation set and saved automatically.

### Subsequent runs

On startup, the app detects an existing saved model and asks whether to retrain:

```
Do you want to retrain the model? (y/N):
```

Press Enter (or type `N`) to skip training and go straight to the UI.

---

## 🖥️ Interface Tabs

| Tab | Description |
|---|---|
| **Single Image** | Upload one eye image, run TTA prediction, view Grad-CAM heatmap |
| **Batch Processing** | Select a folder, process all images, export CSV report |
| **Live Webcam** | Real-time cataract risk overlay on webcam feed |
| **Model Information** | Architecture summary, training configuration, performance metrics |

---

## 📁 Output Files

After training, the following files are created automatically:

```
saved_models/
└── best_cataract_model.h5        # Best checkpoint (monitored by val_auc)

results/
├── EfficientNetB3_Phase1_FrozenHead_history.png
├── EfficientNetB3_Phase2_FineTune_history.png
├── EfficientNetB3_evaluation.png  # Confusion matrix + ROC curve
└── batch_results_<timestamp>.csv  # Batch export (when used)
```

---

## 🏗️ Model Architecture

```
Input (224×224×3)
    │
EfficientNetB3 (ImageNet weights)
    │  └── Top 30 layers unfrozen in Phase 2
GlobalAveragePooling2D
BatchNormalization
Dense(256, relu) → Dropout(0.4)
Dense(64,  relu) → Dropout(0.3)
Dense(1, sigmoid)
```

**Key design decisions:**

- EfficientNetB3's internal rescaling layer handles pixel normalization — input images must remain in `[0, 255]` (no `rescale=1./255` in the generator)
- No label smoothing — it distorts sigmoid probabilities and hurts thresholding for binary classification
- `val_auc` used as the monitor for early stopping and checkpointing (more reliable than `val_accuracy` on imbalanced medical data)
- `LearningRateScheduler` removed to avoid conflict with `ReduceLROnPlateau`

---

## 📌 Notes

- The system is intended as a **screening aid**, not a replacement for professional ophthalmological diagnosis.
- Grad-CAM requires locating the last `Conv2D` layer inside the EfficientNet sub-model; the implementation handles nested layer traversal automatically.
- TTA adds ~10× inference time per image; it can be toggled off in the UI for faster results.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
