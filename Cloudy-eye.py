import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.applications import EfficientNetB3
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import AUC, Precision, Recall
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, roc_auc_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QFrame, QGroupBox, QProgressBar, QMessageBox,
                             QTextEdit, QSlider, QTabWidget,
                             QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import cv2
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
EPOCHS_FROZEN = 20       # Phase 1: train head only
EPOCHS_FINETUNE = 15     # Phase 2: fine-tune top layers
TRAIN_DIR = 'cataract_image_dataset/processed_images/train/'
TEST_DIR = 'cataract_image_dataset/processed_images/test/'
MODEL_SAVE_PATH = 'saved_models/best_cataract_model.h5'
RESULTS_DIR = 'results'

os.makedirs('saved_models', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------
# FIX 1: Use EfficientNetB3 Transfer Learning
# A pretrained backbone gives far better feature extraction than a
# custom ConvNet trained from scratch, especially for small medical datasets.
# ------------------------------
def create_transfer_model():
    """
    Build an EfficientNetB3-based binary classifier.
    Phase 1: freeze the backbone, train the head.
    Phase 2 (fine_tune_model): unfreeze top layers and train end-to-end.
    """
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base_model.trainable = False  # Freeze backbone for phase 1

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    # EfficientNet includes its own rescaling; do NOT rescale again in the generator
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    # FIX 2: Remove label_smoothing — it distorts sigmoid probabilities and
    # hurts precision/recall thresholding for binary classification.
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),   # No label smoothing
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )
    return model, base_model


def fine_tune_model(model, base_model, learning_rate=1e-5):
    """
    Phase 2: Unfreeze the top 30 layers of EfficientNetB3 for fine-tuning.
    Use a very low learning rate to avoid destroying pretrained weights.
    """
    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )
    return model


# ------------------------------
# FIX 3: Tuned Data Augmentation
# Reduced rotation and removed channel_shift_range which is too aggressive
# for eye images and can create unrealistic colour artifacts.
# EfficientNet handles its own preprocessing so rescale=1./255 is REMOVED.
# ------------------------------
def get_data_generators():
    """Create data generators with tuned augmentation for EfficientNetB3."""
    # EfficientNetB3 expects pixel values in [0, 255]; its internal
    # preprocessing layer handles normalisation — do NOT rescale here.
    train_datagen = ImageDataGenerator(
        rotation_range=15,          # Reduced from 30; eyes don't appear rotated >15°
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator()   # No augmentation for test

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


# ------------------------------
# FIX 4: Optimal threshold search
# The default 0.5 threshold rarely maximises F1 on imbalanced datasets.
# We find the best threshold on the validation set using Youden's J statistic.
# ------------------------------
def find_optimal_threshold(model, val_generator):
    """
    Search for the probability threshold that maximises F1 score on the
    validation set.  Returns the best threshold (float in [0, 1]).
    """
    val_generator.reset()
    val_preds = model.predict(val_generator, verbose=0).ravel()
    val_labels = val_generator.classes

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (val_preds >= t).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\n✓ Optimal threshold: {best_threshold:.2f}  (val F1 = {best_f1:.4f})")
    return float(best_threshold)


# ------------------------------
# Test-Time Augmentation (unchanged logic, rescale removed)
# ------------------------------
def predict_with_tta(model, image_path, n_augmentations=10, threshold=0.5):
    """Perform test-time augmentation for robust prediction."""
    original_img = Image.open(image_path).convert('RGB')
    original_img = original_img.resize(IMG_SIZE)
    # EfficientNet expects [0, 255]
    original_array = np.array(original_img, dtype=np.float32)

    predictions = []
    pred = model.predict(np.expand_dims(original_array, axis=0), verbose=0)[0][0]
    predictions.append(pred)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True
    )

    img_batch = np.expand_dims(original_array, axis=0)
    for _ in range(n_augmentations - 1):
        augmented_batch = next(datagen.flow(img_batch, batch_size=1))
        pred = model.predict(augmented_batch, verbose=0)[0][0]
        predictions.append(pred)

    return np.mean(predictions), np.std(predictions)


# ------------------------------
# Grad-CAM Implementation
# ------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """Generate Grad-CAM heatmap for model interpretation."""
    if last_conv_layer_name is None:
        from keras.layers import Conv2D
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                last_conv_layer_name = layer.name
                break
            # Handle EfficientNet sub-model
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, Conv2D):
                        last_conv_layer_name = sub_layer.name
                        break
                if last_conv_layer_name:
                    break

    # Build intermediate model targeting the conv layer inside EfficientNet
    try:
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except ValueError:
        # If the layer is nested, retrieve it from the base_model sub-model
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    inner_layer = layer.get_layer(last_conv_layer_name)
                    grad_model = Model(
                        inputs=model.inputs,
                        outputs=[inner_layer.output, model.output]
                    )
                    break
                except ValueError:
                    continue
        else:
            return None

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            np.expand_dims(img_array, axis=0), training=False
        )
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()


def overlay_heatmap(heatmap, image_path, alpha=0.4):
    """Overlay heatmap on original image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_resized, alpha, 0)

    height, width, channel = superimposed_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(superimposed_img.data, width, height, bytes_per_line,
                   QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_img)


# ------------------------------
# Training and Evaluation
# ------------------------------
def validate_dataset():
    """Check if dataset directories exist and contain images."""
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    train_images = sum(len(files) for _, _, files in os.walk(TRAIN_DIR) if files)
    test_images = sum(len(files) for _, _, files in os.walk(TEST_DIR) if files)

    if train_images == 0 or test_images == 0:
        raise ValueError("No images found in train/test directories")

    print(f"✓ Dataset OK: {train_images} training images, {test_images} test images")
    return train_images, test_images


def plot_training_history(history, model_name, phase=''):
    """Plot training curves for a given phase."""
    title_suffix = f" ({phase})" if phase else ""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax, metric, val_metric, title in [
        (axes[0, 0], 'accuracy', 'val_accuracy', 'Accuracy'),
        (axes[0, 1], 'loss',     'val_loss',     'Loss'),
        (axes[1, 0], 'precision','val_precision', 'Precision'),
        (axes[1, 1], 'auc',      'val_auc',      'ROC AUC'),
    ]:
        if metric in history.history:
            ax.plot(history.history[metric], label='Train', marker='o')
            ax.plot(history.history[val_metric], label='Validation', marker='o')
            ax.set_title(f'{model_name} - {title}{title_suffix}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    safe_phase = phase.replace(' ', '_') if phase else 'full'
    plt.savefig(f'{RESULTS_DIR}/{model_name}_{safe_phase}_history.png',
                dpi=100, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix_and_roc(model, test_generator, test_labels, model_name, threshold=0.5):
    """Plot confusion matrix and ROC curve using the optimal threshold."""
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=0)
    # FIX 5: Use the optimised threshold instead of the hard-coded 0.5
    pred_classes = (predictions >= threshold).astype(int)

    cm = confusion_matrix(test_labels, pred_classes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=test_generator.class_indices.keys(),
                yticklabels=test_generator.class_indices.keys())
    ax1.set_title(f'Confusion Matrix - {model_name} (threshold={threshold:.2f})')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    fpr, tpr, _ = roc_curve(test_labels, predictions)
    auc_score = roc_auc_score(test_labels, predictions)
    ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curve - {model_name}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{model_name}_evaluation.png', dpi=100, bbox_inches='tight')
    plt.show()

    return accuracy_score(test_labels, pred_classes), auc_score


def train_and_evaluate():
    """Two-phase training pipeline with optimal threshold selection."""
    print("\n" + "=" * 60)
    print("STARTING ENHANCED MODEL TRAINING (EfficientNetB3)")
    print("=" * 60)

    train_generator, val_generator, test_generator = get_data_generators()
    print("Class mapping:", train_generator.class_indices)
    test_labels = test_generator.classes

    print(f"Training samples:   {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples:       {test_generator.samples}")
    print(f"Class distribution: {dict(zip(train_generator.class_indices.keys(), [np.sum(train_generator.classes == i) for i in range(2)]))}")

    # Compute class weights for imbalanced datasets
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(train_generator.classes),
                                         y=train_generator.classes)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")

    model, base_model = create_transfer_model()
    model.summary()

    # ── Phase 1: Train the classification head ──────────────────────────────
    print("\n" + "─" * 40)
    print("PHASE 1: Training classification head (backbone frozen)")
    print("─" * 40)

    # FIX 6: Use val_auc as the monitor — more discriminative than val_accuracy
    # for imbalanced medical data.  Also removed LearningRateScheduler which
    # conflicted with ReduceLROnPlateau.
    callbacks_phase1 = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', min_lr=1e-7),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', save_best_only=True, mode='max')
    ]

    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_FROZEN,
        validation_data=val_generator,
        callbacks=callbacks_phase1,
        class_weight=class_weight_dict,
        verbose=1
    )

    plot_training_history(history1, "EfficientNetB3", phase="Phase1_FrozenHead")

    # ── Phase 2: Fine-tune top layers of EfficientNet ───────────────────────
    print("\n" + "─" * 40)
    print("PHASE 2: Fine-tuning top 30 EfficientNet layers")
    print("─" * 40)

    model = fine_tune_model(model, base_model, learning_rate=1e-5)

    callbacks_phase2 = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', min_lr=1e-8),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', save_best_only=True, mode='max')
    ]

    history2 = model.fit(
        train_generator,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_generator,
        callbacks=callbacks_phase2,
        class_weight=class_weight_dict,
        verbose=1
    )

    plot_training_history(history2, "EfficientNetB3", phase="Phase2_FineTune")

    # ── Load best model and find optimal threshold ───────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    best_model = load_model(MODEL_SAVE_PATH, compile=False)
    best_model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )

    # FIX 4 (cont): Find optimal decision threshold on validation set
    optimal_threshold = find_optimal_threshold(best_model, val_generator)

    test_accuracy, test_auc = plot_confusion_matrix_and_roc(
        best_model, test_generator, test_labels,
        "EfficientNetB3", threshold=optimal_threshold
    )

    # Classification report with optimal threshold
    test_generator.reset()
    predictions = best_model.predict(test_generator, verbose=0)
    pred_classes = (predictions >= optimal_threshold).astype(int)
    report = classification_report(
        test_labels, pred_classes,
        target_names=list(test_generator.class_indices.keys())
    )

    test_f1 = f1_score(test_labels, pred_classes)
    print(f"\nTest Accuracy:          {test_accuracy * 100:.2f}%")
    print(f"Test AUC:               {test_auc * 100:.2f}%")
    print(f"Test F1 Score:          {test_f1 * 100:.2f}%")
    print(f"Optimal Threshold:      {optimal_threshold:.2f}")
    print("\nClassification Report:\n", report)

    # Save results + optimal threshold for UI to pick up
    with open(f'{RESULTS_DIR}/model_evaluation.txt', 'w') as f:
        f.write("CATARACT DETECTION MODEL - EfficientNetB3 (Transfer Learning)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy:     {test_accuracy * 100:.2f}%\n")
        f.write(f"Test AUC:          {test_auc * 100:.2f}%\n")
        f.write(f"Test F1 Score:     {test_f1 * 100:.2f}%\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    with open(f'{RESULTS_DIR}/optimal_threshold.txt', 'w') as f:
        f.write(str(optimal_threshold))

    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")
    print(f"✓ Results saved to {RESULTS_DIR}/")

    return best_model, optimal_threshold


def load_optimal_threshold():
    """Load the optimal threshold saved during training, or return 0.5."""
    threshold_file = f'{RESULTS_DIR}/optimal_threshold.txt'
    if os.path.exists(threshold_file):
        with open(threshold_file, 'r') as f:
            try:
                return float(f.read().strip())
            except ValueError:
                pass
    return 0.5


# ------------------------------
# Webcam Thread
# ------------------------------
class WebcamThread(QThread):
    frame_processed = pyqtSignal(object, float, str)

    def __init__(self, model, confidence_threshold=50):
        super().__init__()
        self.model = model
        self.running = True
        self.confidence_threshold = confidence_threshold

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize(IMG_SIZE)
                # EfficientNet expects [0, 255]
                img_array = np.array(img_pil, dtype=np.float32)

                prediction = self.model.predict(
                    np.expand_dims(img_array, axis=0), verbose=0
                )[0][0] * 100

                if prediction >= self.confidence_threshold:
                    result = "Cataract Detected"
                    color = (0, 0, 255)
                else:
                    result = "Normal Eye"
                    color = (0, 255, 0)

                cv2.putText(frame, f"{result} ({prediction:.1f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Risk: {prediction:.1f}%", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                self.frame_processed.emit(frame, prediction, result)

            self.msleep(30)

        cap.release()

    def stop(self):
        self.running = False


# ------------------------------
# Analysis Thread
# ------------------------------
class AnalysisThread(QThread):
    analysis_complete = pyqtSignal(float, str, float, float, object)
    analysis_error = pyqtSignal(str)

    def __init__(self, model, image_path, use_tta, confidence_threshold):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.use_tta = use_tta
        self.confidence_threshold = confidence_threshold  # 0–100 scale

    def run(self):
        try:
            threshold_01 = self.confidence_threshold / 100.0

            if self.use_tta:
                risk_mean, risk_std = predict_with_tta(
                    self.model, self.image_path,
                    n_augmentations=10, threshold=threshold_01
                )
                risk_percentage = risk_mean * 100
                uncertainty = risk_std * 100
            else:
                img = Image.open(self.image_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32)  # [0, 255] for EfficientNet
                prediction = self.model.predict(
                    np.expand_dims(img_array, axis=0), verbose=0
                )[0][0]
                risk_percentage = prediction * 100
                uncertainty = 0.0

            result = "Cataract Detected" if risk_percentage >= self.confidence_threshold else "Normal Eye"

            # FIX 7: Correct confidence calculation
            # Confidence = how far the prediction is from the decision boundary (50%)
            confidence = abs(risk_percentage - 50.0) * 2  # maps [50,100] → [0,100]

            # Grad-CAM
            img = Image.open(self.image_path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)

            try:
                heatmap = make_gradcam_heatmap(img_array, self.model)
                heatmap_pixmap = overlay_heatmap(heatmap, self.image_path) if heatmap is not None else None
            except Exception as e:
                print(f"Heatmap generation failed: {e}")
                heatmap_pixmap = None

            self.analysis_complete.emit(risk_percentage, result, confidence, uncertainty, heatmap_pixmap)

        except Exception as e:
            self.analysis_error.emit(str(e))


# ------------------------------
# Enhanced UI
# ------------------------------
class CataractDetectionUI(QMainWindow):
    def __init__(self, model, optimal_threshold=0.5):
        super().__init__()
        self.model = model
        self.current_image_path = None
        self.webcam_thread = None
        self.tta_enabled = True
        # Convert [0,1] threshold to percentage for the slider
        self.confidence_threshold = int(optimal_threshold * 100)
        self.init_ui()
        self.apply_styling()

    def init_ui(self):
        self.setWindowTitle("Advanced Cataract Detection System - Medical AI")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header = QLabel("Advanced Cataract Detection System")
        header.setObjectName("header_label")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        self.single_image_tab = self.create_single_image_tab()
        self.tabs.addTab(self.single_image_tab, "📸 Single Image Analysis")
        self.batch_tab = self.create_batch_tab()
        self.tabs.addTab(self.batch_tab, "📁 Batch Processing")
        self.webcam_tab = self.create_webcam_tab()
        self.tabs.addTab(self.webcam_tab, "🎥 Live Webcam")
        self.info_tab = self.create_info_tab()
        self.tabs.addTab(self.info_tab, "ℹ️ Model Information")

        main_layout.addWidget(self.tabs)
        self.statusBar().showMessage("Ready - System loaded successfully")

    def create_single_image_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_layout = QVBoxLayout(left_panel)

        self.upload_btn = QPushButton("📁 Upload Eye Image")
        self.upload_btn.clicked.connect(self.upload_image)
        left_layout.addWidget(self.upload_btn)

        self.tta_checkbox = QCheckBox("Enable Test-Time Augmentation (More Accurate)")
        self.tta_checkbox.setChecked(True)
        left_layout.addWidget(self.tta_checkbox)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Decision Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(self.confidence_threshold)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_label = QLabel(f"{self.confidence_threshold}%")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        left_layout.addLayout(threshold_layout)

        threshold_note = QLabel("ℹ️ Threshold auto-set to optimal value found during training")
        threshold_note.setStyleSheet("color: #555; font-size: 11px;")
        threshold_note.setWordWrap(True)
        left_layout.addWidget(threshold_note)

        self.image_label = QLabel("No image selected\n\nClick 'Upload Eye Image' to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; padding: 20px;")
        left_layout.addWidget(self.image_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_layout = QVBoxLayout(right_panel)

        risk_group = QGroupBox("📊 Risk Assessment")
        risk_layout = QVBoxLayout()
        self.risk_label = QLabel("Risk: --%")
        self.risk_label.setObjectName("risk_label")
        self.risk_bar = QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_desc = QLabel("Upload an eye image to see risk assessment")
        self.risk_desc.setWordWrap(True)
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_bar)
        risk_layout.addWidget(self.risk_desc)
        risk_group.setLayout(risk_layout)

        diag_group = QGroupBox("🏥 Diagnosis Result")
        diag_layout = QVBoxLayout()
        self.result_label = QLabel("Result: ---")
        self.result_label.setObjectName("result_label")
        self.conf_label = QLabel("Confidence: --%")
        self.uncertainty_label = QLabel("")
        self.uncertainty_label.setWordWrap(True)
        diag_layout.addWidget(self.result_label)
        diag_layout.addWidget(self.conf_label)
        diag_layout.addWidget(self.uncertainty_label)
        diag_group.setLayout(diag_layout)

        heatmap_group = QGroupBox("🔍 Model Attention (Grad-CAM)")
        heatmap_layout = QVBoxLayout()
        self.heatmap_label = QLabel("Upload an image and click 'Analyze' to see what the model focuses on")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumHeight(300)
        self.heatmap_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        heatmap_layout.addWidget(self.heatmap_label)
        heatmap_group.setLayout(heatmap_layout)

        rec_group = QGroupBox("💡 Recommendations")
        rec_layout = QVBoxLayout()
        self.rec_text = QLabel("Please upload an eye image")
        self.rec_text.setWordWrap(True)
        rec_layout.addWidget(self.rec_text)
        rec_group.setLayout(rec_layout)

        self.predict_btn = QPushButton("🔍 Analyze Image")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.analyze_image)

        right_layout.addWidget(risk_group)
        right_layout.addWidget(diag_group)
        right_layout.addWidget(heatmap_group)
        right_layout.addWidget(rec_group)
        right_layout.addWidget(self.predict_btn)

        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 1)

        return widget

    def create_batch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_layout = QHBoxLayout()
        self.batch_upload_btn = QPushButton("📁 Select Folder")
        self.batch_upload_btn.clicked.connect(self.batch_upload)
        self.batch_process_btn = QPushButton("▶️ Process All Images")
        self.batch_process_btn.setEnabled(False)
        self.batch_process_btn.clicked.connect(self.batch_process)
        self.batch_export_btn = QPushButton("💾 Export Results CSV")
        self.batch_export_btn.setEnabled(False)
        self.batch_export_btn.clicked.connect(self.export_results)

        controls_layout.addWidget(self.batch_upload_btn)
        controls_layout.addWidget(self.batch_process_btn)
        controls_layout.addWidget(self.batch_export_btn)
        layout.addLayout(controls_layout)

        self.batch_results_text = QTextEdit()
        self.batch_results_text.setReadOnly(True)
        layout.addWidget(self.batch_results_text)

        self.batch_progress = QProgressBar()
        layout.addWidget(self.batch_progress)

        self.batch_files = []
        self.batch_results = []

        return widget

    def create_webcam_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_layout = QHBoxLayout()
        self.webcam_start_btn = QPushButton("▶️ Start Webcam")
        self.webcam_start_btn.clicked.connect(self.start_webcam)
        self.webcam_stop_btn = QPushButton("⏹️ Stop Webcam")
        self.webcam_stop_btn.setEnabled(False)
        self.webcam_stop_btn.clicked.connect(self.stop_webcam)
        controls_layout.addWidget(self.webcam_start_btn)
        controls_layout.addWidget(self.webcam_stop_btn)
        layout.addLayout(controls_layout)

        self.webcam_label = QLabel("Click 'Start Webcam' to begin live analysis")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setStyleSheet("border: 2px solid #ccc; background-color: black;")
        layout.addWidget(self.webcam_label)

        self.webcam_result_label = QLabel("Status: Not started")
        self.webcam_result_label.setAlignment(Qt.AlignCenter)
        self.webcam_result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.webcam_result_label)

        return widget

    def create_info_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        arch_group = QGroupBox("🏗️ Model Architecture")
        arch_layout = QVBoxLayout()
        arch_text = QTextEdit()
        arch_text.setPlainText(self.get_model_architecture())
        arch_text.setReadOnly(True)
        arch_layout.addWidget(arch_text)
        arch_group.setLayout(arch_layout)

        metrics_group = QGroupBox("📈 Model Performance")
        metrics_layout = QVBoxLayout()
        metrics_text = QTextEdit()
        eval_path = f'{RESULTS_DIR}/model_evaluation.txt'
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                metrics_text.setPlainText(f.read())
        else:
            metrics_text.setPlainText("Train the model first to see performance metrics")
        metrics_text.setReadOnly(True)
        metrics_layout.addWidget(metrics_text)
        metrics_group.setLayout(metrics_layout)

        layout.addWidget(arch_group)
        layout.addWidget(metrics_group)

        return widget

    def get_model_architecture(self):
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)

    def apply_styling(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f2f5; }
            #header_label {
                font-size: 28px;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                margin-bottom: 10px;
            }
            #panel {
                background: white;
                border-radius: 10px;
                margin: 5px;
                padding: 15px;
            }
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #5a67d8; }
            QPushButton:disabled { background-color: #cbd5e0; }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            #risk_label { font-size: 28px; font-weight: bold; }
            #result_label { font-size: 20px; font-weight: bold; }
            QProgressBar { height: 25px; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #667eea; border-radius: 5px; }
            QTabWidget::pane { border: 1px solid #e2e8f0; border-radius: 5px; }
            QTabBar::tab { padding: 10px; font-weight: bold; }
            QTabBar::tab:selected { background-color: #667eea; color: white; }
        """)

    def update_threshold(self, value):
        self.confidence_threshold = value
        self.threshold_label.setText(f"{value}%")

    def upload_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Eye Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if image_path:
            self.current_image_path = image_path
            pixmap = QPixmap(image_path).scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.predict_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(image_path)}")
            self.reset_display()

    def reset_display(self):
        self.risk_label.setText("Risk: --%")
        self.risk_bar.setValue(0)
        self.result_label.setText("Result: ---")
        self.conf_label.setText("Confidence: --%")
        self.uncertainty_label.setText("")
        self.rec_text.setText("Click 'Analyze Image' to get diagnosis")
        self.heatmap_label.setText("Analysis pending...")

    def analyze_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please upload an image first!")
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.predict_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)

        self.analysis_thread = AnalysisThread(
            self.model, self.current_image_path,
            self.tta_checkbox.isChecked(),
            self.confidence_threshold
        )
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.analysis_error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_complete(self, risk_percentage, result, confidence, uncertainty, heatmap_pixmap):
        self.risk_label.setText(f"Risk: {risk_percentage:.1f}%")
        self.risk_bar.setValue(int(risk_percentage))
        self.conf_label.setText(f"Confidence: {confidence:.1f}%")
        self.result_label.setText(f"Result: {result}")

        if uncertainty > 10:
            self.uncertainty_label.setText(f"⚠️ High uncertainty (±{uncertainty:.1f}%) - Consider multiple tests")
        else:
            self.uncertainty_label.setText(f"✓ Low uncertainty (±{uncertainty:.1f}%)")

        color = "#e74c3c" if result == "Cataract Detected" else "#27ae60"
        self.risk_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")
        self.result_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")

        if risk_percentage >= self.confidence_threshold:
            if risk_percentage >= 70:
                advice = "⚠️ HIGH RISK: Please consult an ophthalmologist immediately."
            elif risk_percentage >= 50:
                advice = "⚠️ MODERATE RISK: Schedule an eye examination soon."
            else:
                advice = "⚠️ ELEVATED RISK: Consider a professional eye examination for confirmation."
        else:
            advice = "✅ LOW RISK: Your eyes appear healthy. Continue regular eye exams every 1-2 years."

        self.rec_text.setText(advice)

        if heatmap_pixmap:
            scaled_heatmap = heatmap_pixmap.scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.heatmap_label.setPixmap(scaled_heatmap)
        else:
            self.heatmap_label.setText("Heatmap generation failed")

        self.progress.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.statusBar().showMessage(f"Analysis complete: {result} with {confidence:.1f}% confidence")

    def on_analysis_error(self, error_message):
        self.progress.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to analyze image:\n{error_message}")
        self.statusBar().showMessage("Error occurred during analysis")

    def batch_upload(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.batch_files = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self.batch_files.append(os.path.join(folder, f))

            self.batch_results_text.clear()
            self.batch_results_text.append(f"Found {len(self.batch_files)} images in {folder}\n")
            self.batch_process_btn.setEnabled(True)
            self.batch_export_btn.setEnabled(False)
            self.batch_results = []

    def batch_process(self):
        if not self.batch_files:
            return

        self.batch_process_btn.setEnabled(False)
        self.batch_progress.setMaximum(len(self.batch_files))
        threshold = self.confidence_threshold  # 0–100 scale

        for i, img_path in enumerate(self.batch_files):
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)  # [0, 255] for EfficientNet
            prediction = self.model.predict(
                np.expand_dims(img_array, axis=0), verbose=0
            )[0][0] * 100

            result = "Cataract" if prediction >= threshold else "Normal"
            self.batch_results.append({
                'filename': os.path.basename(img_path),
                'risk_percentage': prediction,
                'result': result
            })

            self.batch_results_text.append(
                f"{i + 1}. {os.path.basename(img_path)}: {result} ({prediction:.1f}%)"
            )
            self.batch_progress.setValue(i + 1)
            QApplication.processEvents()

        self.batch_process_btn.setEnabled(True)
        self.batch_export_btn.setEnabled(True)
        self.batch_results_text.append(
            f"\n✓ Batch processing complete! Processed {len(self.batch_files)} images."
        )
        self.statusBar().showMessage(f"Batch processing complete: {len(self.batch_files)} images analyzed")

    def export_results(self):
        if not self.batch_results:
            return

        df = pd.DataFrame(self.batch_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_DIR}/batch_results_{timestamp}.csv"
        df.to_csv(filename, index=False)

        QMessageBox.information(self, "Export Complete", f"Results exported to:\n{filename}")
        self.statusBar().showMessage(f"Results exported to {filename}")

    def start_webcam(self):
        self.webcam_thread = WebcamThread(self.model, self.confidence_threshold)
        self.webcam_thread.frame_processed.connect(self.update_webcam_frame)
        self.webcam_thread.start()
        self.webcam_start_btn.setEnabled(False)
        self.webcam_stop_btn.setEnabled(True)
        self.statusBar().showMessage("Webcam started")

    def stop_webcam(self):
        if self.webcam_thread:
            self.webcam_thread.stop()
            self.webcam_thread.wait()
            self.webcam_thread = None
        self.webcam_start_btn.setEnabled(True)
        self.webcam_stop_btn.setEnabled(False)
        self.webcam_label.setText("Webcam stopped")
        self.webcam_result_label.setText("Status: Stopped")
        self.statusBar().showMessage("Webcam stopped")

    def update_webcam_frame(self, frame, risk, result):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.webcam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.webcam_label.setPixmap(scaled_pixmap)
        self.webcam_result_label.setText(f"Result: {result} | Risk: {risk:.1f}%")
        color = "#e74c3c" if risk >= self.confidence_threshold else "#27ae60"
        self.webcam_result_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {color}; padding: 10px;"
        )


# ------------------------------
# Main
# ------------------------------
def main():
    print("\n" + "=" * 60)
    print("ADVANCED CATARACT DETECTION SYSTEM (EfficientNetB3)")
    print("=" * 60)

    try:
        train_count, test_count = validate_dataset()
        print(f"Dataset validated: {train_count} train, {test_count} test images")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        print("\nExpected structure:")
        print("  cataract_image_dataset/processed_images/train/cataract/")
        print("  cataract_image_dataset/processed_images/train/normal/")
        print("  cataract_image_dataset/processed_images/test/cataract/")
        print("  cataract_image_dataset/processed_images/test/normal/")
        sys.exit(1)

    optimal_threshold = load_optimal_threshold()

    if os.path.exists(MODEL_SAVE_PATH):
        print("\nLoading existing trained model...")
        best_model = load_model(MODEL_SAVE_PATH, compile=False)
        # FIX 8: Use proper Keras metric objects when recompiling
        best_model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                'accuracy',
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]
        )
        print("✓ Model loaded successfully")
        print(f"✓ Using decision threshold: {optimal_threshold:.2f}")

        retrain = input("\nDo you want to retrain the model? (y/N): ").lower().strip()
        if retrain == 'y':
            best_model, optimal_threshold = train_and_evaluate()
    else:
        print("\nNo existing model found. Training new model...")
        best_model, optimal_threshold = train_and_evaluate()

    print("\n" + "=" * 60)
    print("LAUNCHING USER INTERFACE")
    print("=" * 60)
    print("\nTips:")
    print("- Use 'Single Image Analysis' for detailed diagnosis with heatmaps")
    print("- Use 'Batch Processing' for multiple images")
    print("- Use 'Live Webcam' for real-time screening")
    print("- Enable TTA for more accurate but slower predictions")
    print(f"- Decision threshold auto-loaded: {optimal_threshold:.2f}\n")


    app = QApplication(sys.argv)
    window = CataractDetectionUI(best_model, optimal_threshold=optimal_threshold)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()