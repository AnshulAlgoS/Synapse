# Synapse sEMG Gesture Classification Submission

## Overview
This submission contains a deep learning solution for classifying 5 hand gestures from 8-channel surface electromyography (sEMG) signals. The core model is a compact 3-layer one-dimensional Convolutional Neural Network (1D CNN) trained exclusively on the official Synapse training dataset.

**Key Performance:**
- **Best Validation F1-Score:** ~0.70 (Subject-wise split)
- **Model Architecture:** 3-layer 1D CNN with Batch Normalization and Dropout
- **Preprocessing:** Band-pass filtering (20-450 Hz), Sliding Window (200ms, 50ms stride), Channel-wise Z-score normalization.

## Methods (Short Summary)
Raw 8-channel sEMG signals are first band-pass filtered (20–450 Hz), segmented into overlapping 200 ms windows with 50 ms stride, and normalized per channel using statistics computed from the training set only. Each window is fed to a compact 3-layer 1D CNN that learns temporal features and inter-channel relationships, followed by global average pooling and a linear classifier over the 5 gestures. Training uses only the official Synapse training data, with a strict subject-wise split so that one subject is held out for validation to mimic evaluation on unseen users. The model is optimized with Adam and selected based on the best macro F1-score on the held-out subject, balancing accuracy, robustness, and model complexity in line with the challenge criteria.

## Submission Contents
- Complete Python codebase for data loading, preprocessing, training, evaluation, and inference.
- Trained model artefacts: weights, preprocessing statistics, and configuration.
- Dependency specification in `requirements.txt`.
- This README describing the approach, model design, file structure, and inference usage.

## File Structure
```
submission/
├── configs/
│   └── config.yaml
├── models/
│   ├── emg_gesture_model.pth
│   ├── preprocessing_stats.npz
│   └── metrics.json
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Inference
The `inference.py` script supports running predictions on a single CSV file or the entire dataset structure.

### 1. Run on a Single CSV File
To generate predictions for a specific sEMG recording:

```bash
python src/inference.py \
  --config configs/config.yaml \
  --csv /path/to/gesture_file.csv \
  --out /path/to/output_predictions.npy
```
This will produce a `.npy` file containing the predicted class labels for each window in the recording.

### 2. Run on the Entire Dataset
To process all CSV files in the dataset root:

```bash
python src/inference.py \
  --config configs/config.yaml \
  --all \
  --dataset-root /path/to/Synapse_Dataset \
  --out-dir predictions/
```
This preserves the directory structure (Session/Subject) in the output folder.

## Training (Reproduction)
To retrain the model from scratch using the provided configuration:

1. Update `configs/config.yaml` to point to your dataset location (`raw_root`).
2. Run the training script:

```bash
python src/train.py --config configs/config.yaml
```

The script will:
- Load data from the official Synapse training dataset with a subject-wise split (one subject held out for validation).
- Apply band-pass filtering, sliding-window segmentation, and channel-wise normalization using training-only statistics.
- Train for 50 epochs (default).
- Save the best model (by validation macro F1-score) to `models/emg_gesture_model.pth`.

## Model & Methodology
### Signal Processing
1.  **Band-pass Filter:** 4th order Butterworth filter (20-450 Hz) to remove motion artifacts and high-frequency noise.
2.  **Segmentation:** Sliding window approach (Window: 200ms, Stride: 50ms).
3.  **Normalization:** Channel-wise Z-score normalization using statistics computed from the training subjects (excluding validation subject).

### Architecture
The model is a 1D CNN designed for temporal multi-channel data:
- **Input:** (Batch, 8 Channels, 200 Timepoints)
- **Layer 1:** Conv1D (32 filters, k=7) -> BatchNorm -> ReLU -> MaxPool
- **Layer 2:** Conv1D (64 filters, k=5) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.3)
- **Layer 3:** Conv1D (128 filters, k=3) -> BatchNorm -> ReLU -> GlobalAveragePool
- **Classifier:** Linear (128 -> 5 classes)

### Training Strategy
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (LR=0.001, Weight Decay=1e-4)
- **Validation:** A held-out subject is used as an unseen validation set to estimate cross-subject generalization and to select the best model checkpoint.

## Compliance with Challenge Rules
- Only the official Synapse training dataset is used; no external data or pretraining are introduced.
- Subject-wise splitting ensures that no windows from the validation subject appear in the training set.
- Normalization statistics are computed strictly on the training subset and reused for validation and inference, preventing data leakage.
- The model is intentionally compact to provide a favorable trade-off between accuracy, macro F1-score, and parameter count, as required by the evaluation protocol.