# Explainable Deep Learning Framework for Intelligent Power Quality Disturbance Diagnosis and Decision Support

## Dataset

**PQD-16c Data:**  
https://www.kaggle.com/datasets/jkexin/pqd-16c-data

The dataset contains 16 power quality disturbance classes, with each sample containing 640 signal values.

## Completed

- Prepared the PQD-16c dataset.
- Created a reproducible train/validation/test split:
  - Train: 1024 samples
  - Validation: 256 samples
  - Test: 320 samples
- Maintained equal class distribution across all splits.
- Implemented training-data-only normalization.
- Implemented a 1D Residual CNN for classification.
- Added signal augmentation during training.
- Added learning-rate scheduling and early stopping.
- Trained and evaluated the model.
- Achieved:
  - Validation Accuracy: **86.72%**
  - Test Accuracy: **82.81%**
- Saved the trained model to:
  `models/cnn_pq_best.pth`
- Individual test-sample prediction interface.
- Actual vs. predicted output display.
- Test signal visualization.
- Confusion matrix.
- Per-class Precision, Recall and F1-score.

## Pending


- Explainable AI (XAI) analysis - SHAP.
- Llamma 3.2 1B light-weight LLM Integration
- Final results and analysis.

## Model

The current model is a lightweight **1D Residual CNN** designed for power-quality signal classification.

**Input:** `640` signal samples  
**Output:** `16` disturbance classes  
**Trainable parameters:** approximately `226K`

## Project Status

**CNN classification pipeline: Completed**

**XAI and detailed evaluation: In Progress**