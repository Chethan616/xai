Absolutely. Here is a **project context document** written so another LLM can quickly understand the entire project without needing to reconstruct it from our previous conversations.

---

# PQD-16c Power Quality Disturbance Classification + Explainable AI — Project Context

## 1. Project Overview

This project is a **Power Quality Disturbance (PQD) classification system** based on the **PQD-16c dataset**. The goal is to classify a 640-point electrical signal into one of **16 power-quality disturbance classes** using a deep learning model, and then explain the model's prediction using **Explainable AI (XAI)** techniques.

The main deep learning model is a custom **CNN-Res-BiLSTM with an attention mechanism**, called **CNN-RES-BiLSTM V2**. The CNN extracts local signal patterns, residual blocks improve feature extraction, the BiLSTM learns patterns across time, and the attention mechanism gives different importance to different parts of the sequence before classification.

The project therefore has two major parts:

1. **Training / classification system**

   * Load and split the PQD-16c dataset.
   * Train the CNN-RES-BiLSTM model.
   * Select and save the best model based on validation performance.
   * Evaluate it on the unseen test set.

2. **Prediction + XAI system**

   * Load the trained model.
   * Select a test signal.
   * Predict its PQD class.
   * Explain the prediction using:

     * **SHAP**
     * **Attention-based explanation**
   * Visualize which parts of the signal influenced or received attention during the prediction.
   * A lightweight **Llama 3.2 1B** LLM is planned as a future addition to convert the numerical prediction/XAI results into human-readable explanations.

---

# 2. Dataset

The project uses the **PQD-16c dataset** stored as a CSV file.

Each row represents one electrical signal.

### Data format

* First column → class label
* Remaining **640 columns → signal samples**
* Each input signal therefore contains **640 data points**
* Total dataset size → **1600 samples**
* Number of classes → **16**
* Dataset is balanced.

### 16 classes

The classes are:

1. Flicker
2. Flicker+Harmonics
3. Flicker+Sag
4. Flicker+Swell
5. Harmonics
6. Impulsive Transient
7. Interruption
8. Interruption+Harmonics
9. Normal
10. Notch
11. Oscillatory transient
12. Sag
13. Sag+Harmonics
14. Spike
15. Swell
16. Swell+Harmonics

Each class contains 100 samples in the original 1600-sample dataset.

---

# 3. Dataset Splitting

The original dataset is divided into three subsets using **stratified random splitting** with `random_state = 42`.

### Final split

| Dataset    |  Samples |
| ---------- | -------: |
| Training   |     1024 |
| Validation |      256 |
| Testing    |      320 |
| **Total**  | **1600** |

The split maintains the class distribution using `stratify`.

The resulting CSV files are:

* `train.csv`
* `validation.csv`
* `test.csv`

The original format is preserved:

```text
Label | Signal_1 | Signal_2 | ... | Signal_640
```

The test set contains **320 completely unseen samples**, with **20 samples per class**.

---

# 4. Overall Training Pipeline

The training pipeline can be represented as:

```text
PQD-16c Dataset
       ↓
Read CSV
       ↓
Separate Labels and 640-point Signals
       ↓
Stratified Dataset Split
       ↓
 ┌─────────────┬──────────────┬─────────────┐
 │   Training  │ Validation   │    Testing  │
 │    1024     │     256      │     320     │
 └─────────────┴──────────────┴─────────────┘
       ↓
Prepare PyTorch DataLoaders
       ↓
CNN-RES-BiLSTM V2
       ↓
Training
       ↓
Validation after each epoch
       ↓
Choose Best Model
       ↓
Save Best Model
       ↓
Evaluate on Test Set
```

---

# 5. CNN-RES-BiLSTM V2 Architecture

The main classifier is a **1D CNN + Residual CNN + BiLSTM + Attention + Fully Connected classifier**.

The input is:

```text
(1 × 640)
```

meaning one signal channel containing 640 points.

### CNN feature extraction

The first convolution converts:

```text
1 channel → 32 channels
```

using a 1D convolution with kernel size 7.

Then the model contains residual convolution blocks:

```text
32 → 32
32 → 64
64 → 128
128 → 128
```

Three `MaxPool1d(2)` layers reduce the temporal length:

```text
640 → 320 → 160 → 80
```

Therefore, after CNN processing, the representation is approximately:

```text
128 features × 80 time positions
```

The data is then rearranged so that the 80 positions become the sequence processed by the LSTM.

---

# 6. BiLSTM

The CNN output is passed through a **2-layer bidirectional LSTM**.

LSTM input size:

```text
128
```

Hidden size:

```text
128
```

Because it is bidirectional:

```text
128 forward + 128 backward = 256 features
```

So each sequence position becomes a **256-dimensional representation**.

The BiLSTM allows the model to learn relationships between different parts of the electrical signal.

---

# 7. Attention Mechanism

After the BiLSTM, an attention mechanism calculates an importance value for each sequence position.

The attention network is:

```text
256 → 128 → 1
```

The resulting values are passed through **softmax**, producing attention weights.

Conceptually:

```text
BiLSTM features
      ↓
Attention network
      ↓
Attention scores
      ↓
Softmax
      ↓
Attention weights
      ↓
Weighted sum of BiLSTM features
      ↓
256-dimensional representation
```

The attention mechanism is **part of the trained model itself**.

It is not a separate model and no separate attention model needs to be trained.

For every input test sample, the model naturally produces its own attention weights during the forward pass.

The XAI implementation extracts those weights and maps them back to the original **640 signal points**.

For the current attention analysis, the system reports and highlights the **top 20 most attended signal points**.

Important interpretation:

> Attention shows where the model is focusing more strongly, but attention alone does not prove that a point caused the prediction.

---

# 8. Final Classifier

The attention output is passed through:

```text
256 → 128 → 16
```

The final 16 outputs correspond to the 16 PQD classes.

The model produces class scores/logits, which are converted to probabilities using **softmax**.

The class with the highest probability becomes the predicted class.

---

# 9. Training Configuration

Current training configuration:

```text
Batch size       = 64
Maximum epochs   = 150
Learning rate    = 2e-4
Weight decay     = 1e-4
Optimizer        = AdamW
Loss             = CrossEntropyLoss
```

Learning-rate scheduling:

```text
ReduceLROnPlateau
Factor = 0.5
Patience = 8
Minimum LR = 1e-6
```

Other training controls:

```text
Early stopping patience = 25
Gradient clipping       = 1.0
Random seed             = 42
```

The training data is shuffled while validation and test data are not.

---

# 10. Best Model Selection

The model does **not simply save the final epoch**.

After every training epoch, the model is evaluated on the validation dataset.

The best model is selected using:

1. Higher validation accuracy is better.
2. If validation accuracy is equal, lower validation loss is better.

The best model is saved to:

```text
models/cnn_res_bilstm_pq.pth
```

The checkpoint stores information such as:

```text
model_state_dict
number of classes
input length
best validation accuracy
best validation loss
best epoch
architecture name
```

The saved best model is then loaded before final testing.

---

# 11. Experimental Results

The current trained model achieved:

### Validation

```text
Best Validation Accuracy = 90.62%
Best Validation Loss     = 0.2482
```

### Test

```text
Test Accuracy = 85.94%
Test Loss     = 0.2487

Correct = 275 / 320
Wrong   = 45 / 320
```

Therefore, the model correctly classified:

> **275 out of 320 unseen test signals.**

### Overall metrics

```text
Precision = 0.8561
Recall    = 0.8594
F1-score  = 0.8549
```

The macro and weighted averages are almost identical because the test dataset is balanced.

---

# 12. Class-wise Performance

Several classes are classified extremely well.

Examples with F1-score of approximately 1.0 include:

* Flicker+Harmonics
* Flicker+Swell
* Harmonics
* Oscillatory transient
* Swell
* Swell+Harmonics

The more difficult classes are mainly:

```text
Interruption
F1 = 0.1935

Sag
F1 = 0.4898

Sag+Harmonics
F1 = 0.5405

Interruption+Harmonics
F1 = 0.6047
```

The confusion matrix shows that the model has particular difficulty distinguishing some related disturbance types, especially **Interruption/Sag-related classes**.

---

# 13. Prediction Visualization

The evaluation system produces an **Actual vs Predicted Classes** graph.

It compares, for all 320 test samples:

```text
Actual class
     vs
Predicted class
```

The graph helps visually show where the model predictions match the true labels and where they differ.

A **prediction distribution graph** is also generated, showing how many times each of the 16 classes was predicted.

---

# 14. Confusion Matrix

A 16 × 16 confusion matrix is generated.

```text
Rows    → Actual class
Columns → Predicted class
```

The diagonal represents correctly classified samples.

The off-diagonal values show misclassifications.

The current confusion matrix shows strong diagonal performance for many classes, while errors are concentrated in some similar PQD categories.

---

# 15. Prediction + XAI System

After training, the separate XAI notebook loads:

```text
Trained model
       +
Test dataset
       +
Training statistics
```

The user can select:

```text
Random sample
OR
Specific test sample
```

The selected sample is then passed to the trained model.

The system obtains:

```text
Actual class
Predicted class
Class probabilities
Prediction confidence
Correct / Incorrect result
```

Then the same prediction is explained using two XAI approaches:

```text
                    Test Signal
                         ↓
                  Trained Model
                         ↓
                 Predicted Class
                   /           \
                  /             \
               SHAP          Attention
                 ↓                ↓
        Feature influence     Model focus
                 ↓                ↓
              XAI results + visualizations
```

---

# 16. SHAP XAI

SHAP is used to estimate how individual input signal points contribute to the predicted class.

The system:

1. Selects the test signal.
2. Gets the model prediction.
3. Creates a background dataset from up to 32 training samples.
4. Uses `GradientExplainer`.
5. Calculates SHAP values for the selected test signal.
6. Extracts SHAP values corresponding to the predicted class.
7. Ranks signal points by absolute SHAP value.
8. Displays the most influential points.

Current SHAP configuration:

```text
Background samples = up to 32
SHAP samples       = 200
Top influential points = 100
```

SHAP interpretation:

```text
Positive SHAP → supports predicted class
Negative SHAP → opposes predicted class
Large |SHAP|  → stronger influence
```

The visualization plots the original 640-point signal and highlights the highly influential points.

The SHAP explanation is calculated **for each selected test sample**.

There is no separate SHAP model that is stored with the trained classifier.

---

# 17. Attention XAI

The model already contains an attention layer.

For each selected test signal:

```text
Test signal
     ↓
CNN
     ↓
Residual blocks
     ↓
BiLSTM
     ↓
Attention
     ↓
Attention weights
     ↓
Prediction
```

The XAI code extracts the attention weights generated during this process.

Because the CNN reduces the signal length, the attention initially exists at a shorter sequence length.

The XAI implementation therefore:

```text
BiLSTM attention weights
          ↓
Interpolation
          ↓
640-point signal length
          ↓
Compare attention with original signal
```

The system then identifies the **top 20 most attended signal points**.

These points are highlighted on the original signal.

Attention explanation means:

> These are the parts of the signal that received higher weight from the model's attention mechanism.

Again, attention is **not being stored separately for every test sample beforehand**. It is calculated when the selected test sample is passed through the model.

---

# 18. SHAP vs Attention

The two XAI methods provide slightly different information.

### SHAP

Answers approximately:

> **Which signal points contributed toward or against the predicted class?**

It gives positive and negative contributions.

### Attention

Answers approximately:

> **Which parts of the signal received more focus from the model?**

It gives attention weights.

Therefore:

```text
SHAP
→ Contribution / influence

Attention
→ Model focus / importance weighting
```

Using both provides a more complete explanation of the model's decision.

---

# 19. Current XAI Workflow

The current implementation is organized approximately as:

```text
Load trained model
        ↓
Load test dataset
        ↓
Select test sample
        ↓
Run model prediction
        ↓
Get:
  - Actual class
  - Predicted class
  - Confidence
  - Class probabilities
        ↓
   ┌───────────────┐
   │               │
   ↓               ↓
 SHAP          Attention
   ↓               ↓
SHAP values    Attention weights
   ↓               ↓
Top influential Top 20 attended
points          points
   ↓               ↓
   └───────┬───────┘
           ↓
    XAI visualizations
```

---

# 20. Planned LLM Explanation Layer

A future part of the project will add **Llama 3.2 1B**, a very lightweight LLM.

This part is **not implemented yet**.

The idea is to combine:

```text
Training information
        +
Test sample information
        +
Actual class
        +
Predicted class
        +
Prediction confidence
        +
Class probabilities
        +
SHAP results
        +
Attention results
```

and provide these results to the lightweight LLM.

The LLM will then generate a **human-readable explanation** instead of requiring the user to manually interpret numerical SHAP values, attention values, and model outputs.

Conceptually:

```text
                  Trained Model
                       ↓
                  Prediction
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
      SHAP                         Attention
        ↓                             ↓
        └──────────────┬──────────────┘
                       ↓
             Prediction + XAI Data
                       +
                Sample Information
                       +
                Training Information
                       ↓
                Llama 3.2 1B
                       ↓
          Human-readable explanation
```

This LLM component is intended as a **future addition**, not part of the current experimental results.

---

# 21. Overall Project Architecture

The project can be divided into two major diagrams.

## Training System

```text
PQD-16c Dataset
       ↓
Data Loading
       ↓
Train / Validation / Test Split
       ↓
Training Data
       ↓
CNN
       ↓
Residual CNN Blocks
       ↓
BiLSTM
       ↓
Attention
       ↓
Fully Connected Layer
       ↓
16-Class Prediction
       ↓
Validation
       ↓
Best Model Selection
       ↓
Saved Trained Model
```

## Prediction + XAI System

```text
Saved Trained Model
        +
Test Dataset
        ↓
Select Test Sample
        ↓
Model Prediction
        ↓
Actual + Predicted Class
        +
Confidence
        +
Class Probabilities
        ↓
      ┌───────────────┐
      ↓               ↓
    SHAP          Attention
      ↓               ↓
Influential       Top 20
Signal Points     Focused Points
      ↓               ↓
      └───────┬───────┘
              ↓
        XAI Visualization
              ↓
      Future: Llama 3.2 1B
              ↓
    Human-readable explanation
```

---

# 22. Project's Main Contribution

The project is not only trying to classify PQD signals.

The main idea is:

> **Build a PQD classification model and make its predictions easier to understand using XAI.**

The deep learning model provides the classification capability, while SHAP and attention provide different views of why/how the model arrived at its decision.

The planned Llama 3.2 1B layer will eventually make those technical XAI results easier for a human to understand.

---

## One-line project summary

**A CNN-Res-BiLSTM deep learning system for 16-class power quality disturbance classification, enhanced with SHAP and attention-based XAI to identify influential signal regions and provide interpretable predictions, with a lightweight Llama 3.2 1B layer planned for generating human-readable explanations.**
