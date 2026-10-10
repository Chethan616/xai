# Architecture

The diagrams below are Mermaid (rendered by GitHub). Standalone copies for
slides and reports — SVG and 3× PNG — are in [`diagrams/`](diagrams/), with
their `.mmd` sources.

## 1. End-to-end pipeline

```mermaid
flowchart LR
    subgraph DATA["1 · Data"]
        A1[("XPQRS<br/>17 classes · 1 cycle")]
        A2[("PQD-16c<br/>16 classes · 10 cycles")]
    end
    subgraph PREP["2 · Pre-processing  (src/data.py)"]
        B1["stratified split<br/>train / val / test"]
        B2["z-score with<br/>train mean & std only"]
    end
    subgraph MODEL["3 · Classification"]
        C1["ResNet-CAM<br/>(src/models.py)"]
        C2["physics features → Extra-Trees<br/>(src/features.py)"]
    end
    subgraph XAI["4 · Explanation"]
        D1["Integrated Gradients"]
        D2["Class Activation Map"]
        D3["Tree SHAP"]
    end
    subgraph EVAL["5 · Trust checks"]
        E1["accuracy · macro-F1<br/>confusion matrix"]
        E2["faithfulness<br/>deletion test"]
    end
    F["JSON summary<br/>→ LLM narration"]

    A1 & A2 --> B1 --> B2 --> C1 & C2
    C1 --> D1 & D2
    C2 --> D3
    C1 & C2 --> E1
    D1 & D2 & D3 --> E2
    D1 & D2 --> F
```

| Stage | Code | Output |
|---|---|---|
| Load + split + normalise | `src/data.py` | `X, y, raw, class_names, mean, std` |
| Train signal models | `src/train.py` | `outputs/benchmark/results.jsonl`, checkpoints |
| Train feature models | `src/train_features.py` | same results file |
| Explain one signal | `src/explain.py` | PNG + JSON in `outputs/explanations/` |
| Faithfulness test | `src/faithfulness.py` | `outputs/benchmark/faithfulness.json` |
| Figures for the docs | `src/make_figures.py` | `docs/figures/*.png` |
| Live GUI | `src/stream.py` + `src/gui.py` | interactive Qt window ([GUI.md](GUI.md)) |

## 2. Selected model — ResNet-CAM

A fully-convolutional 1-D ResNet followed by global average pooling (GAP) and a
single linear layer (Wang et al. 2017, *Time-series classification from
scratch*). Because the classifier is one linear layer on top of GAP, the class
score decomposes exactly over time, which gives a **Class Activation Map** for
free (Zhou et al. 2016). 455 k parameters.

```mermaid
flowchart TB
    IN["input signal<br/>1 × L<br/>(L = 100 XPQRS · 640 PQD-16c)"]
    S["Conv1d k7, 64 ch → BatchNorm → GELU"]
    R1["Residual block 64→64, k7"]
    P1["MaxPool ÷2  →  L/2"]
    R2["Residual block 64→128, k5"]
    P2["MaxPool ÷2  →  L/4"]
    R3["Residual block 128→128, k5, dilation 2"]
    R4["Residual block 128→128, k3, dilation 4"]
    FM[["feature map  128 × L/4"]]
    GAP["global average pool over time → 128"]
    DO["Dropout 0.3"]
    FC["Linear 128 → C classes"]
    OUT(["class probabilities"])
    CAM(["CAM_c(t) = Σ_k w_ck · f_k(t)<br/>upsampled to L"])

    IN --> S --> R1 --> P1 --> R2 --> P2 --> R3 --> R4 --> FM --> GAP --> DO --> FC --> OUT
    FM -. "feature maps f_k(t)" .-> CAM
    FC -. "class weights w_ck" .-> CAM
```

**Residual block** (used four times):

```mermaid
flowchart LR
    x["x"] --> c1["Conv1d k, dilation d<br/>BatchNorm · GELU"] --> c2["Conv1d k, dilation d<br/>BatchNorm"] --> add(("+")) --> g["GELU"] --> y["out"]
    x --> sk["1×1 Conv + BatchNorm<br/>(only if channels change)"] --> add
```

Dilations 2 and 4 in the last two blocks widen the receptive field to cover a
full cycle without further down-sampling, so the CAM keeps a resolution of 4
input samples.

## 3. Models compared

| Model | Idea | Params | Built-in explanation |
|---|---|---|---|
| `cnn_res_bilstm` (baseline 1) | residual CNN → 2-layer BiLSTM → additive attention | 1.0 M | attention weights |
| `cnn_complex` (baseline 2) | multi-scale (k 3/5/7) stem → 4 residual blocks → MLP head | 1.42 M | — |
| **`resnet_cam`** | ResNet + GAP + linear (above) | **455 k** | **CAM** |
| `resnet_cam_mv` | ResNet-CAM with two extra input views computed inside the model: sliding 1-cycle RMS and a one-cycle-difference high-pass | 456 k | CAM |
| `inception_cam_mv` | InceptionTime (Fawaz et al. 2020), depth 6, GAP head, multiview | 474 k | CAM |
| `cycle_transformer` | half-cycle patches → 3-layer Transformer, CLS token | 236 k | attention per half-cycle |
| `feat_*` | 48 physics features (`src/features.py`) → logistic regression / random forest / extra-trees / gradient boosting | — | Tree SHAP |

The physics features follow IEEE 1159 vocabulary so every attribution reads
like an engineer's note: per-cycle RMS, minimum/maximum sliding RMS, time below
0.9 / 0.5 / 0.1 p.u. and above 1.1 p.u., crest factor, harmonic ratios
(2nd–11th) and THD, flicker side-band energy, residual-transient energy, etc.

## 4. Training protocol (identical for every signal model)

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 2e-3, weight decay 1e-4 |
| Schedule | One-cycle, 15 % warm-up, stepped per batch |
| Batch | 64, gradient-norm clip 1.0 |
| Loss | cross-entropy |
| Model selection | best validation accuracy (ties → lower validation loss) |
| Epochs / patience | PQD-16c 70 / 25 · XPQRS 40 / 15 |
| Threads | CPU only (4 threads) — no GPU needed |

The test split is touched exactly once, after selection.

## 5. Faithfulness test

```mermaid
flowchart LR
    s["test signal"] --> m1["model → predicted class c,<br/>confidence p"]
    s --> a["attribution for c<br/>(IG / CAM / SHAP)"]
    a --> k["split signal into 20 segments;<br/>rank by summed attribution"]
    k --> t["replace top-k segments<br/>with a clean sine"]
    k --> r["replace k random segments<br/>with a clean sine"]
    t --> d1["drop_top = p − p'"]
    r --> d2["drop_random = p − p''"]
    d1 & d2 --> g["gap = drop_top − drop_random<br/>(> 0 ⇒ explanation points at real evidence)"]
```

Contiguous segments are deleted for **both** arms so that the random baseline
is not penalised by the many sharp edges that scattered single-sample edits
create. For feature models the "segment" is a single feature and the
replacement is the median value of the Normal class.
`tests/test_faithfulness.py` checks the procedure on a toy model with known
evidence.

## 6. Repository layout

```
├── README.md
├── requirements.txt          core dependencies (requirements-lock.txt = original frozen env)
├── src/
│   ├── data.py               dataset loaders + splits
│   ├── models.py             model zoo (MODELS registry)
│   ├── features.py           48 physics features
│   ├── train.py              train/evaluate signal models
│   ├── train_features.py     train/evaluate feature models
│   ├── explain.py            classify + explain one signal (PNG + JSON)
│   ├── faithfulness.py       deletion test for explanations
│   ├── make_figures.py       regenerate docs/figures
│   ├── stream.py             GUI signal source + live classifier (no Qt)
│   ├── gui.py                Qt live disturbance explainer
│   └── legacy/               first-iteration SHAP / attention scripts
├── tests/                    test_faithfulness.py, test_stream.py
├── notebooks/                01–05 original exploration notebooks, archive/
├── dataset/                  XPQRS + PQD-16c CSVs, README
├── models/                   final checkpoints (+ legacy/ first-iteration ones)
├── outputs/benchmark/        results.jsonl, faithfulness.json
└── docs/                     ARCHITECTURE · RESULTS · XAI · GUI, figures/, diagrams/, papers/
```
