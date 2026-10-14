# Legacy SHAP explainers

> **Legacy code** from the first iteration of the project. For the current
> pipeline (ResNet-CAM, Integrated Gradients, CAM, Tree SHAP) see
> [`src/explain.py`](../explain.py) and [docs/XAI.md](../../docs/XAI.md).

`shap_cnn.py` (and `shap_cnn_bilstm.py` for the CNN-Res-BiLSTM baseline) are
the original SHAP explainers for the legacy checkpoints in `models/legacy/`.
They only read the checkpoints and the PQD-16c CSVs; nothing is modified.

SHAP is used here instead of LIME because the trained model is differentiable
and the input is a 640-point signal. SHAP can assign an attribution to every
signal position and reports which positions support or oppose the predicted
class. LIME would require manually choosing signal segments and can be more
sensitive to that segmentation choice.

## Install

From the repository root:

```text
pip install -r requirements.txt
```

The same `requirements.txt` as the main pipeline; no extra packages are needed.

## Explain a test sample

From the repository root:

```text
python src/legacy/shap_cnn.py --split test --sample 1
```

The sample number is one-based. Validation and training samples can be
explained with `--split validation` or `--split train`.

Useful options:

```text
python src/legacy/shap_cnn.py --split test --sample 1 --background-size 64 --nsamples 400 --top-k 30
```

The output is written to `outputs/shap/`:

- `*_explanation.png`: original waveform with red/blue SHAP attributions and
  a ranked top-impact plot.
- `*_attributions.csv`: one row per signal point, suitable for further
  analysis.
- `*_summary.json`: actual/predicted class, confidence, and the most
  influential positions.
- `*_shap_values.npy`: raw attribution vector for downstream Python use.

Positive SHAP values support the predicted class; negative values push the
model away from that class. The values explain the model's output for the
normalized signal, while the plot displays the original signal amplitude.
