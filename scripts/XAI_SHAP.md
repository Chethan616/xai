# SHAP explainability

`shap_explain.py` is a standalone companion to `final_scripts.ipynb`. It does
not modify the notebook, the training code, or the saved checkpoint.

SHAP is used here instead of LIME because the trained model is differentiable
and the input is a 640-point signal. SHAP can assign an attribution to every
signal position and reports which positions support or oppose the predicted
class. LIME would require manually choosing signal segments and can be more
sensitive to that segmentation choice.

## Install

From the repository root:

```text
pip install -r requirements-xai.txt
```

This XAI requirements file is intentionally standalone. The repository's
original exported requirements pin `numpy==2.5.2`, which cannot be installed
on Python 3.11.

## Explain a test sample

From the repository root:

```text
python scripts/shap_explain.py --split test --sample 1
```

The sample number is one-based. Validation and training samples can be
explained with `--split validation` or `--split train`.

Useful options:

```text
python scripts/shap_explain.py --split test --sample 1 --background-size 64 --nsamples 400 --top-k 30
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
