"""Generate SHAP explanations for the trained PQD-16c 1D CNN.

This is intentionally a standalone companion to ``final_scripts.ipynb``.
It recreates the saved model architecture, loads the existing checkpoint,
uses the checkpoint's training normalization values, and writes explanation
artifacts without changing the training or evaluation code.

Example:
    python src/legacy/shap_cnn.py --split test --sample 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "legacy" / "cnn_pq_best.pth"
DEFAULT_DATA_DIR = ROOT / "dataset" / "PQD-16c-Data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "shap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain one PQD-16c CNN prediction with SHAP."
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split containing the sample (default: test).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="One-based sample number within the selected split (default: 1).",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=32,
        help="Number of training examples used as the SHAP background (default: 32).",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help="SHAP integration samples (larger is slower but more stable).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of highest-impact signal positions to include in the summary.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to cnn_pq_best.pth.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.csv, validation.csv, and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the explanation artifacts are written.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to select the SHAP background examples.",
    )
    return parser.parse_args()


def import_runtime() -> tuple[Any, Any, Any, Any]:
    """Import optional runtime dependencies with a useful install message."""

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import shap
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(
            "Missing XAI dependency. Install it in the same environment as the "
            "project with: pip install -r requirements-xai.txt"
        ) from exc

    return torch, nn, pd, (plt, shap)


def build_model(nn: Any, num_classes: int) -> Any:
    """Recreate the architecture used to produce cnn_pq_best.pth."""

    class ResidualBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            dropout: float = 0.05,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            )
            self.bn2 = nn.BatchNorm1d(out_channels)
            # The training notebook uses in-place ReLU. Non-in-place ReLU is
            # numerically equivalent in eval mode and is safer for SHAP's
            # gradient hooks.
            self.relu = nn.ReLU(inplace=False)
            self.dropout = nn.Dropout(dropout)
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x: Any) -> Any:
            identity = self.shortcut(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = out + identity
            return self.relu(out)

    class CNNPQ(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=False),
            )
            self.features = nn.Sequential(
                ResidualBlock(32, 32, stride=2, dropout=0.05),
                ResidualBlock(32, 64, stride=2, dropout=0.05),
                ResidualBlock(64, 96, stride=2, dropout=0.10),
                ResidualBlock(96, 96, stride=1, dropout=0.10),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(96, 48),
                nn.ReLU(inplace=False),
                nn.Dropout(0.30),
                nn.Linear(48, num_classes),
            )

        def forward(self, x: Any) -> Any:
            x = self.stem(x)
            x = self.features(x)
            x = self.pool(x)
            return self.classifier(x)

    return CNNPQ()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_checkpoint(torch: Any, model_path: Path, device: Any) -> dict[str, Any]:
    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return torch.load(model_path, map_location=device, weights_only=False)


def load_split(
    pd: Any,
    torch: Any,
    csv_path: Path,
    checkpoint: dict[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray, list[str], list[str]]:
    csv_path = resolve_path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if frame.shape[1] != 641:
        raise ValueError(
            f"Expected one label column plus 640 signal columns; found {frame.shape[1]} columns in {csv_path}"
        )

    labels = frame.iloc[:, 0].astype(str).to_numpy()
    raw_signals = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    train_mean = float(checkpoint["train_mean"])
    train_std = float(checkpoint["train_std"])
    if train_std == 0:
        raise ValueError("Checkpoint train_std is zero; cannot normalize signals.")

    normalized = (raw_signals - train_mean) / train_std
    signals = torch.from_numpy(normalized).float().unsqueeze(1)
    class_names = [str(name) for name in checkpoint["class_names"]]
    return signals, raw_signals, labels, class_names, frame.columns.tolist()[1:]


def choose_background(signals: Any, size: int, seed: int) -> Any:
    if size < 1:
        raise ValueError("--background-size must be at least 1")
    size = min(size, len(signals))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(signals), size=size, replace=False))
    return signals[indices]


def select_class_shap_values(
    raw_values: Any,
    torch: Any,
    target_class: int,
    num_classes: int,
    n_features: int,
) -> np.ndarray:
    """Normalize SHAP's version-dependent output format to [n_features]."""

    if isinstance(raw_values, tuple):
        # Some SHAP APIs return (values, ranked_output_indices).
        raw_values = raw_values[0]

    if isinstance(raw_values, list):
        if len(raw_values) == num_classes:
            selected = raw_values[target_class]
        elif len(raw_values) == 1:
            selected = raw_values[0]
        else:
            raise ValueError(
                f"Unexpected SHAP list output: {len(raw_values)} arrays for {num_classes} classes"
            )
    else:
        selected = raw_values

    if hasattr(selected, "detach"):
        selected = selected.detach().cpu().numpy()
    values = np.asarray(selected)

    # Newer SHAP versions may place the output dimension last; older versions
    # return one array per output. Handle both, then remove batch/channel axes.
    if values.ndim == 4:
        if values.shape[-1] == num_classes:
            values = values[..., target_class]
        elif values.shape[1] == num_classes:
            values = values[:, target_class, ...]
    elif values.ndim == 3:
        if values.shape[-1] == num_classes and values.shape[1] != num_classes:
            values = values[..., target_class]
        elif values.shape[1] == num_classes and values.shape[-1] != n_features:
            values = values[:, target_class, ...]

    values = np.asarray(values).squeeze()
    if values.size != n_features:
        raise ValueError(
            f"Could not reduce SHAP output with shape {np.asarray(selected).shape} to {n_features} features"
        )
    return values.reshape(n_features).astype(np.float64)


def make_summary(
    shap_values: np.ndarray,
    raw_signal: np.ndarray,
    normalized_signal: np.ndarray,
    actual_label: str,
    predicted_label: str,
    predicted_probability: float,
    top_k: int,
) -> dict[str, Any]:
    top_k = max(1, min(top_k, len(shap_values)))
    order = np.argsort(-np.abs(shap_values))[:top_k]
    top_features = []
    for index in order:
        contribution = float(shap_values[index])
        top_features.append(
            {
                "sample_point": int(index),
                "raw_signal": float(raw_signal[index]),
                "normalized_signal": float(normalized_signal[index]),
                "shap_value": contribution,
                "absolute_shap_value": abs(contribution),
                "direction": "supports prediction" if contribution >= 0 else "opposes prediction",
            }
        )

    return {
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "predicted_probability": float(predicted_probability),
        "top_features": top_features,
        "total_positive_attribution": float(shap_values[shap_values > 0].sum()),
        "total_negative_attribution": float(shap_values[shap_values < 0].sum()),
        "largest_absolute_attribution": float(np.abs(shap_values).max()),
    }


def plot_explanation(
    plt: Any,
    shap_values: np.ndarray,
    raw_signal: np.ndarray,
    actual_label: str,
    predicted_label: str,
    predicted_probability: float,
    top_k: int,
    output_path: Path,
) -> None:
    order = np.argsort(-np.abs(shap_values))[:top_k]
    colors = np.where(shap_values >= 0, "#c43d4b", "#2e6db4")
    scale = max(float(np.abs(shap_values).max()), 1e-12)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        gridspec_kw={"height_ratios": (2.2, 1.4)},
    )
    axes[0].plot(raw_signal, color="#333333", linewidth=1.0, alpha=0.75)
    axes[0].scatter(
        np.arange(len(raw_signal)),
        raw_signal,
        c=shap_values,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        s=12,
        alpha=0.85,
    )
    axes[0].set_title(
        f"SHAP attribution | actual: {actual_label} | "
        f"predicted: {predicted_label} ({predicted_probability:.2%})"
    )
    axes[0].set_xlabel("Signal sample index")
    axes[0].set_ylabel("Original signal amplitude")
    axes[0].grid(alpha=0.25)

    axes[1].barh(
        np.arange(len(order)),
        shap_values[order][::-1],
        color=colors[order][::-1],
    )
    axes[1].set_yticks(np.arange(len(order)), [str(int(i)) for i in order[::-1]])
    axes[1].set_xlabel("SHAP value (red supports prediction, blue opposes it)")
    axes[1].set_ylabel("Most influential sample points")
    axes[1].axvline(0, color="#444444", linewidth=0.8)
    axes[1].grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.sample < 1:
        raise SystemExit("--sample must be at least 1")
    if args.nsamples < 1:
        raise SystemExit("--nsamples must be at least 1")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    torch, nn, pd, (plt, shap) = import_runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but it is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    checkpoint = load_checkpoint(torch, args.model, device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    num_classes = int(checkpoint["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError("Checkpoint class_names and num_classes do not agree.")

    model = build_model(nn, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_path = resolve_path(args.data_dir) / f"{args.split}.csv"
    signals, raw_signals, labels, class_names, signal_columns = load_split(
        pd, torch, split_path, checkpoint
    )
    if args.sample > len(signals):
        raise SystemExit(
            f"--sample must be between 1 and {len(signals)} for the {args.split} split."
        )

    sample_index = args.sample - 1
    sample = signals[sample_index : sample_index + 1].to(device)
    background = choose_background(signals, args.background_size, args.seed).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(sample), dim=1)[0]
    predicted_index = int(torch.argmax(probabilities).item())
    predicted_probability = float(probabilities[predicted_index].item())
    actual_label = str(labels[sample_index])
    predicted_label = class_names[predicted_index]

    explainer = shap.GradientExplainer(model, background)
    raw_shap_values = explainer.shap_values(sample, nsamples=args.nsamples)
    shap_values = select_class_shap_values(
        raw_shap_values,
        torch,
        predicted_index,
        num_classes,
        signals.shape[-1],
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_sample_{args.sample:03d}"

    attribution_frame = pd.DataFrame(
        {
            "sample_point": np.arange(len(shap_values)),
            "signal_column": signal_columns,
            "raw_signal": raw_signals[sample_index],
            "normalized_signal": signals[sample_index, 0].cpu().numpy(),
            "shap_value": shap_values,
            "absolute_shap_value": np.abs(shap_values),
            "direction": np.where(
                shap_values >= 0,
                "supports prediction",
                "opposes prediction",
            ),
        }
    )
    attribution_frame.to_csv(output_dir / f"{stem}_attributions.csv", index=False)
    np.save(output_dir / f"{stem}_shap_values.npy", shap_values)

    summary = make_summary(
        shap_values,
        raw_signals[sample_index],
        signals[sample_index, 0].cpu().numpy(),
        actual_label,
        predicted_label,
        predicted_probability,
        args.top_k,
    )
    summary.update(
        {
            "sample_number": args.sample,
            "split": args.split,
            "model": str(resolve_path(args.model)),
            "device": str(device),
            "background_size": int(len(background)),
            "nsamples": int(args.nsamples),
            "normalization": {
                "train_mean": float(checkpoint["train_mean"]),
                "train_std": float(checkpoint["train_std"]),
            },
        }
    )
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_explanation(
        plt,
        shap_values,
        raw_signals[sample_index],
        actual_label,
        predicted_label,
        predicted_probability,
        args.top_k,
        output_dir / f"{stem}_explanation.png",
    )

    print(f"Explained {args.split} sample {args.sample}")
    print(f"Actual: {actual_label}")
    print(f"Predicted: {predicted_label} ({predicted_probability:.2%})")
    print(f"Artifacts: {output_dir}")
    print("Top influential sample points:")
    for feature in summary["top_features"][: min(args.top_k, 10)]:
        print(
            f"  {feature['sample_point']:>3}: "
            f"{feature['shap_value']:+.6f} ({feature['direction']})"
        )


if __name__ == "__main__":
    main()
