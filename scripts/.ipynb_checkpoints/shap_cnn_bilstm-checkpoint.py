"""
SHAP explanation for PQD-16c CNN-RES-BiLSTM V2.

Standalone script.

Example:
    python shap_new.py

Specific test sample:
    python shap_new.py --split test --sample 1

Other examples:
    python shap_new.py --split test --sample 25
    python shap_new.py --split validation --sample 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    ROOT / "models" / "cnn_res_bilstm_pq.pth"
)

DEFAULT_DATA_DIR = (
    ROOT / "dataset" / "PQD-16c-Data"
)

DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "shap"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Explain one PQD-16c CNN-RES-BiLSTM "
            "prediction using SHAP."
        )
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "test"
        ),
        default="test",
        help="Dataset split. Default: test."
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help=(
            "One-based sample number "
            "within the selected split."
        )
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=32,
        help=(
            "Number of training samples used "
            "as SHAP background."
        )
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help="Number of SHAP integration samples."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help=(
            "Number of most influential signal "
            "points to report."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to trained model checkpoint."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing dataset CSV files."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SHAP artifacts."
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda"
        ),
        default="auto"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


# ============================================================
# IMPORT OPTIONAL RUNTIME DEPENDENCIES
# ============================================================

def import_runtime():

    try:

        import matplotlib.pyplot as plt
        import pandas as pd
        import shap
        import torch.nn as nn

    except ImportError as exc:

        raise SystemExit(
            "\nMissing XAI dependency.\n\n"
            "Install the required packages with:\n\n"
            "    pip install shap matplotlib pandas\n"
        ) from exc

    return (
        nn,
        pd,
        plt,
        shap
    )


# ============================================================
# RESIDUAL CNN BLOCK
# ============================================================

def build_model(
    nn: Any,
    num_classes: int
):

    class ResidualConvBlock(nn.Module):

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=5,
            dilation=1
        ):

            super().__init__()

            padding = (
                (kernel_size - 1)
                * dilation
            ) // 2

            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            )

            self.bn1 = nn.BatchNorm1d(
                out_channels
            )

            self.act1 = nn.GELU()

            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            )

            self.bn2 = nn.BatchNorm1d(
                out_channels
            )

            if in_channels != out_channels:

                self.skip = nn.Sequential(

                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        1,
                        bias=False
                    ),

                    nn.BatchNorm1d(
                        out_channels
                    )
                )

            else:

                self.skip = nn.Identity()

            self.activation = nn.GELU()


        def forward(self, x):

            identity = self.skip(x)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.bn2(x)

            return self.activation(
                x + identity
            )


    # ========================================================
    # CNN-RES-BiLSTM V2
    # ========================================================

    class CNNResBiLSTM(nn.Module):

        def __init__(
            self,
            num_classes=16
        ):

            super().__init__()

            self.cnn = nn.Sequential(

                nn.Conv1d(
                    1,
                    32,
                    7,
                    padding=3,
                    bias=False
                ),

                nn.BatchNorm1d(
                    32
                ),

                nn.GELU(),

                ResidualConvBlock(
                    32,
                    32,
                    5,
                    1
                ),

                nn.MaxPool1d(
                    2
                ),

                ResidualConvBlock(
                    32,
                    64,
                    5,
                    1
                ),

                nn.MaxPool1d(
                    2
                ),

                ResidualConvBlock(
                    64,
                    128,
                    5,
                    2
                ),

                nn.MaxPool1d(
                    2
                ),

                ResidualConvBlock(
                    128,
                    128,
                    3,
                    2
                )
            )

            self.sequence_norm = nn.LayerNorm(
                128
            )

            self.lstm = nn.LSTM(
                128,
                128,
                num_layers=2,
                batch_first=True,
                dropout=0.25,
                bidirectional=True
            )

            self.attention = nn.Sequential(

                nn.Linear(
                    256,
                    128
                ),

                nn.Tanh(),

                nn.Linear(
                    128,
                    1
                )
            )

            self.classifier = nn.Sequential(

                nn.Linear(
                    256,
                    128
                ),

                nn.GELU(),

                nn.Dropout(
                    0.30
                ),

                nn.Linear(
                    128,
                    num_classes
                )
            )


        def forward(self, x):

            if x.ndim == 4:

                x = x.squeeze(2)

            x = self.cnn(x)

            x = x.transpose(
                1,
                2
            )

            x = self.sequence_norm(
                x
            )

            x, _ = self.lstm(
                x
            )

            weights = torch.softmax(
                self.attention(x),
                dim=1
            )

            x = torch.sum(
                x * weights,
                dim=1
            )

            return self.classifier(
                x
            )


    return CNNResBiLSTM(
        num_classes=num_classes
    )


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(path: Path) -> Path:

    path = path.expanduser()

    if not path.is_absolute():

        path = ROOT / path

    return path.resolve()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model_path: Path,
    device
):

    model_path = resolve_path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n"
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    return checkpoint


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    pd,
    csv_path: Path,
    train_mean: float,
    train_std: float
):

    csv_path = resolve_path(
        csv_path
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"\nDataset file not found:\n"
            f"{csv_path}"
        )

    frame = pd.read_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Expected:
    #
    # 1 label column
    # 640 signal columns
    # --------------------------------------------------------

    if frame.shape[1] != 641:

        raise ValueError(
            f"\nExpected 641 columns "
            f"(1 label + 640 signal points).\n"
            f"Found: {frame.shape[1]}"
        )

    labels = (
        frame.iloc[:, 0]
        .astype(str)
        .to_numpy()
    )

    raw_signals = (
        frame.iloc[:, 1:]
        .to_numpy(
            dtype=np.float32
        )
    )

    if train_std == 0:

        raise ValueError(
            "Training standard deviation is zero."
        )

    normalized = (
        raw_signals - train_mean
    ) / train_std

    signals = torch.from_numpy(
        normalized
    ).float().unsqueeze(1)

    signal_columns = (
        frame.columns[1:]
        .tolist()
    )

    return (
        signals,
        raw_signals,
        labels,
        signal_columns
    )


# ============================================================
# NORMALIZATION + CLASS METADATA
# ============================================================

def get_dataset_metadata(
    pd,
    data_dir: Path,
    checkpoint: dict
):

    data_dir = resolve_path(
        data_dir
    )

    train_csv = (
        data_dir / "train.csv"
    )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    if "class_names" in checkpoint:

        class_names = [
            str(x)
            for x in checkpoint[
                "class_names"
            ]
        ]

    else:

        print(
            "\nCheckpoint does not contain "
            "'class_names'."
        )

        print(
            "Recomputing from train.csv..."
        )

        train_df = pd.read_csv(
            train_csv
        )

        class_names = sorted(
            train_df.iloc[:, 0]
            .astype(str)
            .unique()
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    if (
        "train_mean" in checkpoint
        and
        "train_std" in checkpoint
    ):

        train_mean = float(
            checkpoint[
                "train_mean"
            ]
        )

        train_std = float(
            checkpoint[
                "train_std"
            ]
        )

    else:

        print(
            "\nCheckpoint does not contain "
            "training normalization values."
        )

        print(
            "Recomputing from train.csv..."
        )

        train_df = pd.read_csv(
            train_csv
        )

        train_signals = (
            train_df.iloc[:, 1:]
            .to_numpy(
                dtype=np.float32
            )
        )

        train_mean = float(
            train_signals.mean()
        )

        train_std = float(
            train_signals.std()
        )

    return (
        class_names,
        train_mean,
        train_std
    )


# ============================================================
# CHOOSE SHAP BACKGROUND
# ============================================================

def choose_background(
    signals,
    size,
    seed
):

    if size < 1:

        raise ValueError(
            "--background-size must be >= 1"
        )

    size = min(
        size,
        len(signals)
    )

    rng = np.random.default_rng(
        seed
    )

    indices = np.sort(
        rng.choice(
            len(signals),
            size=size,
            replace=False
        )
    )

    return signals[
        indices
    ]


# ============================================================
# SHAP OUTPUT HANDLING
# ============================================================

def select_class_shap_values(
    raw_values,
    target_class,
    num_classes,
    n_features
):

    # --------------------------------------------------------
    # Some SHAP versions return:
    #
    # list[class] -> [batch, channel, features]
    #
    # Newer versions may return:
    #
    # [batch, channel, features, classes]
    # --------------------------------------------------------

    if isinstance(
        raw_values,
        tuple
    ):

        raw_values = raw_values[0]

    if isinstance(
        raw_values,
        list
    ):

        if len(raw_values) == num_classes:

            selected = (
                raw_values[
                    target_class
                ]
            )

        elif len(raw_values) == 1:

            selected = raw_values[0]

        else:

            raise ValueError(
                "Unexpected SHAP list output: "
                f"{len(raw_values)} arrays."
            )

    else:

        selected = raw_values

    if hasattr(
        selected,
        "detach"
    ):

        selected = (
            selected
            .detach()
            .cpu()
            .numpy()
        )

    values = np.asarray(
        selected
    )

    # --------------------------------------------------------
    # Remove output class dimension
    # --------------------------------------------------------

    if values.ndim == 4:

        if values.shape[-1] == num_classes:

            values = values[
                ...,
                target_class
            ]

        elif values.shape[1] == num_classes:

            values = values[
                :,
                target_class,
                ...
            ]

    elif values.ndim == 3:

        if (
            values.shape[-1] == num_classes
            and
            values.shape[1] != num_classes
        ):

            values = values[
                ...,
                target_class
            ]

        elif (
            values.shape[1] == num_classes
            and
            values.shape[-1] != n_features
        ):

            values = values[
                :,
                target_class,
                ...
            ]

    values = np.asarray(
        values
    ).squeeze()

    if values.size != n_features:

        raise ValueError(
            "\nCould not reduce SHAP output.\n"
            f"SHAP shape: {np.asarray(selected).shape}\n"
            f"Expected features: {n_features}\n"
            f"Final shape: {values.shape}"
        )

    return values.reshape(
        n_features
    ).astype(
        np.float64
    )


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    shap_values,
    raw_signal,
    normalized_signal,
    actual_label,
    predicted_label,
    predicted_probability,
    top_k
):

    top_k = max(
        1,
        min(
            top_k,
            len(shap_values)
        )
    )

    order = np.argsort(
        -np.abs(shap_values)
    )[:top_k]

    top_features = []

    for index in order:

        contribution = float(
            shap_values[index]
        )

        top_features.append({

            "sample_point": int(
                index
            ),

            "raw_signal": float(
                raw_signal[index]
            ),

            "normalized_signal": float(
                normalized_signal[index]
            ),

            "shap_value": contribution,

            "absolute_shap_value": abs(
                contribution
            ),

            "direction": (
                "supports prediction"
                if contribution >= 0
                else "opposes prediction"
            )
        })

    return {

        "actual_label":
            actual_label,

        "predicted_label":
            predicted_label,

        "predicted_probability":
            float(
                predicted_probability
            ),

        "top_features":
            top_features,

        "total_positive_attribution":
            float(
                shap_values[
                    shap_values > 0
                ].sum()
            ),

        "total_negative_attribution":
            float(
                shap_values[
                    shap_values < 0
                ].sum()
            ),

        "largest_absolute_attribution":
            float(
                np.abs(
                    shap_values
                ).max()
            )
    }


# ============================================================
# PLOT SHAP EXPLANATION
# ============================================================

def plot_explanation(
    plt,
    shap_values,
    raw_signal,
    actual_label,
    predicted_label,
    predicted_probability,
    top_k,
    output_path
):

    order = np.argsort(
        -np.abs(shap_values)
    )[:top_k]

    scale = max(
        float(
            np.abs(
                shap_values
            ).max()
        ),
        1e-12
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(
            15,
            9
        ),
        gridspec_kw={
            "height_ratios": (
                2.2,
                1.4
            )
        }
    )

    # --------------------------------------------------------
    # Signal + SHAP attribution
    # --------------------------------------------------------

    axes[0].plot(
        raw_signal,
        linewidth=1.0,
        alpha=0.75
    )

    axes[0].scatter(
        np.arange(
            len(raw_signal)
        ),
        raw_signal,
        c=shap_values,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        s=12,
        alpha=0.85
    )

    axes[0].set_title(
        "SHAP Attribution\n"
        f"Actual: {actual_label} | "
        f"Predicted: {predicted_label} "
        f"({predicted_probability:.2%})"
    )

    axes[0].set_xlabel(
        "Signal sample index"
    )

    axes[0].set_ylabel(
        "Original signal amplitude"
    )

    axes[0].grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Most influential points
    # --------------------------------------------------------

    selected_values = (
        shap_values[
            order
        ][::-1]
    )

    axes[1].barh(
        np.arange(
            len(order)
        ),
        selected_values
    )

    axes[1].set_yticks(
        np.arange(
            len(order)
        )
    )

    axes[1].set_yticklabels(
        [
            str(int(i))
            for i in order[::-1]
        ]
    )

    axes[1].set_xlabel(
        "SHAP value"
    )

    axes[1].set_ylabel(
        "Signal sample point"
    )

    axes[1].axvline(
        0,
        linewidth=0.8
    )

    axes[1].grid(
        axis="x",
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.sample < 1:

        raise SystemExit(
            "--sample must be at least 1."
        )

    if args.nsamples < 1:

        raise SystemExit(
            "--nsamples must be at least 1."
        )

    if args.top_k < 1:

        raise SystemExit(
            "--top-k must be at least 1."
        )

    # --------------------------------------------------------
    # Runtime imports
    # --------------------------------------------------------

    nn, pd, plt, shap = (
        import_runtime()
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if (
        args.device == "cuda"
        and
        not torch.cuda.is_available()
    ):

        raise SystemExit(
            "CUDA requested but CUDA "
            "is not available."
        )

    if args.device == "cuda":

        device = torch.device(
            "cuda"
        )

    elif args.device == "cpu":

        device = torch.device(
            "cpu"
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PQD-16c CNN-RES-BiLSTM V2 SHAP"
    )
    print("=" * 70)

    print(
        "Device     :",
        device
    )

    print(
        "Model      :",
        resolve_path(
            args.model
        )
    )

    print(
        "Data       :",
        resolve_path(
            args.data_dir
        )
    )

    print(
        "Split      :",
        args.split
    )

    print(
        "Sample     :",
        args.sample
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        args.model,
        device
    )

    num_classes = int(
        checkpoint.get(
            "num_classes",
            16
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    (
        class_names,
        train_mean,
        train_std
    ) = get_dataset_metadata(
        pd,
        args.data_dir,
        checkpoint
    )

    if len(class_names) != num_classes:

        raise ValueError(
            "Number of class names does not "
            "match num_classes."
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        nn,
        num_classes
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # Load selected split
    # --------------------------------------------------------

    split_path = (
        resolve_path(
            args.data_dir
        )
        /
        f"{args.split}.csv"
    )

    (
        signals,
        raw_signals,
        labels,
        signal_columns
    ) = load_dataset(
        pd,
        split_path,
        train_mean,
        train_std
    )

    # --------------------------------------------------------
    # Validate sample
    # --------------------------------------------------------

    if args.sample > len(
        signals
    ):

        raise SystemExit(
            f"\nSample must be between "
            f"1 and {len(signals)}."
        )

    sample_index = (
        args.sample - 1
    )

    # --------------------------------------------------------
    # Select sample
    # --------------------------------------------------------

    sample = signals[
        sample_index:
        sample_index + 1
    ].to(device)

    # --------------------------------------------------------
    # SHAP background
    #
    # IMPORTANT:
    # Background comes from TRAINING DATA.
    # --------------------------------------------------------

    train_path = (
        resolve_path(
            args.data_dir
        )
        /
        "train.csv"
    )

    (
        train_signals,
        _,
        _,
        _
    ) = load_dataset(
        pd,
        train_path,
        train_mean,
        train_std
    )

    background = choose_background(
        train_signals,
        args.background_size,
        args.seed
    ).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            sample
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_probability = float(
            probabilities[
                predicted_index
            ].item()
        )

    actual_label = str(
        labels[
            sample_index
        ]
    )

    predicted_label = (
        class_names[
            predicted_index
        ]
    )

    # --------------------------------------------------------
    # Print prediction
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL PREDICTION")
    print("=" * 70)

    print(
        f"Sample       : {args.sample}"
    )

    print(
        f"Actual       : {actual_label}"
    )

    print(
        f"Predicted    : {predicted_label}"
    )

    print(
        f"Confidence   : "
        f"{predicted_probability:.2%}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print()
    print(
        "Creating SHAP explainer..."
    )

    explainer = shap.GradientExplainer(
        model,
        background
    )

    print(
        "Calculating SHAP values..."
    )

    raw_shap_values = (
        explainer.shap_values(
            sample,
            nsamples=args.nsamples
        )
    )

    shap_values = (
        select_class_shap_values(
            raw_shap_values,
            predicted_index,
            num_classes,
            signals.shape[-1]
        )
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = (
        f"{args.split}"
        f"_sample_"
        f"{args.sample:03d}"
    )

    # --------------------------------------------------------
    # Attribution CSV
    # --------------------------------------------------------

    normalized_signal = (
        signals[
            sample_index,
            0
        ]
        .cpu()
        .numpy()
    )

    attribution_frame = pd.DataFrame({

        "sample_point":
            np.arange(
                len(shap_values)
            ),

        "signal_column":
            signal_columns,

        "raw_signal":
            raw_signals[
                sample_index
            ],

        "normalized_signal":
            normalized_signal,

        "shap_value":
            shap_values,

        "absolute_shap_value":
            np.abs(
                shap_values
            ),

        "direction":
            np.where(
                shap_values >= 0,
                "supports prediction",
                "opposes prediction"
            )
    })

    attribution_frame.to_csv(
        output_dir /
        f"{stem}_attributions.csv",
        index=False
    )

    # --------------------------------------------------------
    # NPY
    # --------------------------------------------------------

    np.save(
        output_dir /
        f"{stem}_shap_values.npy",
        shap_values
    )

    # --------------------------------------------------------
    # JSON SUMMARY
    # --------------------------------------------------------

    summary = make_summary(
        shap_values,
        raw_signals[
            sample_index
        ],
        normalized_signal,
        actual_label,
        predicted_label,
        predicted_probability,
        args.top_k
    )

    summary.update({

        "sample_number":
            args.sample,

        "split":
            args.split,

        "model":
            str(
                resolve_path(
                    args.model
                )
            ),

        "architecture":
            "CNNResBiLSTM",

        "device":
            str(device),

        "background_size":
            int(
                len(background)
            ),

        "nsamples":
            int(
                args.nsamples
            ),

        "normalization": {

            "train_mean":
                float(
                    train_mean
                ),

            "train_std":
                float(
                    train_std
                )
        }
    })

    with open(
        output_dir /
        f"{stem}_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_explanation(
        plt,
        shap_values,
        raw_signals[
            sample_index
        ],
        actual_label,
        predicted_label,
        predicted_probability,
        args.top_k,
        output_dir /
        f"{stem}_explanation.png"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SHAP EXPLANATION COMPLETE")
    print("=" * 70)

    print(
        f"Actual       : {actual_label}"
    )

    print(
        f"Predicted    : {predicted_label}"
    )

    print(
        f"Confidence   : "
        f"{predicted_probability:.2%}"
    )

    print(
        f"Artifacts    : {output_dir}"
    )

    print()
    print(
        "Top influential signal points:"
    )

    for feature in summary[
        "top_features"
    ][:min(args.top_k, 10)]:

        print(
            f"  "
            f"{feature['sample_point']:>3} : "
            f"{feature['shap_value']:+.6f} "
            f"({feature['direction']})"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()