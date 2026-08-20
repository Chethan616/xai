"""
Attention explanation utilities for PQD-16c CNN-Res-BiLSTM V2.

Extracts the learned temporal attention weights from the trained model.

The model processes the 640-point waveform through CNN layers with
three MaxPool1d(2) operations, reducing the temporal dimension:

    640 -> 320 -> 160 -> 80

The BiLSTM therefore produces 80 temporal representations.

The attention layer assigns one weight to each of these 80 positions.
For visualization, the 80 attention weights can be interpolated back
to the original 640 signal points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# ============================================================
# EXTRACT RAW ATTENTION WEIGHTS
# ============================================================

def get_attention_weights(
    model: torch.nn.Module,
    sample: torch.Tensor
) -> torch.Tensor:
    """
    Extract normalized temporal attention weights.

    Parameters
    ----------
    model:
        Trained CNN-Res-BiLSTM model.

    sample:
        Input tensor of shape:
            [1, 1, 640]

    Returns
    -------
    torch.Tensor
        Attention weights with shape:
            [sequence_length]

        For the PQD-16c model this should normally be:
            [80]

        The weights sum to 1.
    """

    model.eval()

    with torch.no_grad():

        # ----------------------------------------------------
        # CNN feature extraction
        # ----------------------------------------------------

        x = sample

        if x.ndim == 4:
            x = x.squeeze(2)

        x = model.cnn(x)

        # ----------------------------------------------------
        # Convert:
        #
        # [batch, channels, time]
        #
        # to:
        #
        # [batch, time, channels]
        # ----------------------------------------------------

        x = x.transpose(1, 2)

        # ----------------------------------------------------
        # Layer normalization
        # ----------------------------------------------------

        x = model.sequence_norm(x)

        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        x, _ = model.lstm(x)

        # ----------------------------------------------------
        # Attention scores
        #
        # Shape:
        # [batch, time, 1]
        # ----------------------------------------------------

        scores = model.attention(x)

        # ----------------------------------------------------
        # Convert scores into normalized attention weights
        # ----------------------------------------------------

        weights = torch.softmax(
            scores,
            dim=1
        )

        # ----------------------------------------------------
        # Remove batch/channel dimensions
        # ----------------------------------------------------

        weights = weights[0, :, 0]

    return weights


# ============================================================
# CONVERT ATTENTION TO NUMPY
# ============================================================

def attention_to_numpy(
    model: torch.nn.Module,
    sample: torch.Tensor
) -> np.ndarray:
    """
    Return attention weights as a NumPy array.
    """

    weights = get_attention_weights(
        model,
        sample
    )

    return (
        weights
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


# ============================================================
# INTERPOLATE ATTENTION TO ORIGINAL SIGNAL LENGTH
# ============================================================

def interpolate_attention(
    attention_weights: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Interpolate CNN/BiLSTM attention weights from their
    reduced temporal resolution back to the original
    signal length.

    Example:

        80 attention values
              ↓
        interpolation
              ↓
        640 values

    IMPORTANT:
    The interpolated values are for visualization.
    The model itself only produced attention at the
    reduced temporal sequence positions.
    """

    attention_weights = np.asarray(
        attention_weights,
        dtype=np.float64
    ).reshape(-1)

    if target_length < 1:
        raise ValueError(
            "target_length must be >= 1."
        )

    if len(attention_weights) == target_length:
        return attention_weights.copy()

    source_positions = np.linspace(
        0,
        target_length - 1,
        len(attention_weights)
    )

    target_positions = np.arange(
        target_length
    )

    interpolated = np.interp(
        target_positions,
        source_positions,
        attention_weights
    )

    return interpolated


# ============================================================
# NORMALIZE FOR VISUALIZATION
# ============================================================

def normalize_attention(
    attention_weights: np.ndarray
) -> np.ndarray:
    """
    Normalize attention values to [0, 1] for visualization.

    This does NOT change the original attention values.
    """

    attention_weights = np.asarray(
        attention_weights,
        dtype=np.float64
    )

    minimum = attention_weights.min()
    maximum = attention_weights.max()

    if maximum - minimum < 1e-12:
        return np.zeros_like(
            attention_weights
        )

    return (
        attention_weights - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# TOP ATTENTION POINTS
# ============================================================

def get_top_attention_points(
    attention_weights: np.ndarray,
    top_k: int = 8
) -> np.ndarray:
    """
    Return indices of the most strongly attended positions.

    Returns indices sorted from highest attention
    to lowest attention.
    """

    attention_weights = np.asarray(
        attention_weights
    ).reshape(-1)

    top_k = max(
        1,
        min(
            top_k,
            len(attention_weights)
        )
    )

    return np.argsort(
        attention_weights
    )[::-1][:top_k]


# ============================================================
# ATTENTION SUMMARY
# ============================================================

def make_attention_summary(
    attention_weights: np.ndarray,
    raw_signal: np.ndarray,
    top_k: int = 8
) -> dict[str, Any]:
    """
    Create an interpretable attention summary.
    """

    attention_weights = np.asarray(
        attention_weights
    ).reshape(-1)

    raw_signal = np.asarray(
        raw_signal
    ).reshape(-1)

    top_indices = get_top_attention_points(
        attention_weights,
        top_k
    )

    top_points = []

    for rank, index in enumerate(
        top_indices,
        1
    ):

        top_points.append({

            "rank": int(rank),

            "sample_point": int(index),

            "signal_value": float(
                raw_signal[index]
            ),

            "attention_weight": float(
                attention_weights[index]
            )
        })

    return {

        "sequence_length":
            int(len(attention_weights)),

        "attention_sum":
            float(attention_weights.sum()),

        "maximum_attention":
            float(attention_weights.max()),

        "mean_attention":
            float(attention_weights.mean()),

        "top_points":
            top_points
    }


# ============================================================
# COMPLETE ATTENTION EXTRACTION
# ============================================================

def explain_attention(
    model: torch.nn.Module,
    sample: torch.Tensor,
    signal_length: int = 640,
    top_k: int = 8
) -> dict[str, Any]:
    """
    Complete attention explanation.

    Returns:

        raw_attention
            Actual attention weights produced by the model.

        interpolated_attention
            Attention mapped to the original signal length.

        normalized_attention
            0-1 version useful for plotting.

        top_points
            Most strongly attended original signal points.
    """

    raw_attention = attention_to_numpy(
        model,
        sample
    )

    interpolated = interpolate_attention(
        raw_attention,
        signal_length
    )

    normalized = normalize_attention(
        interpolated
    )

    top_indices = get_top_attention_points(
        interpolated,
        top_k
    )

    return {

        "raw_attention":
            raw_attention,

        "interpolated_attention":
            interpolated,

        "normalized_attention":
            normalized,

        "top_indices":
            top_indices,

        "summary":
            make_attention_summary(
                interpolated,
                sample[0, 0]
                .detach()
                .cpu()
                .numpy(),
                top_k
            )
    }

# ============================================================
# MAIN — INDEPENDENT ATTENTION TEST
# ============================================================

def main():

    import sys
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # PROJECT PATH
    # --------------------------------------------------------

    ROOT = Path(__file__).resolve().parents[1]

    MODEL_PATH = (
        ROOT
        / "models"
        / "cnn_res_bilstm_pq.pth"
    )

    DATA_DIR = (
        ROOT
        / "dataset"
        / "PQD-16c-Data"
    )

    TEST_PATH = DATA_DIR / "test.csv"

    # --------------------------------------------------------
    # IMPORT MODEL UTILITIES
    # --------------------------------------------------------

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent)
    )

    import shap_cnn_bilstm as shap_lib

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("PQD-16c CNN-RES-BiLSTM ATTENTION ANALYSIS")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Model  : {MODEL_PATH}")
    print(f"Test   : {TEST_PATH}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = shap_lib.load_checkpoint(
        MODEL_PATH,
        device
    )

    num_classes = int(
        checkpoint.get(
            "num_classes",
            16
        )
    )

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    (
        class_names,
        train_mean,
        train_std
    ) = shap_lib.get_dataset_metadata(
        pd,
        DATA_DIR,
        checkpoint
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    nn = shap_lib.import_runtime()[0]

    model = shap_lib.build_model(
        nn,
        num_classes
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    (
        signals,
        raw_signals,
        labels,
        signal_columns
    ) = shap_lib.load_dataset(
        pd,
        TEST_PATH,
        train_mean,
        train_std
    )

    print()
    print(f"Test samples : {len(signals)}")
    print(f"Signal points: {signals.shape[-1]}")

    # --------------------------------------------------------
    # SAMPLE SELECTION
    # --------------------------------------------------------

    while True:

        choice = input(
            "\nAnalyze a random sample or choose one? "
            "[random/sample]: "
        ).strip().lower()

        if choice == "random":

            sample_idx = np.random.randint(
                0,
                len(signals)
            )

            print(
                f"\nRandomly selected sample: "
                f"{sample_idx + 1}"
            )

            break

        elif choice == "sample":

            try:

                number = int(
                    input(
                        f"Enter sample number "
                        f"(1-{len(signals)}): "
                    )
                )

                if 1 <= number <= len(signals):

                    sample_idx = number - 1

                    break

                print(
                    f"Please enter a number "
                    f"between 1 and {len(signals)}."
                )

            except ValueError:

                print(
                    "Please enter a valid integer."
                )

        else:

            print(
                "Please enter 'random' or 'sample'."
            )

    # --------------------------------------------------------
    # SELECT SAMPLE
    # --------------------------------------------------------

    sample = signals[
        sample_idx:
        sample_idx + 1
    ].to(device)

    raw_signal = raw_signals[
        sample_idx
    ]

    actual_class = str(
        labels[
            sample_idx
        ]
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(sample)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_idx = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[
                predicted_idx
            ].item()
        )

    predicted_class = class_names[
        predicted_idx
    ]

    correct = (
        actual_class ==
        predicted_class
    )

    # --------------------------------------------------------
    # PREDICTION OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print(
        f"Sample number : {sample_idx + 1}"
    )

    print(
        f"Actual class  : {actual_class}"
    )

    print(
        f"Predicted     : {predicted_class}"
    )

    print(
        f"Confidence    : {confidence:.2%}"
    )

    print(
        f"Result        : "
        f"{'✓ CORRECT' if correct else '✗ INCORRECT'}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # ATTENTION
    # --------------------------------------------------------

    print()
    print("Extracting attention weights...")

    explanation = explain_attention(
        model,
        sample,
        signal_length=len(raw_signal),
        top_k=8
    )

    raw_attention = explanation[
        "raw_attention"
    ]

    interpolated_attention = explanation[
        "interpolated_attention"
    ]

    normalized_attention = explanation[
        "normalized_attention"
    ]

    top_indices = explanation[
        "top_indices"
    ]

    # --------------------------------------------------------
    # ATTENTION INFORMATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ATTENTION ANALYSIS")
    print("=" * 70)

    print(
        f"BiLSTM sequence length : "
        f"{len(raw_attention)}"
    )

    print(
        f"Original signal length : "
        f"{len(raw_signal)}"
    )

    print(
        f"Attention sum          : "
        f"{raw_attention.sum():.6f}"
    )

    print(
        f"Maximum attention      : "
        f"{raw_attention.max():.6f}"
    )

    print(
        f"Mean attention         : "
        f"{raw_attention.mean():.6f}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # TOP ATTENTION POINTS
    # --------------------------------------------------------

    print()
    print("TOP 8 MOST ATTENDED SIGNAL POINTS")
    print("-" * 70)

    for rank, idx in enumerate(
        top_indices,
        1
    ):

        idx = int(idx)

        print(
            f"{rank:>2}. "
            f"Point {idx:>3} | "
            f"Signal = {raw_signal[idx]:>9.4f} | "
            f"Attention = "
            f"{interpolated_attention[idx]:.6f}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # ATTENTION INTERPRETATION
    # --------------------------------------------------------

    strongest_idx = int(
        top_indices[0]
    )

    print()
    print("INTERPRETATION")
    print("-" * 70)

    print(
        f"The model placed its strongest attention "
        f"around signal point {strongest_idx}."
    )

    print(
        f"The predicted class is "
        f"'{predicted_class}' "
        f"with {confidence:.2%} confidence."
    )

    print(
        "Higher attention means the learned attention "
        "mechanism assigned more weight to that temporal "
        "representation."
    )

    print(
        "Attention indicates model focus; it does not "
        "by itself prove causal importance."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    x = np.arange(
        len(raw_signal)
    )

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    ax.plot(
        x,
        raw_signal,
        linewidth=1.4,
        label="PQD Signal",
        zorder=2
    )

    # --------------------------------------------------------
    # ATTENTION CURVE
    # --------------------------------------------------------

    # Scale attention to waveform range only
    # for visualization.

    signal_min = float(
        raw_signal.min()
    )

    signal_max = float(
        raw_signal.max()
    )

    signal_range = max(
        signal_max - signal_min,
        1e-12
    )

    attention_scaled = (
        signal_min
        +
        normalized_attention
        * signal_range
        * 0.30
    )

    attention_scaled += (
        signal_max
        +
        signal_range * 0.05
    )

    ax.plot(
        x,
        attention_scaled,
        linewidth=2.0,
        alpha=0.8,
        label="Attention"
    )

    # --------------------------------------------------------
    # TOP ATTENTION POINTS
    # --------------------------------------------------------

    for idx in top_indices:

        idx = int(idx)

        ax.scatter(
            idx,
            raw_signal[idx],
            s=90,
            edgecolors="black",
            linewidths=0.8,
            zorder=5
        )

    # --------------------------------------------------------
    # ANNOTATIONS
    # --------------------------------------------------------

    label_height = (
        signal_max
        +
        signal_range * 0.42
    )

    spacing = (
        signal_range * 0.08
    )

    for rank, idx in enumerate(
        top_indices,
        1
    ):

        idx = int(idx)

        y = (
            label_height
            +
            rank * spacing
        )

        ax.annotate(
            f"#{rank} Point {idx}\n"
            f"Attention "
            f"{interpolated_attention[idx]:.4f}",
            xy=(
                idx,
                raw_signal[idx]
            ),
            xytext=(
                idx,
                y
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                alpha=0.9
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=0.8
            )
        )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    result = (
        "✓ CORRECT"
        if correct
        else
        "✗ INCORRECT"
    )

    ax.set_title(
        "PQD-16c CNN-Res-BiLSTM — Attention Explanation\n"
        f"Sample {sample_idx + 1} | "
        f"Actual: {actual_class} | "
        f"Predicted: {predicted_class} | "
        f"Confidence: {confidence:.2%} | "
        f"{result}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    ax.set_xlabel(
        "Signal sample point"
    )

    ax.set_ylabel(
        "Signal amplitude"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    # --------------------------------------------------------
    # EXTRA SPACE FOR LABELS
    # --------------------------------------------------------

    ax.set_ylim(
        signal_min - signal_range * 0.05,
        signal_max + signal_range * 1.15
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()