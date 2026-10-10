"""Dataset loaders. Both return (X, y, raw, class_names, mean, std) where
X/raw/y are dicts keyed by split ("train", "validation", "test"), X is
z-scored with training-set mean/std only, and raw is the un-normalised signal.

PQD-16c: 16 classes x 100 signals, 640 points = 10 cycles x 64 samples/cycle,
         fixed CSV splits (1024 / 256 / 320).
XPQRS:   17 classes x 1000 signals, 100 points = 1 cycle @ 5 kHz,
         stratified 64/16/20 split.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "PQD-16c-Data"


def load_splits():
    dfs = {s: pd.read_csv(DATA_DIR / f"{s}.csv") for s in ("train", "validation", "test")}
    class_names = sorted(dfs["train"].iloc[:, 0].unique())
    idx = {c: i for i, c in enumerate(class_names)}
    raw = {s: d.iloc[:, 1:].values.astype(np.float32) for s, d in dfs.items()}
    y = {s: np.array([idx[c] for c in d.iloc[:, 0]], dtype=np.int64) for s, d in dfs.items()}
    mean, std = float(raw["train"].mean()), float(raw["train"].std())
    X = {s: (r - mean) / std for s, r in raw.items()}
    return X, y, raw, class_names, mean, std


XPQRS_DIR = ROOT / "dataset" / "XPQRS"


def load_xpqrs(seed=42):
    """XPQRS (17 classes x 1000 signals, 1 cycle = 100 samples @ 5 kHz).
    Stratified 64/16/20 train/val/test split, same ratios as PQD-16c."""
    from sklearn.model_selection import train_test_split
    files = sorted(XPQRS_DIR.glob("*.csv"))
    class_names = [f.stem for f in files]
    sig = np.vstack([pd.read_csv(f, header=None).values for f in files]).astype(np.float32)
    lab = np.repeat(np.arange(len(files)), [1000] * len(files)).astype(np.int64)
    idx = np.arange(len(lab))
    tr, te = train_test_split(idx, test_size=0.2, stratify=lab, random_state=seed)
    tr, va = train_test_split(tr, test_size=0.2, stratify=lab[tr], random_state=seed)
    split = {"train": tr, "validation": va, "test": te}
    raw = {s: sig[i] for s, i in split.items()}
    y = {s: lab[i] for s, i in split.items()}
    mean, std = float(raw["train"].mean()), float(raw["train"].std())
    X = {s: (r - mean) / std for s, r in raw.items()}
    return X, y, raw, class_names, mean, std
