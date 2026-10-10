# Datasets

Two public, synthetic power-quality-disturbance (PQD) datasets are used. Both
are committed as CSV so the repository runs straight after cloning.

| | **XPQRS** (headline) | **PQD-16c** (secondary) |
|---|---|---|
| Folder | `dataset/XPQRS/` | `dataset/PQD-16c-Data/` |
| Classes | 17 | 16 |
| Signals | 17 000 (1 000 / class) | 1 600 (100 / class) |
| Signal length | 100 samples = 1 cycle | 640 samples = 10 cycles |
| Sampling | 5 kHz, 50 Hz fundamental | 64 samples / cycle |
| Amplitude | scaled to −1…1 (1 = nominal peak) | 1 = nominal peak |
| Split | stratified 64 / 16 / 20 %, `random_state=42` (done in `src/data.py`) | fixed CSVs: 1024 / 256 / 320 |
| Source | [Zenodo – SEED_PQD_v1](https://zenodo.org/records/11843312) | [Kaggle – PQD-16c](https://www.kaggle.com/datasets/jkexin/pqd-16c-data) |
| Licence | CC BY 4.0 | see the Kaggle page |

## File formats

**XPQRS** — one headerless CSV per class (`Sag.csv`, `Interruption.csv`, …),
1 000 rows × 100 columns. The class name is the file name. `Details.txt` is the
original dataset description.

**PQD-16c** — `PQD-16C-Data.csv` holds all 1 600 signals; `train.csv`,
`validation.csv` and `test.csv` are the stratified splits used everywhere in
this project. Column 0 is the class label, columns 1–640 the signal.
`class_distribution.txt` lists the per-class counts (100 each).

## Classes

| XPQRS | PQD-16c |
|---|---|
| Pure_Sinusoidal, Sag, Swell, Interruption, Transient, Oscillatory_Transient, Harmonics, Harmonics_with_Sag, Harmonics_with_Swell, Flicker, Flicker_with_Sag, Flicker_with_Swell, Sag_with_Oscillatory_Transient, Swell_with_Oscillatory_Transient, Sag_with_Harmonics, Swell_with_Harmonics, Notch | Normal, Sag, Swell, Interruption, Impulsive Transient, Oscillatory transient, Harmonics, Flicker, Notch, Spike, Flicker+Harmonics, Flicker+Sag, Flicker+Swell, Interruption+Harmonics, Sag+Harmonics, Swell+Harmonics |

## Known data issues (measured, not assumed)

* **PQD-16c — Interruption is labelled with sag depth.** IEEE 1159 defines an
  interruption as voltage below 0.1 p.u. None of the 200 *Interruption* /
  *Interruption+Harmonics* signals ever drops below 0.1 p.u. (lowest 1-cycle
  RMS: 0.101), and their depth distribution matches *Sag* / *Sag+Harmonics*
  (median 0.53 vs 0.58 p.u.). The two pairs are therefore indistinguishable,
  which caps 16-class accuracy at ≈ 87.5 %. See
  [`docs/figures/pqd16_label_noise.png`](../docs/figures/pqd16_label_noise.png)
  and [`docs/XAI.md`](../docs/XAI.md#what-xai-revealed-about-pqd-16c).
* **XPQRS — Pure_Sinusoidal is one signal repeated 1 000 times** (no noise), so
  its 200 test samples also occur in training. Removing every test signal that
  has an exact copy in training changes ResNet-CAM accuracy from 98.3 % to
  98.2 %, i.e. the headline number is not driven by this. XPQRS signals are
  noise-free; robustness to measurement noise has not been evaluated yet.

## Citation

If you use XPQRS, cite the original authors:

> M. U. Khan, S. Aziz, A. Usman, "XPQRS: Expert power quality recognition
> system for sensitive load applications," *Measurement*, vol. 216, 112889,
> 2023. doi:[10.1016/j.measurement.2023.112889](https://doi.org/10.1016/j.measurement.2023.112889)
