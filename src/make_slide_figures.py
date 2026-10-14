"""Slide-sized figures for the project presentation, written to docs/figures/slides/.

    python src/make_slide_figures.py
    python src/make_slide_figures.py --gui shot.png    # also the zoomed GUI view

Each figure is drawn at its final size on a 16:9 slide (12.1 x 5.0 in), so the
point sizes here are the point sizes on the slide. For the GUI view, take the
screenshot first with
    QT_QPA_PLATFORM=offscreen QT_SCALE_FACTOR=2 python src/gui.py --demo shot.png --theme light
"""
import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from data import ROOT, load_splits, load_xpqrs  # noqa: E402
from faithfulness import integrated_gradients  # noqa: E402
from features import extract  # noqa: E402
from models import MODELS  # noqa: E402
from PIL import Image  # noqa: E402

OUT = ROOT / "docs" / "figures" / "slides"
NAVY, BLUE, LIGHT, ORANGE, TEAL, PURPLE = "#1f4e79", "#2a78d6", "#a9c8ee", "#eb6834", "#1b9e77", "#7b61b8"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
FILL = {"navy": ("#1f4e79", "white"), "blue": ("#dcebfb", INK), "orange": ("#fde3d6", INK),
        "teal": ("#d5f0e6", INK), "purple": ("#e8e2f6", INK), "grey": ("#f1f0ec", INK)}
EDGE = {"navy": NAVY, "blue": BLUE, "orange": ORANGE, "teal": TEAL, "purple": PURPLE, "grey": "#b9b7ae"}
FIG_W, FIG_H = 12.1, 5.0

plt.rcParams.update({
    "font.size": 13, "axes.edgecolor": "#c9c7bf", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "savefig.dpi": 220, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def nice(name):
    n = name.replace("Pure_Sinusoidal", "Normal").replace("Oscillatory_Transient", "Osc. Transient")
    return n.replace("_with_", " + ").replace("_", " ")


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------- flowchart helpers (coordinates in inches) ----------
def canvas(w=FIG_W, h=FIG_H):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub="", style="blue", fs=14, sfs=11.5):
    face, txt = FILL[style]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=face, ec=EDGE[style], lw=1.8))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.66, title, ha="center", va="center", fontsize=fs, weight="bold", color=txt)
        ax.text(cx, y + h * 0.30, sub, ha="center", va="center", fontsize=sfs, color=txt, linespacing=1.3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold", color=txt,
                linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, a, b, label="", color="#6b6a64", rad=0.0, lfs=11):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color,
                                                    mutation_scale=18, shrinkA=2, shrinkB=2,
                                                    connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.14, label, ha="center", va="bottom",
                fontsize=lfs, color=MUTED, style="italic")


def R(b):  # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def T(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def B(b):
    return (b[0] + b[2] / 2, b[1])


# ---------- figures ----------
def waveforms():
    X, y, raw, names, *_ = load_xpqrs()
    order = ["Pure_Sinusoidal", "Sag", "Swell", "Interruption", "Transient", "Oscillatory_Transient",
             "Harmonics", "Flicker", "Notch", "Harmonics_with_Sag", "Harmonics_with_Swell",
             "Flicker_with_Sag", "Flicker_with_Swell", "Sag_with_Harmonics", "Swell_with_Harmonics",
             "Sag_with_Oscillatory_Transient", "Swell_with_Oscillatory_Transient"]
    fig, axes = plt.subplots(3, 6, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
    t = np.arange(100) * 0.2  # ms
    for ax, cls in zip(axes.flat, order):
        i = int(np.flatnonzero(y["test"] == names.index(cls))[0])
        ax.plot(t, raw["test"][i], color=NAVY, lw=1.6)
        ax.set_title(nice(cls).replace("Osc. Transient", "Osc. Tr.") if "+" in nice(cls) else nice(cls),
                     fontsize=11.5, loc="center", weight="bold")
        ax.set_ylim(-1.9, 1.9)
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([0, 10, 20])
        ax.tick_params(labelsize=10)
        ax.grid(True)
    last = axes.flat[-1]
    last.axis("off")
    last.text(0.5, 0.5, "XPQRS test signals\n1 cycle = 20 ms\n100 samples, 5 kHz\ny-axis: p.u.\nx-axis: ms",
              ha="center", va="center", fontsize=11.5, color=MUTED, transform=last.transAxes, linespacing=1.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "waveforms.png")


def preprocess():
    fig, ax = canvas()
    w, h = 2.55, 1.25
    xp = box(ax, 0.1, 3.35, w, h, "XPQRS (headline)", "17 classes x 1000 signals\n1 cycle = 100 samples", "navy")
    pq = box(ax, 0.1, 0.55, w, h, "PQD-16c", "16 classes x 100 signals\n10 cycles = 640 samples", "navy")
    s1 = box(ax, 3.2, 3.35, w, h, "Stratified split", "64 / 16 / 20 %, seed 42\n10 880 / 2 720 / 3 400", "blue")
    s2 = box(ax, 3.2, 0.55, w, h, "Fixed CSV split", "train / val / test\n1 024 / 256 / 320", "blue")
    z = box(ax, 6.3, 1.95, w, h, "z-score normalise", "mean and std from the\ntraining split only", "teal")
    m = box(ax, 9.4, 3.35, w, h, "ResNet-CAM input", "tensor 1 x L\nL = 100 or 640", "orange")
    f = box(ax, 9.4, 0.55, w, h, "48 physics features", "RMS, THD, harmonics,\nflicker, transients", "purple")
    sy = box(ax, 6.3, 3.95, w, 0.95, "Zero-crossing sync", "GUI model only", "grey", fs=12.5, sfs=11)
    arrow(ax, R(xp), L(s1))
    arrow(ax, R(pq), L(s2))
    arrow(ax, R(s1), (z[0], z[1] + z[3] * 0.75))
    arrow(ax, (s2[0] + s2[2], s2[1] + s2[3] * 0.75), (z[0], z[1] + z[3] * 0.25))
    arrow(ax, (s1[0] + s1[2], s1[1] + s1[3] * 0.85), L(sy), color="#9a988f")
    arrow(ax, R(sy), (m[0], m[1] + m[3] * 0.8), color="#9a988f")
    arrow(ax, (z[0] + z[2], z[1] + z[3] * 0.6), (m[0], m[1] + m[3] * 0.3))
    arrow(ax, (s2[0] + s2[2], s2[1] + 0.3), (f[0], f[1] + 0.3), label="raw p.u. signal (PQD-16c)")
    ax.text(0.1 + w / 2, 2.55, "test split used once,\nafter model selection", ha="center", va="center",
            fontsize=11.5, color=MUTED, style="italic")
    save(fig, "preprocess.png")


def architecture():
    fig, ax = canvas()
    w, h, g = 1.55, 1.15, 0.42
    row1 = [("Input", "1 x 100", "navy"), ("Conv stem", "k7, 64 ch\nBN, GELU", "blue"),
            ("Res block 1", "64 ch, k7", "blue"), ("MaxPool", "/2  (L 50)", "grey"),
            ("Res block 2", "64 to 128, k5", "blue"), ("MaxPool", "/2  (L 25)", "grey")]
    y1, y2 = 3.55, 1.85
    xs = [0.15 + i * (w + g) for i in range(6)]
    b1 = [box(ax, x, y1, w, h, t, s, st, fs=13, sfs=11) for x, (t, s, st) in zip(xs, row1)]
    for a, b in zip(b1, b1[1:]):
        arrow(ax, R(a), L(b))
    row2 = [("Res block 3", "128 ch, k5\ndilation 2", "blue"), ("Res block 4", "128 ch, k3\ndilation 4", "blue"),
            ("Feature map", "128 x 25", "teal"), ("Global avg\npool", "", "grey"),
            ("Linear", "128 to 17\n(dropout 0.3)", "orange"), ("Softmax", "class\nprobabilities", "navy")]
    b2 = [box(ax, x, y2, w, h, t, s, st, fs=13, sfs=11) for x, (t, s, st) in zip(xs[::-1], row2)]
    arrow(ax, B(b1[-1]), T(b2[0]))
    for a, b in zip(b2, b2[1:]):
        arrow(ax, L(a), R(b))
    cam = box(ax, xs[2] - 0.2, 0.1, 2 * w + g + 0.4, 1.15, "Class Activation Map",
              "CAM_c(t) = sum_k  w_ck x f_k(t)", "purple", fs=13, sfs=12)
    arrow(ax, B(b2[2]), (B(b2[2])[0], cam[1] + cam[3]))
    arrow(ax, B(b2[4]), (cam[0] + cam[2] * 0.2, cam[1] + cam[3]))
    ax.text(B(b2[2])[0] + 0.1, 1.5, "feature maps f_k(t)", fontsize=11, color=MUTED, style="italic", va="center")
    ax.text(B(b2[4])[0] - 0.1, 1.5, "class weights w_ck", fontsize=11, color=MUTED, style="italic", va="center",
            ha="right")
    ax.text(xs[0] + w, 0.7, "455 k parameters\nfully convolutional\nCAM resolution: 4 samples",
            ha="center", va="center", fontsize=11.5, color=MUTED, linespacing=1.4)
    rb = (9.0, 0.12, 3.0, 1.15)
    ax.add_patch(FancyBboxPatch(rb[:2], rb[2], rb[3], boxstyle="round,pad=0,rounding_size=0.1",
                                fc="white", ec="#b9b7ae", lw=1.2, ls="--"))
    ax.text(rb[0] + 0.12, rb[1] + rb[3] - 0.12, "Residual block", fontsize=11.5, weight="bold", va="top")
    ax.text(rb[0] + 0.12, rb[1] + 0.12, "x -> Conv, BN, GELU -> Conv, BN\n     (+ x via 1x1 conv) -> GELU",
            fontsize=10.5, va="bottom", family="DejaVu Sans Mono")
    save(fig, "architecture.png")


def xai_flow():
    fig, ax = canvas()
    w, h = 2.25, 1.15
    s = box(ax, 0.1, 1.95, 1.9, h, "Test signal", "1 cycle, 100 samples", "navy")
    m = box(ax, 2.55, 3.4, w, h, "ResNet-CAM", "class, confidence,\ntop-3", "orange")
    ft = box(ax, 2.55, 0.5, w, h, "Physics features", "48 features ->\nExtra-Trees", "purple")
    ig = box(ax, 5.35, 3.95, w, 0.95, "Integrated Gradients", "32 steps, clean-sine baseline", "blue", fs=12.5, sfs=10.5)
    cm = box(ax, 5.35, 2.75, w, 0.95, "CAM", "free, one forward pass", "blue", fs=12.5, sfs=10.5)
    sh = box(ax, 5.35, 0.6, w, 0.95, "Tree SHAP", "exact per-feature values", "purple", fs=12.5, sfs=10.5)
    fa = box(ax, 8.2, 1.7, w, h, "Faithfulness test", "delete top-k vs random\nsegments, compare drop", "teal")
    out = box(ax, 8.2, 3.6, w, h, "Figure + JSON", "waveform, IG and CAM,\nevidence window", "grey")
    llm = box(ax, 10.85, 3.6, 1.2, h, "LLM\nnarration", "", "grey", fs=11.5)
    gui = box(ax, 10.85, 2.15, 1.2, h, "Live\nQt GUI", "", "grey", fs=11.5)
    arrow(ax, (s[0] + s[2], s[1] + s[3] * 0.75), L(m))
    arrow(ax, (s[0] + s[2], s[1] + s[3] * 0.25), L(ft))
    arrow(ax, (m[0] + m[2], m[1] + m[3] * 0.8), L(ig))
    arrow(ax, (m[0] + m[2], m[1] + m[3] * 0.2), L(cm))
    arrow(ax, R(ft), L(sh))
    for b, yy in ((ig, 0.85), (cm, 0.55), (sh, 0.25)):
        arrow(ax, (b[0] + b[2], b[1] + b[3] * (0.25 if b is ig else 0.5)), (fa[0], fa[1] + fa[3] * yy))
    arrow(ax, (ig[0] + ig[2], ig[1] + ig[3] * 0.75), (out[0], ig[1] + ig[3] * 0.75))
    arrow(ax, R(out), L(llm))
    arrow(ax, (out[0] + out[2], out[1] + out[3] * 0.15), L(gui))
    ax.text(fa[0] + fa[2] / 2, fa[1] - 0.12, "is the explanation\nreal evidence?", ha="center", va="top",
            fontsize=11, color=MUTED, style="italic")
    save(fig, "xai_flow.png")


def model_comparison():
    rows, seen = defaultdict(list), set()
    for line in open(ROOT / "outputs/benchmark/results.jsonl"):
        r = json.loads(line)
        key = (r.get("dataset", "pqd16"), r["model"], r["seed"])
        if key in seen:  # later re-runs (smoke tests, repeats) don't replace the reported run
            continue
        seen.add(key)
        rows[key[:2]].append(r["test_acc"])
    label = {"resnet_cam": "ResNet-CAM (ours)", "cnn_complex": "CNN-Complex", "cnn_res_bilstm": "CNN-Res-BiLSTM",
             "inception_cam_mv": "InceptionTime-CAM", "feat_extra_trees": "Features + Extra-Trees",
             "feat_hist_gbm": "Features + Grad. Boost", "feat_random_forest": "Features + Random Forest",
             "resnet_cam_mv": "ResNet-CAM multiview", "feat_logreg": "Features + Log. Reg.",
             "cycle_transformer": "Cycle Transformer"}
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), gridspec_kw={"width_ratios": [1.0, 1.55]})
    for ax, ds, title, lo in ((axes[0], "xpqrs", "XPQRS: 17 classes, 3 400 test signals", 90),
                              (axes[1], "pqd16", "PQD-16c: 16 classes, 320 test signals", 75)):
        items = sorted(((np.mean(v) * 100, m) for (d, m), v in rows.items() if d == ds))
        acc, ms = zip(*items)
        ax.barh([label[m] for m in ms], acc, color=[NAVY if m == "resnet_cam" else LIGHT for m in ms],
                height=0.62 if len(ms) > 3 else 0.45)
        for i, a in enumerate(acc):
            ax.text(a + 0.25, i, f"{a:.1f}%", va="center", fontsize=12.5, zorder=5,
                    weight="bold" if ms[i] == "resnet_cam" else "normal",
                    bbox=dict(fc="white", ec="none", pad=0.6))
        ax.set_xlim(lo, 101)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("test accuracy (%)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=12.5)
        if ds == "pqd16":
            ax.axvline(87.5, color=ORANGE, lw=2, ls="--")
            ax.text(88.0, -0.45, "label-noise ceiling 87.5%", color=ORANGE, fontsize=12, va="center")
        else:
            for i, m in enumerate(ms):
                ax.text(90.3, i, "455 k params" if m == "resnet_cam" else "1.42 M params", fontsize=12,
                        va="center", color="white" if m == "resnet_cam" else INK)
            ax.set_xticks([90, 92, 94, 96, 98, 100])
    fig.tight_layout(w_pad=2.5)
    save(fig, "model_comparison.png")


def predictions(ds):
    X, y, raw, names, *_ = (load_xpqrs if ds == "xpqrs" else load_splits)()
    model = MODELS["resnet_cam"](len(names))
    model.load_state_dict(torch.load(ROOT / f"models/resnet_cam_{ds}.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X["test"]).unsqueeze(1)).argmax(1).numpy()
    return y["test"], p, names


CONFUSABLE = {"Interruption", "Interruption+Harmonics", "Sag", "Sag+Harmonics"}  # PQD-16c label-noise family


def confusion_and_f1(ds="xpqrs"):
    y, p, names = predictions(ds)
    short = [nice(n) for n in names]
    n = len(names)
    cm = confusion_matrix(y, p, normalize="true")
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0.175, 0.1, 0.4, 0.84))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), [str(k + 1) for k in range(n)], fontsize=10.5)
    ax.set_yticks(range(n), [f"{s}  {k + 1}" for k, s in enumerate(short)], fontsize=11)
    for i in range(n):
        for j in range(n):
            if cm[i, j] >= 0.02:
                ax.text(j, i, f"{cm[i, j]*100:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.6 else INK)
    ax.set_title("Confusion matrix (% of true class)", fontsize=13.5)
    ax.set_xlabel("predicted class (number as on the left)", fontsize=11.5)
    f1 = f1_score(y, p, average=None)
    o = np.argsort(f1)
    ax2 = fig.add_axes((0.765, 0.1, 0.205, 0.84))
    weak = [names[i] in CONFUSABLE if ds == "pqd16" else f1[i] < 0.97 for i in range(n)]
    ax2.barh([short[i] for i in o], f1[o], color=[ORANGE if weak[i] else NAVY for i in o], height=0.66)
    lo = 0.85 if ds == "xpqrs" else 0.0
    for k, i in enumerate(o):
        ax2.text(f1[i] - (1 - lo) * 0.02, k, f"{f1[i]:.3f}", va="center", ha="right", fontsize=10, color="white")
    ax2.set_xlim(lo, 1.0)
    ax2.set_xticks([0.85, 0.9, 0.95, 1.0] if ds == "xpqrs" else [0, 0.25, 0.5, 0.75, 1.0])
    ax2.tick_params(axis="y", labelsize=11)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Per-class F1", fontsize=13.5)
    save(fig, "confusion_f1.png" if ds == "xpqrs" else f"confusion_f1_{ds}.png")


def label_noise():
    _, y, raw, names, *_ = load_splits()
    sig = np.vstack([raw[s] for s in raw])
    lab = np.concatenate([y[s] for s in y])
    feats, fn = extract(sig)
    rms = feats[:, fn.index("rms_min")]
    pairs = [("Interruption", "Sag"), ("Interruption+Harmonics", "Sag+Harmonics")]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.2), sharey=True)
    bins = np.arange(0, 1.076, 0.05)
    for ax, (a, b) in zip(axes, pairs):
        for cls, col in ((a, ORANGE), (b, BLUE)):
            v = rms[lab == names.index(cls)]
            ax.hist(v, bins=bins, histtype="step", lw=2.6, color=col, label=f"{cls}  (median {np.median(v):.2f})")
        ax.axvspan(0, 0.1, color="#d9d7cf")
        ax.annotate("IEEE 1159 interruption\nzone (< 0.1 p.u.): no signals", xy=(0.05, 0.92),
                    xycoords=("data", "axes fraction"), xytext=(0.16, 0.92), textcoords=("data", "axes fraction"),
                    fontsize=11, va="center", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
        ax.set_xlabel("deepest 1-cycle RMS of the signal (p.u.)")
        ax.legend(fontsize=11.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(0, 1.08)
        ax.set_ylim(0, 15.5)
    axes[0].set_ylabel("signals")
    fig.tight_layout(w_pad=2)
    save(fig, "label_noise.png")


def faithfulness():
    d = json.loads((ROOT / "outputs/benchmark/faithfulness.json").read_text())
    pick = [("feat_extra_trees", "tree_shap", "Tree SHAP, physics features", "PQD-16c"),
            ("cnn_res_bilstm", "integrated_gradients", "IG, CNN-Res-BiLSTM", "PQD-16c"),
            ("cnn_complex", "integrated_gradients", "IG, CNN-Complex", "PQD-16c"),
            ("resnet_cam", "integrated_gradients", "IG, ResNet-CAM", "PQD-16c"),
            ("resnet_cam", "cam", "CAM, ResNet-CAM", "PQD-16c"),
            ("xpqrs_resnet_cam", "integrated_gradients", "IG, ResNet-CAM", "XPQRS"),
            ("xpqrs_resnet_cam", "cam", "CAM, ResNet-CAM", "XPQRS")]
    ks = ["10%", "20%"]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    yy = np.arange(len(pick))[::-1]
    cols = [BLUE, NAVY]
    for j, k in enumerate(ks):
        gaps = [d[m][meth][k]["drop_top"] - d[m][meth][k]["drop_random"] for m, meth, *_ in pick]
        ax.barh(yy + (0.5 - j) * 0.36, gaps, height=0.34, color=cols[j], label=f"delete {k} of the signal")
        for yv, gv in zip(yy + (0.5 - j) * 0.36, gaps):
            ax.text(gv + (0.006 if gv >= 0 else -0.006), yv, f"{gv:+.2f}", va="center",
                    ha="left" if gv >= 0 else "right", fontsize=11.5)
    ax.set_yticks(yy, [f"{lab}  [{ds}]" for *_, lab, ds in pick], fontsize=12.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-0.24, 0.45)
    ax.set_xlabel("faithfulness gap = confidence drop (top-attributed deleted) minus (random deleted)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout()
    save(fig, "faithfulness.png")


def explanations():
    X, y, raw, names, mean, std = load_xpqrs()
    model = MODELS["resnet_cam"](len(names))
    model.load_state_dict(torch.load(ROOT / "models/resnet_cam_xpqrs.pth", weights_only=True))
    model.eval()
    t = np.arange(100)
    base = torch.tensor(((np.sin(2 * np.pi * t / 100) - mean) / std).astype(np.float32)).view(1, 1, -1)
    fig = plt.figure(figsize=(FIG_W, FIG_H + 0.3))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.12)
    for k, cls in enumerate(("Sag", "Interruption", "Oscillatory_Transient", "Harmonics_with_Swell")):
        i = int(np.flatnonzero(y["test"] == names.index(cls))[0])
        x = torch.tensor(X["test"][i:i + 1]).unsqueeze(1)
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0].numpy()
        pred = int(prob.argmax())
        ig = integrated_gradients(model, x, torch.tensor([pred]), base)[0].detach().numpy()
        sub = gs[k // 2, k % 2].subgridspec(2, 1, height_ratios=[2.3, 1], hspace=0.08)
        a0, a1 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        ms = t * 0.2
        pos = np.clip(ig, 0, None)
        pos = pos / (pos.max() + 1e-9)
        for tt, v in zip(ms, pos):
            if v > 0.25:
                a0.axvspan(tt - 0.1, tt + 0.1, color=TEAL, alpha=0.4 * v, lw=0)
        a0.plot(ms, np.sin(2 * np.pi * t / 100), color="#b9b7ae", lw=1.3, ls="--")
        a0.plot(ms, raw["test"][i], color=NAVY, lw=2)
        a0.set_ylim(-1.9, 1.9)
        a0.set_xticks([])
        a0.set_ylabel("p.u.", fontsize=11)
        a0.tick_params(labelsize=10)
        verdict = "correct" if names[pred] == cls else f"predicted {nice(names[pred])}"
        a0.set_title(f"{nice(cls)}:  {verdict}, {prob[pred]:.1%} confidence", fontsize=12.5)
        a1.bar(ms, ig, width=0.2, color=np.where(ig >= 0, TEAL, ORANGE))
        a1.set_yticks([])
        a1.set_ylabel("IG", fontsize=11)
        a1.tick_params(labelsize=10)
        if k >= 2:
            a1.set_xlabel("time (ms)", fontsize=11)
        else:
            a1.set_xticklabels([])
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.1)
    save(fig, "explanations.png")


def gui_zoom(shot):
    """Scope and prediction panel of a 2x GUI screenshot, large enough to read on a slide."""
    im = Image.open(shot).convert("RGB")
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    a = fig.add_axes((0, 0, 8.45 / FIG_W, 1))
    a.imshow(im.crop((0, 0, 1870, 1105)))
    a.axis("off")
    b = fig.add_axes((8.6 / FIG_W, 2.3 / FIG_H, 3.5 / FIG_W, 2.7 / FIG_H))
    b.imshow(im.crop((1880, 15, 2550, 535)))
    b.axis("off")
    fig.text(8.65 / FIG_W, 2.05 / FIG_H, "How to read it", fontsize=13, weight="bold", va="top")
    fig.text(8.65 / FIG_W, 1.68 / FIG_H,
             "\u2022 Hold a button to inject a\n   disturbance into the live sine.\n"
             "\u2022 Orange shading: where the\n   model found its evidence (IG).\n"
             "\u2022 Tag above each cycle: the\n   prediction, \u2713 correct, \u2717 wrong.",
             fontsize=11.5, va="top", linespacing=1.4)
    save(fig, "gui.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", help="2x GUI screenshot to turn into the zoomed GUI view")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (waveforms, preprocess, architecture, xai_flow, model_comparison, confusion_and_f1,
               label_noise, faithfulness, explanations):
        fn()
    confusion_and_f1("pqd16")
    if a.gui:
        gui_zoom(a.gui)
