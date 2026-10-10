"""Regenerate every figure in docs/figures/ from the saved results and the two
final checkpoints.   python src/make_figures.py"""
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from data import ROOT, load_splits, load_xpqrs  # noqa: E402
from explain import explain  # noqa: E402
from features import extract  # noqa: E402
from models import MODELS  # noqa: E402

OUT = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
CONFUSABLE = {"Interruption", "Interruption+Harmonics", "Sag", "Sag+Harmonics"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def predictions(dataset):
    X, y, raw, names, *_ = (load_xpqrs if dataset == "xpqrs" else load_splits)()
    model = MODELS["resnet_cam"](len(names))
    model.load_state_dict(torch.load(ROOT / f"models/resnet_cam_{dataset}.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X["test"]).unsqueeze(1)).argmax(1).numpy()
    return y["test"], pred, names


def model_comparison():
    rows = defaultdict(list)
    for line in open(ROOT / "outputs/benchmark/results.jsonl"):
        r = json.loads(line)
        rows[(r.get("dataset", "pqd16"), r["model"])].append(r["test_acc"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1.2]})
    for ax, ds, title in ((axes[0], "pqd16", "PQD-16c (16 classes, 320 test signals)"),
                          (axes[1], "xpqrs", "XPQRS (17 classes, 3400 test signals)")):
        items = sorted(((np.mean(v) * 100, m) for (d, m), v in rows.items() if d == ds))
        acc, names = zip(*items)
        colors = [BLUE if m == "resnet_cam" else "#9ec5f4" for m in names]
        ax.barh(names, acc, color=colors, height=0.6 if len(names) > 3 else 0.35)
        for i, a in enumerate(acc):
            ax.text(a + 0.3, i, f"{a:.1f}%", va="center", fontsize=9, color=INK)
        lo = 75 if ds == "pqd16" else 95
        ax.set_xlim(lo, 100)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("test accuracy (%)")
        ax.grid(axis="y", visible=False)
        if ds == "pqd16":
            ax.axvline(87.5, color=ORANGE, lw=1.5, ls="--")
            ax.text(87.7, len(names) - 0.6, "label-noise ceiling ≈ 87.5%", color=ORANGE, fontsize=8)
    fig.suptitle("Model comparison — ResNet-CAM (dark blue) is the selected model", x=0.01, ha="left")
    fig.savefig(OUT / "model_comparison.png")
    plt.close(fig)


def per_class_f1(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, ds, title in ((axes[0], "pqd16", "PQD-16c"), (axes[1], "xpqrs", "XPQRS")):
        y, p, names = results[ds]
        f1 = f1_score(y, p, average=None)
        order = np.argsort(f1)
        colors = [ORANGE if ds == "pqd16" and names[i] in CONFUSABLE else BLUE for i in order]
        ax.barh([names[i] for i in order], f1[order], color=colors, height=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{title} — ResNet-CAM per-class F1", fontsize=10, loc="left")
        ax.set_xlabel("F1 score")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
               labels=["Sag / Interruption family (indistinguishable in PQD-16c)", "other classes"],
               loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "per_class_f1.png")
    plt.close(fig)


def confusion(results, ds):
    y, p, names = results[ds]
    cm = confusion_matrix(y, p, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.6 else INK)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    label = "XPQRS" if ds == "xpqrs" else "PQD-16c"
    ax.set_title(f"{label} — ResNet-CAM confusion matrix (row-normalised)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="fraction of true class")
    fig.savefig(OUT / f"confusion_{ds}.png")
    plt.close(fig)


def faithfulness():
    d = json.loads((ROOT / "outputs/benchmark/faithfulness.json").read_text())
    nice = {"integrated_gradients": "Integrated Gradients", "cam": "CAM",
            "tree_shap": "Tree SHAP (physics features)", "attention": "attention"}
    rows = []
    for model, methods in d.items():
        if model.startswith("inception"):
            continue
        ds = "XPQRS" if model.startswith("xpqrs_") else "PQD-16c"
        for method, r in methods.items():
            gap = r["10%"]["drop_top"] - r["10%"]["drop_random"]
            rows.append((gap, f"{model.removeprefix('xpqrs_')} · {nice[method]}  [{ds}]", ds))
    rows.sort()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(rows)), [r[0] for r in rows], tick_label=[r[1] for r in rows],
            color=[BLUE if r[2] == "PQD-16c" else ORANGE for r in rows], height=0.6)
    for i, r in enumerate(rows):
        ax.text(max(r[0], 0) + 0.005, i, f"{r[0]:+.2f}", va="center", fontsize=8)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("confidence drop: delete top-10% attributed − delete random 10%  (higher = more faithful)")
    ax.set_title("Explanation faithfulness (deletion test)", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["PQD-16c (10-cycle signals)", "XPQRS (1-cycle signals)"], loc="lower right",
              fontsize=8, frameon=False)
    fig.savefig(OUT / "faithfulness.png")
    plt.close(fig)


def label_noise():
    _, y, raw, names, *_ = load_splits()
    sig = np.vstack([raw[s] for s in raw])
    lab = np.concatenate([y[s] for s in y])
    feats, fn = extract(sig)
    rms_min = feats[:, fn.index("rms_min")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    bins = np.arange(0, 1.076, 0.025)  # an edge sits exactly on 0.1 p.u.
    for ax, (a, b) in zip(axes, (("Interruption", "Sag"), ("Interruption+Harmonics", "Sag+Harmonics"))):
        for cls, color in ((a, ORANGE), (b, BLUE)):
            ax.hist(rms_min[lab == names.index(cls)], bins=bins, histtype="step", lw=2, color=color, label=cls)
        ax.axvspan(0, 0.1, color=GRID, alpha=0.8)
        ax.text(0.012, ax.get_ylim()[1] * 0.85, "IEEE 1159\ninterruption\n< 0.1 p.u.", fontsize=7.5, color=MUTED)
        ax.set_xlabel("minimum 1-cycle RMS (p.u.)")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("signals")
    fig.suptitle("PQD-16c: 'Interruption' signals never drop below 0.1 p.u. — they have sag depth",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "pqd16_label_noise.png")
    plt.close(fig)


def explanation_gallery(results):
    y, _, names = results["xpqrs"]
    for cls in ("Sag", "Interruption", "Oscillatory_Transient", "Harmonics_with_Swell"):
        idx = int(np.flatnonzero(y == names.index(cls))[0])
        explain("xpqrs", idx, out_dir=OUT / "explanations")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {ds: predictions(ds) for ds in ("pqd16", "xpqrs")}
    model_comparison()
    per_class_f1(results)
    confusion(results, "xpqrs")
    confusion(results, "pqd16")
    faithfulness()
    label_noise()
    explanation_gallery(results)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
