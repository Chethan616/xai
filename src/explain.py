"""Classify one test signal with a trained ResNet-CAM and explain the decision.

Prints the top-3 classes and saves a figure with the waveform, the
Integrated-Gradients attribution and the Class Activation Map (CAM), plus a
JSON summary that can be handed to an LLM for a plain-language explanation.

    python src/explain.py --dataset xpqrs --sample 7
    python src/explain.py --dataset pqd16 --sample 120 --ckpt models/resnet_cam_pqd16.pth
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import ROOT, load_splits, load_xpqrs  # noqa: E402
from faithfulness import integrated_gradients  # noqa: E402
from models import MODELS  # noqa: E402

DATASETS = {  # loader, default checkpoint, samples per cycle
    "xpqrs": (load_xpqrs, "models/resnet_cam_xpqrs.pth", 100),
    "pqd16": (load_splits, "models/resnet_cam_pqd16.pth", 64),
}


def explain(dataset, sample, ckpt=None, out_dir=None):
    loader, default_ckpt, cycle = DATASETS[dataset]
    X, y, raw, names, mean, std = loader()
    if not 0 <= sample < len(X["test"]):
        raise SystemExit(f"--sample must be in [0, {len(X['test']) - 1}]")
    model = MODELS["resnet_cam"](len(names))
    model.load_state_dict(torch.load(ROOT / (ckpt or default_ckpt), weights_only=True))
    model.eval()

    x = torch.tensor(X["test"][sample:sample + 1]).unsqueeze(1)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0].numpy()
        pred = int(prob.argmax())
        cam = model.cam(x)[0, pred].numpy()
    t = np.arange(x.shape[-1])
    baseline = torch.tensor(((np.sin(2 * np.pi * t / cycle) - mean) / std).astype(np.float32)).view(1, 1, -1)
    ig = integrated_gradients(model, x, torch.tensor([pred]), baseline)[0].detach().numpy()

    top3 = [(names[i], round(float(prob[i]), 4)) for i in prob.argsort()[::-1][:3]]
    seg = len(t) // 20
    hot = np.abs(ig).reshape(-1, seg).sum(1).argmax() * seg  # most influential segment
    summary = {
        "dataset": dataset, "sample": sample, "true_class": names[int(y["test"][sample])],
        "predicted_class": names[pred], "confidence": round(float(prob[pred]), 4), "top3": top3,
        "most_influential_segment_samples": [int(hot), int(hot + seg)],
        "signal_peak_pu": round(float(np.abs(raw["test"][sample]).max()), 3),
    }

    out_dir = out_dir or ROOT / "outputs" / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{dataset}_test{sample:04d}"
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, raw["test"][sample], color="#1f4e79", lw=1.2)
    ax[0].axvspan(hot, hot + seg, color="#f4a261", alpha=0.35, label="most influential segment")
    ax[0].set_ylabel("signal (p.u.)")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"true: {summary['true_class']}  |  predicted: {names[pred]} ({prob[pred]:.1%})")
    ax[1].bar(t, ig, color=np.where(ig >= 0, "#2a9d8f", "#e76f51"), width=1.0)
    ax[1].set_ylabel("Integrated\nGradients")
    ax[2].fill_between(t, cam, color="#8e7cc3", alpha=0.6)
    ax[2].set_ylabel("CAM")
    ax[2].set_xlabel("sample index")
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)
    plt.close(fig)
    with open(f"{stem}.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary, f"{stem}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="xpqrs")
    ap.add_argument("--sample", type=int, default=0, help="index into the test split")
    ap.add_argument("--ckpt", help="checkpoint path relative to the repo root")
    a = ap.parse_args()
    summary, png = explain(a.dataset, a.sample, a.ckpt)
    print(json.dumps(summary, indent=1))
    print("figure:", png)


if __name__ == "__main__":
    main()
