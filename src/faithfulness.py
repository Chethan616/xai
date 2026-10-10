"""Explanation faithfulness (deletion test) for trained models.

For each test sample: rank inputs by attribution for the predicted class,
replace the top-k% with the *nominal* value (a clean 1 p.u. sine for signal
models, Normal-class median for feature models), and measure the drop in the
predicted-class probability. Faithfulness = drop(top-k) / drop(random-k):
>1 means the explanation points at evidence the model really uses.

    python src/faithfulness.py resnet_cam_mv:outputs/benchmark/ckpt/resnet_cam_mv_s0.pth ...
    python src/faithfulness.py --dataset xpqrs resnet_cam:outputs/benchmark/ckpt/xpqrs_resnet_cam_s0.pth
"""
import json
import sys

import numpy as np
import torch

from data import ROOT, load_splits, load_xpqrs
from models import MODELS

FRACTIONS = (0.05, 0.10, 0.20)


def integrated_gradients(model, x, target, baseline, steps=32):
    alphas = torch.linspace(1 / steps, 1, steps).view(-1, 1, 1)
    grads = torch.zeros_like(x)
    for i in range(len(x)):
        path = (baseline + alphas * (x[i:i + 1] - baseline)).requires_grad_(True)
        out = model(path)[:, target[i]].sum()
        grads[i] = torch.autograd.grad(out, path)[0].mean(0)
    return ((x - baseline) * grads).squeeze(1)


def attributions(model, name, x, target, baseline):
    out = {"integrated_gradients": integrated_gradients(model, x, target, baseline)}
    if hasattr(model, "cam"):
        with torch.no_grad():
            out["cam"] = model.cam(x)[torch.arange(len(x)), target]
    if name == "cnn_res_bilstm":
        with torch.no_grad():
            w = model(x, return_attention=True)[1]  # (B, 80) over the pooled sequence
        out["attention"] = torch.repeat_interleave(w, x.shape[-1] // w.shape[1], dim=1)
    return out


def deletion(prob_fn, x, target, scores, nominal, rng, seg=1):
    """x: (N, D) numpy, scores: (N, D) higher = more important.
    seg > 1 deletes contiguous segments (ranked by summed attribution) for both
    the explanation and the random baseline, so neither is penalised for the
    extra discontinuities that scattered single-sample edits create."""
    base = prob_fn(x)[np.arange(len(x)), target]
    res = {}
    for frac in FRACTIONS:
        nseg = x.shape[1] // seg
        k = max(1, int(round(frac * nseg)))
        top = np.argsort(-scores.reshape(len(x), nseg, seg).sum(2), 1)[:, :k]
        rnd = np.stack([rng.choice(nseg, k, replace=False) for _ in x])
        top, rnd = ((i[:, :, None] * seg + np.arange(seg)).reshape(len(x), -1) for i in (top, rnd))
        drops = []
        for idx in (top, rnd):
            xd = x.copy()
            np.put_along_axis(xd, idx, np.take_along_axis(np.broadcast_to(nominal, x.shape), idx, 1), 1)
            drops.append(float((base - prob_fn(xd)[np.arange(len(x)), target]).mean()))
        res[f"{int(frac * 100)}%"] = {"drop_top": round(drops[0], 4), "drop_random": round(drops[1], 4),
                                     "ratio": round(drops[0] / max(drops[1], 1e-4), 2)}
    return res


def evaluate_signal_model(name, ckpt, X, y, mean, std, n=160, cycle=64):
    model = MODELS[name](int(y["train"].max()) + 1)
    model.load_state_dict(torch.load(ROOT / ckpt, weights_only=True))
    model.eval()
    x = torch.tensor(X["test"][:n]).unsqueeze(1)
    nominal = ((np.sin(2 * np.pi * np.arange(X["test"].shape[1]) / cycle) - mean) / std).astype(np.float32)
    with torch.no_grad():
        target = model(x).argmax(1)

    def prob_fn(a):
        with torch.no_grad():
            return torch.softmax(model(torch.tensor(a).unsqueeze(1)), 1).numpy()

    rng = np.random.default_rng(0)
    out = {}
    for method, a in attributions(model, name, x, target, torch.tensor(nominal).view(1, 1, -1)).items():
        out[method] = deletion(prob_fn, x.squeeze(1).numpy(), target.numpy(), a.detach().numpy(), nominal, rng,
                               seg=x.shape[-1] // 20)
    return out


def evaluate_feature_model(raw, y, n=160, seed=0):
    import shap
    from sklearn.ensemble import ExtraTreesClassifier

    from features import extract

    F = {s: extract(raw[s])[0] for s in raw}
    clf = ExtraTreesClassifier(800, random_state=seed, n_jobs=2).fit(F["train"], y["train"])
    x = F["test"][:n]
    target = clf.predict(x)
    explainer = shap.TreeExplainer(clf)
    # chunked: one shot over 800 deep trees x 16 classes OOMs a 6 GB machine
    sv = np.concatenate([np.asarray(explainer.shap_values(x[i:i + 16])) for i in range(0, n, 16)])  # (n, F, C)
    scores = sv[np.arange(n), :, target]
    nominal = np.median(F["train"][y["train"] == list(clf.classes_).index(8)], 0)  # class 8 = Normal
    return {"tree_shap": deletion(clf.predict_proba, x, target, scores, nominal, np.random.default_rng(0))}


def main():
    args = sys.argv[1:]
    xpqrs = args[:2] == ["--dataset", "xpqrs"]
    if xpqrs:  # 1 cycle = 100 samples; nominal = clean sine, 340 test samples (20/class)
        args = args[2:]
        X, y, raw, names, mean, std = load_xpqrs()
    else:
        X, y, raw, names, mean, std = load_splits()
        assert names[8] == "Normal"
    results = {}
    for spec in args:
        if spec == "feat_extra_trees":
            assert not xpqrs, "physics features assume 10-cycle PQD-16c signals"
            name, results[spec] = spec, evaluate_feature_model(raw, y)
        else:
            name, ckpt = spec.split(":")
            if xpqrs:
                name = f"xpqrs_{name}"
                results[name] = evaluate_signal_model(spec.split(":")[0], ckpt, X, y, mean, std, n=340, cycle=100)
            else:
                results[name] = evaluate_signal_model(name, ckpt, X, y, mean, std)
        print(name, json.dumps(results[name]), flush=True)
    path = ROOT / "outputs" / "benchmark" / "faithfulness.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **results}, indent=1))


if __name__ == "__main__":
    main()
