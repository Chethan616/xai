"""Train every model in models.MODELS with one shared protocol over several
seeds and record val/test accuracy, macro-F1 and per-class F1.

    python src/train.py --models resnet_cam inception_cam --seeds 0 1 2
Results append to outputs/benchmark/results.jsonl; best checkpoint per run is
kept in outputs/benchmark/ckpt/.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from data import ROOT, load_splits, load_xpqrs
from models import MODELS

DATASETS = {"pqd16": load_splits, "xpqrs": load_xpqrs,
            "xpqrs_sync": lambda: load_xpqrs(sync=True)}  # zero-crossing-synchronised (GUI model)

OUT = ROOT / "outputs" / "benchmark"


def run(name, seed, X, y, epochs, lr, patience, augment, dataset="pqd16"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[name](int(y["train"].max()) + 1)
    t = {s: torch.tensor(X[s]).unsqueeze(1) for s in X}
    yt = {s: torch.tensor(y[s]) for s in y}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * -(-len(X["train"]) // 64), pct_start=0.15)
    crit = nn.CrossEntropyLoss()

    def evaluate(s):
        model.eval()
        with torch.no_grad():
            logits = model(t[s])
        return crit(logits, yt[s]).item(), logits.argmax(1).numpy()

    best, bad, ckpt = (-1, 0), 0, OUT / "ckpt" / (f"{name}_s{seed}.pth" if dataset == "pqd16" else f"{dataset}_{name}_s{seed}.pth")
    start = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(t["train"]))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb = t["train"][b]
            if augment:  # label-preserving: whole-cycle circular shift + tiny noise
                shift = int(np.random.randint(10)) * 64
                xb = torch.roll(xb, shift, -1) + 0.01 * torch.randn_like(xb)
            loss = crit(model(xb), yt["train"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vloss, vpred = evaluate("validation")
        vacc = (vpred == y["validation"]).mean()
        if (vacc, -vloss) > best:
            best, bad = (vacc, -vloss), 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"[{name} s{seed}] ep {ep + 1}/{epochs} loss {loss.item():.3f} "
              f"vloss {vloss:.3f} vacc {vacc:.3f} best {best[0]:.3f} bad {bad}/{patience}",
              file=sys.stderr, flush=True)
        if bad >= patience:
            break
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    _, pred = evaluate("test")
    return {
        "model": name, "dataset": dataset, "seed": seed, "augment": augment, "epochs_run": ep + 1,
        "val_acc": float(best[0]), "test_acc": float((pred == y["test"]).mean()),
        "test_macro_f1": float(f1_score(y["test"], pred, average="macro")),
        "per_class_f1": f1_score(y["test"], pred, average=None).round(3).tolist(),
        "params": sum(p.numel() for p in model.parameters()),
        "minutes": round((time.time() - start) / 60, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dataset", choices=list(DATASETS), default="pqd16")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
    X, y, *_ = DATASETS[a.dataset]()
    for name in a.models:
        for seed in a.seeds:
            r = run(name, seed, X, y, a.epochs, a.lr, a.patience, a.augment, a.dataset)
            print(json.dumps({k: v for k, v in r.items() if k != "per_class_f1"}), flush=True)
            with open(OUT / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
