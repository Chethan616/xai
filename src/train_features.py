"""Feature-based (interpretable) classifiers on features.py, same splits/seeds
as train.py. Appends to outputs/benchmark/results.jsonl."""
import json
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import extract
from data import ROOT, load_splits

MODELS = {
    "feat_logreg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=3, max_iter=5000)),
    "feat_random_forest": lambda s: RandomForestClassifier(500, random_state=s, n_jobs=4),
    "feat_extra_trees": lambda s: ExtraTreesClassifier(800, random_state=s, n_jobs=4),
    "feat_hist_gbm": lambda s: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400,
                                                              early_stopping=False, random_state=s),
}


def main():
    _, y, raw, *_ = load_splits()
    F = {s: extract(raw[s])[0] for s in raw}
    for name, make in MODELS.items():
        for seed in (0, 1, 2):
            t = time.time()
            clf = make(seed).fit(F["train"], y["train"])
            pv, pt = clf.predict(F["validation"]), clf.predict(F["test"])
            r = {"model": name, "seed": seed, "val_acc": float((pv == y["validation"]).mean()),
                 "test_acc": float((pt == y["test"]).mean()),
                 "test_macro_f1": float(f1_score(y["test"], pt, average="macro")),
                 "per_class_f1": f1_score(y["test"], pt, average=None).round(3).tolist(),
                 "params": F["train"].shape[1], "minutes": round((time.time() - t) / 60, 2)}
            print(json.dumps({k: v for k, v in r.items() if k != "per_class_f1"}), flush=True)
            with open(ROOT / "outputs" / "benchmark" / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            if name == "feat_logreg":
                break  # deterministic


if __name__ == "__main__":
    main()
