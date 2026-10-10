"""Sanity check for the segment deletion test: a toy model that only looks at
samples 40-44 must lose far more confidence when the attributed segment is
deleted than when random segments are.   python tests/test_faithfulness.py"""
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faithfulness import deletion  # noqa: E402


def test_segment_deletion_rewards_true_evidence():
    def prob(a):
        p0 = np.tanh(np.abs(a[:, 40:45]).sum(1))
        return np.stack([p0, 1 - p0], 1)

    x, scores = np.ones((50, 100)), np.zeros((50, 100))
    scores[:, 40:45] = 1
    r = deletion(prob, x, np.zeros(50, int), scores, np.zeros(100), np.random.default_rng(0), seg=5)
    assert r["5%"]["ratio"] > 5, r


if __name__ == "__main__":
    test_segment_deletion_rewards_true_evidence()
    print("ok")
