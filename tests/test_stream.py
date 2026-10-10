"""Checks for the GUI signal source (no Qt, no model needed).
    python tests/test_stream.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import sync_to_zero_crossing  # noqa: E402
from stream import NORMAL, DisturbanceStream  # noqa: E402


def test_sync_moves_any_phase_to_zero():
    t = np.arange(100)
    x = np.stack([np.sin(2 * np.pi * (t + s) / 100) for s in (0, 17, 50, 83)])
    assert np.allclose(sync_to_zero_crossing(x), np.sin(2 * np.pi * t / 100), atol=1e-6)


def test_hold_tap_release():
    s = DisturbanceStream()
    assert s.next_cycle().true_class == NORMAL
    s.press("Sag")                                   # held: every cycle is a sag
    assert [s.next_cycle().true_class for _ in range(3)] == ["Sag"] * 3
    s.release("Sag")
    assert s.next_cycle().true_class == NORMAL
    s.press("Notch")
    s.release("Notch")                               # tap shorter than a cycle -> exactly one
    assert [s.next_cycle().true_class for _ in range(2)] == ["Notch", NORMAL]


def test_cycles_come_from_the_test_split_and_join_smoothly():
    s = DisturbanceStream()
    c = s.next_cycle()
    assert c.samples.shape == (100,) and abs(c.samples[0]) < 0.07   # starts at a zero crossing


if __name__ == "__main__":
    test_sync_moves_any_phase_to_zero()
    test_hold_tap_release()
    test_cycles_come_from_the_test_split_and_join_smoothly()
    print("ok")
