"""Signal source and live classifier behind the GUI (no Qt in this module).

DisturbanceStream emits one 50 Hz cycle at a time. Normally the cycle is a
Pure_Sinusoidal window; while a disturbance is selected, cycles are drawn
from that class instead. Every cycle comes from the XPQRS *test* split (never
seen in training), zero-crossing synchronised so consecutive cycles join into
a continuous waveform.

LiveClassifier runs ResNet-CAM on each cycle and returns the class
probabilities plus an Integrated-Gradients and a CAM attribution.
"""
from dataclasses import dataclass

import numpy as np
import torch

from data import ROOT, load_xpqrs
from faithfulness import integrated_gradients
from models import MODELS

NORMAL = "Pure_Sinusoidal"
GUI_CKPT = "models/resnet_cam_xpqrs_sync.pth"


def pretty(name):
    """'Sag_with_Oscillatory_Transient' -> 'Sag + oscillatory transient'."""
    if name == NORMAL:
        return "Normal (pure sine)"
    return name.replace("_with_", " + ").replace("_", " ").capitalize()


@dataclass
class Cycle:
    samples: np.ndarray        # (L,) signal in p.u.
    true_class: str            # class the cycle was drawn from
    index: int                 # row in the class's test bank
    pred_class: str = ""
    confidence: float = 0.0
    top3: tuple = ()
    ig: np.ndarray = None      # (L,) Integrated Gradients for the predicted class
    cam: np.ndarray = None     # (L,) Class Activation Map for the predicted class


class DisturbanceStream:
    def __init__(self, split="test", seed=0):
        _, y, raw, names, self.mean, self.std = load_xpqrs(sync=True)
        self.class_names = names
        self.bank = {n: raw[split][y[split] == i] for i, n in enumerate(names)}
        self.disturbances = [n for n in names if n != NORMAL]
        self.cycle_len = raw[split].shape[1]
        self.rng = np.random.default_rng(seed)
        self.active = None       # class whose button is currently held
        self._tap = None         # pressed and released before a cycle was emitted

    def press(self, name):
        if name not in self.bank:
            raise KeyError(name)
        self.active = self._tap = name

    def release(self, name):
        if self.active == name:
            self.active = None

    def next_cycle(self):
        """Class held now wins; a tap shorter than one cycle still yields one."""
        name = self.active or self._tap or NORMAL
        self._tap = None
        i = int(self.rng.integers(len(self.bank[name])))
        return Cycle(self.bank[name][i].copy(), name, i)


class LiveClassifier:
    def __init__(self, class_names, mean, std, ckpt=GUI_CKPT, cycle_len=100):
        self.names, self.mean, self.std = class_names, mean, std
        self.model = MODELS["resnet_cam"](len(class_names))
        self.model.load_state_dict(torch.load(ROOT / ckpt, weights_only=True))
        self.model.eval()
        t = np.arange(cycle_len)
        sine = ((np.sin(2 * np.pi * t / cycle_len) - mean) / std).astype(np.float32)
        self.baseline = torch.tensor(sine).view(1, 1, -1)  # IG reference = healthy cycle

    def classify(self, cycle, ig_steps=16):
        x = torch.tensor(((cycle.samples - self.mean) / self.std).astype(np.float32)).view(1, 1, -1)
        with torch.no_grad():
            prob = torch.softmax(self.model(x), 1)[0].numpy()
            pred = int(prob.argmax())
            cycle.cam = self.model.cam(x)[0, pred].numpy()
        # IG costs ig_steps forward/backward passes; a clean cycle the model also
        # calls clean has nothing to explain, so skip it there (most of the stream)
        if self.names[pred] != NORMAL or cycle.true_class != NORMAL:
            cycle.ig = integrated_gradients(self.model, x, torch.tensor([pred]), self.baseline,
                                            steps=ig_steps)[0].detach().numpy()
        cycle.pred_class, cycle.confidence = self.names[pred], float(prob[pred])
        cycle.top3 = tuple((self.names[i], float(prob[i])) for i in prob.argsort()[::-1][:3])
        return cycle
