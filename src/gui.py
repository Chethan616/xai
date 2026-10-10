"""Live power-quality disturbance explainer (Qt / PySide6).

A scrolling oscilloscope shows a healthy 50 Hz sine. Hold any disturbance
button and the waveform turns into that disturbance for as long as the button
is held (one cycle minimum); release it and the sine returns. Every cycle is a
real XPQRS test-set window, classified live by ResNet-CAM, with the model's
attribution (Integrated Gradients or CAM) drawn under the waveform.

    python src/gui.py                       # interactive
    python src/gui.py --speed 4             # slower playback
    python src/gui.py --demo docs/figures/gui.png   # scripted run + screenshot
"""
import argparse
import sys
import time
from collections import Counter, deque

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

from data import ROOT
from stream import GUI_CKPT, NORMAL, DisturbanceStream, LiveClassifier, pretty

VISIBLE_CYCLES = 8
FRAME_MS = 30
SINGLE = ["Sag", "Swell", "Interruption", "Harmonics", "Flicker", "Notch", "Transient", "Oscillatory_Transient"]

# every colour in the app comes from the active theme; nothing is left to the OS palette
THEMES = {
    "light": dict(window="#f4f4f1", base="#ffffff", text="#1f1f1e", muted="#6b6a64", grid="#e4e3de",
                  wave="#1c5cab", attr="#eb6834", inject="#eef1f8", ok="#1a7f37", bad="#c0392b",
                  button="#fafaf8", border="#c9c8c1", accent="#2a78d6"),
    "dark": dict(window="#202124", base="#17181a", text="#e8e8e6", muted="#a3a39d", grid="#34363b",
                 wave="#6ea8ff", attr="#ff8a4c", inject="#262c3a", ok="#3fb950", bad="#ff6b5e",
                 button="#2b2d31", border="#4a4c52", accent="#4c8fe6"),
}
T = dict(THEMES["light"])  # active theme, updated in place by apply_theme()


def apply_theme(app, name):
    """Set the Qt palette and the app stylesheet from THEMES[name]."""
    T.update(THEMES[name])
    pal = QPalette()
    for role, key in ((QPalette.Window, "window"), (QPalette.WindowText, "text"), (QPalette.Base, "base"),
                      (QPalette.AlternateBase, "window"), (QPalette.Text, "text"), (QPalette.Button, "button"),
                      (QPalette.ButtonText, "text"), (QPalette.Highlight, "accent"),
                      (QPalette.ToolTipBase, "base"), (QPalette.ToolTipText, "text"),
                      (QPalette.PlaceholderText, "muted"), (QPalette.Mid, "border"), (QPalette.Dark, "border")):
        pal.setColor(role, QColor(T[key]))
    pal.setColor(QPalette.HighlightedText, QColor("white"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(T["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QPushButton[dist="true"] {{ padding:8px 6px; border:1px solid {T['border']}; border-radius:6px;
                                    background:{T['button']}; color:{T['text']}; }}
        QPushButton[dist="true"]:hover {{ border-color:{T['accent']}; }}
        QPushButton[dist="true"]:pressed {{ background:{T['accent']}; color:white; border-color:{T['accent']}; }}
        QLabel#caption {{ color:{T['muted']}; font-size:10px; letter-spacing:1px; }}
        QLabel#why, QLabel#stats {{ color:{T['text']}; }}
        QLabel#stats {{ font-size:11px; }}
        QLabel#holding {{ color:{T['attr']}; font-weight:600; }}
    """)


def short(name):
    """Compact label for the per-cycle prediction tag on the scope."""
    words = {"Pure_Sinusoidal": "Normal", "Oscillatory_Transient": "OscT", "Interruption": "Intr",
             "Harmonics": "H", "Flicker": "Flk", "Transient": "Trans"}
    for k, v in words.items():
        name = name.replace(k, v)
    return name.replace("_with_", "+")


class Scope(QWidget):
    """Scrolling oscilloscope. Cycles are revealed sample by sample; each
    finished cycle gets a prediction tag (green ✓ / red ✗) and an attribution
    heat strip."""

    def __init__(self, cycle_len):
        super().__init__()
        self.cycle_len = cycle_len
        self.cycles = deque(maxlen=VISIBLE_CYCLES + 1)  # [Cycle, revealed_samples]
        self.attribution = "ig"
        self.setMinimumSize(640, 300)

    def push(self, cycle):
        self.cycles.append([cycle, 0])

    def reveal(self, n):
        """Reveal up to n samples of the newest cycle; returns samples consumed."""
        head = self.cycles[-1]
        take = min(n, self.cycle_len - head[1])
        head[1] += take
        return take

    def head_done(self):
        return not self.cycles or self.cycles[-1][1] >= self.cycle_len

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(T["base"]))
        plot = QRectF(52, 30, self.width() - 64, self.height() - 62)
        ymax = 2.0
        to_y = lambda v: plot.center().y() - max(-ymax, min(ymax, v)) / ymax * plot.height() / 2  # noqa: E731
        span = VISIBLE_CYCLES * self.cycle_len
        filled = sum(c[1] for c in self.cycles)
        offset = max(0, filled - span)                    # samples scrolled off the left edge
        px = plot.width() / span
        to_x = lambda s: plot.left() + (s - offset) * px  # noqa: E731

        # axes: 0 and ±1 p.u.
        p.setPen(QPen(QColor(T["grid"]), 1))
        for v in (-1.5, -1, -0.5, 0.5, 1, 1.5):
            p.drawLine(QPointF(plot.left(), to_y(v)), QPointF(plot.right(), to_y(v)))
        p.setPen(QPen(QColor(T["muted"]), 1))
        p.drawLine(QPointF(plot.left(), to_y(0)), QPointF(plot.right(), to_y(0)))
        p.setFont(QFont(p.font().family(), 8))
        for v in (-2, -1, 0, 1, 2):
            p.drawText(QRectF(0, to_y(v) - 8, 46, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:+d}" if v else "0")
        p.save()
        p.translate(12, plot.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-40, -10, 80, 20), Qt.AlignCenter, "voltage (p.u.)")
        p.restore()
        p.drawText(QRectF(plot.left(), plot.bottom() + 8, plot.width(), 18), Qt.AlignCenter,
                   "time →  (each band = one 50 Hz cycle, 20 ms)")

        p.setClipRect(plot)
        start, path = 0, QPainterPath()
        for cycle, shown in self.cycles:
            x0, x1 = to_x(start), to_x(start + self.cycle_len)
            if cycle.true_class != NORMAL:                # injected disturbance band
                p.fillRect(QRectF(x0, plot.top(), x1 - x0, plot.height()), QColor(T["inject"]))
            attr = {"ig": cycle.ig, "cam": cycle.cam}.get(self.attribution)
            if attr is not None and shown == self.cycle_len and cycle.true_class != NORMAL:
                a = np.convolve(np.abs(attr), np.ones(5) / 5, mode="same")  # smooth 1 ms
                a = a / (a.max() + 1e-9)
                for i, w in enumerate(a):                 # attribution heat strip
                    c = QColor(T["attr"])
                    c.setAlphaF(float(w) ** 2 * 0.6)
                    p.fillRect(QRectF(to_x(start + i), plot.top(), px + 0.6, plot.height()), c)
            p.setPen(QPen(QColor(T["grid"]), 1, Qt.DashLine))
            p.drawLine(QPointF(x1, plot.top()), QPointF(x1, plot.bottom()))
            for i in range(shown):
                pt = QPointF(to_x(start + i), to_y(float(cycle.samples[i])))
                path.lineTo(pt) if path.elementCount() else path.moveTo(pt)
            if shown == self.cycle_len and cycle.pred_class:   # prediction tag
                good = cycle.pred_class == cycle.true_class
                p.setPen(QColor(T["ok" if good else "bad"]))
                p.setFont(QFont(p.font().family(), 8, QFont.Bold))
                p.drawText(QRectF(x0 + 2, plot.top() + 2, x1 - x0 - 4, 16), Qt.AlignCenter,
                           f"{'✓' if good else '✗'} {short(cycle.pred_class)}")
            start += self.cycle_len
        p.setPen(QPen(QColor(T["wave"]), 2))
        p.drawPath(path)
        p.setClipping(False)
        p.setPen(QColor(T["muted"]))
        p.setFont(QFont(p.font().family(), 8))
        p.drawText(QRectF(plot.left(), 6, plot.width(), 18), Qt.AlignLeft,
                   "grey band = injected disturbance   ·   orange = where the model looked   ·   "
                   "tag = model prediction (✓ correct, ✗ wrong)")


class PredictionPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(330)
        lay = QVBoxLayout(self)
        caption = QLabel("MODEL PREDICTION  (last full cycle)")
        caption.setObjectName("caption")
        self.pred = QLabel("—")
        self.pred.setStyleSheet("font-size:20px; font-weight:600;")
        self.pred.setWordWrap(True)
        self.truth = QLabel("")
        self.truth.setWordWrap(True)
        lay.addWidget(caption)
        lay.addWidget(self.pred)
        lay.addWidget(self.truth)
        lay.addSpacing(6)
        self.last = None
        self.bars = []
        for _ in range(3):
            name, bar = QLabel(""), QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            name.setStyleSheet("font-size:11px;")
            lay.addWidget(name)
            lay.addWidget(bar)
            self.bars.append((name, bar))
        lay.addSpacing(6)
        self.why = QLabel("")
        self.why.setWordWrap(True)
        self.why.setObjectName("why")
        lay.addWidget(self.why)
        lay.addStretch(1)
        self.stats = QLabel("")
        self.stats.setObjectName("stats")
        lay.addWidget(self.stats)

    def show_cycle(self, c, cycle_ms=20.0):
        self.last = c
        good = c.pred_class == c.true_class
        color = T["ok" if good else "bad"]
        self.pred.setText(f"{pretty(c.pred_class)}  <span style='color:{color}'>{'✓' if good else '✗'}</span>")
        self.truth.setText(f"confidence {c.confidence:.1%}<br>actually injected: <b>{pretty(c.true_class)}</b>")
        for (name, bar), (cls, prob) in zip(self.bars, c.top3):
            name.setText(f"{pretty(cls)} — {prob:.1%}")
            bar.setValue(int(prob * 1000))
        if c.pred_class == NORMAL:
            self.why.setText("<b>Why:</b> the cycle matches a clean 1 p.u. sine — "
                             "no disturbance evidence found.")
            return
        seg = len(c.samples) // 10
        hot = int(np.abs(c.ig).reshape(-1, seg).sum(1).argmax()) * seg
        ms = cycle_ms / len(c.samples)
        self.why.setText(f"<b>Why:</b> the strongest evidence is in samples {hot}–{hot + seg} "
                         f"({hot * ms:.1f}–{(hot + seg) * ms:.1f} ms into the cycle), "
                         f"peak |v| = {np.abs(c.samples).max():.2f} p.u.")


class MainWindow(QMainWindow):
    def __init__(self, ckpt, speed, theme):
        super().__init__()
        self.theme = theme
        self.setWindowTitle("PQD Explainer — live disturbance injection")
        self.stream = DisturbanceStream()
        self.clf = LiveClassifier(self.stream.class_names, self.stream.mean, self.stream.std, ckpt,
                                  self.stream.cycle_len)
        self.scope = Scope(self.stream.cycle_len)
        self.panel = PredictionPanel()
        self.counts = {"dist": [0, 0], "normal": [0, 0]}  # [correct, total]
        self.run = None                                   # current run of same-class cycles
        self.held_since = {}
        self.budget = 0.0

        root = QWidget()
        lay = QVBoxLayout(root)
        top = QHBoxLayout()
        top.addWidget(self.scope, 1)
        top.addWidget(self.panel)
        lay.addLayout(top, 1)
        lay.addLayout(self._controls(speed))
        lay.addLayout(self._buttons())
        self.log = QListWidget()
        self.log.setFixedHeight(96)
        self.log.addItem("Hold a disturbance button to inject it; release to return to a clean sine.")
        lay.addWidget(self.log)
        self.setCentralWidget(root)
        self.resize(1280, 820)

        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self, interval=FRAME_MS, timeout=self.tick)
        self.timer.start()

    # ---------------------------------------------------------------- layout
    def _controls(self, speed):
        row = QHBoxLayout()
        self.pause = QPushButton("Pause")
        self.pause.setCheckable(True)
        self.pause.toggled.connect(lambda on: (self.timer.stop() if on else self.timer.start(),
                                               self.pause.setText("Resume" if on else "Pause")))
        self.speed = QSlider(Qt.Horizontal, minimum=1, maximum=25, value=speed)
        self.speed.setFixedWidth(180)
        self.speed_label = QLabel()
        self.speed.valueChanged.connect(self._speed_text)
        self._speed_text(speed)
        self.attr = QComboBox()
        self.attr.addItems(["Integrated Gradients", "Class Activation Map", "No explanation"])
        self.attr.currentIndexChanged.connect(lambda i: setattr(self.scope, "attribution", ("ig", "cam", None)[i]))
        self.holding = QLabel("")
        self.holding.setObjectName("holding")
        self.theme_btn = QPushButton()
        self.theme_btn.clicked.connect(lambda: self.set_theme("light" if self.theme == "dark" else "dark"))
        row.addWidget(self.theme_btn)
        self._theme_text()
        for w in (self.pause, QLabel("Speed"), self.speed, self.speed_label, QLabel("   Explanation"), self.attr):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(self.holding)
        return row

    def _theme_text(self):
        self.theme_btn.setText("☀ Light mode" if self.theme == "dark" else "☾ Dark mode")

    def set_theme(self, name):
        self.theme = name
        apply_theme(QApplication.instance(), name)
        QSettings("pqd-explainer", "gui").setValue("theme", name)
        self._theme_text()
        if self.panel.last:
            self.panel.show_cycle(self.panel.last)   # re-colour the ✓ / ✗
        self.scope.update()

    def _speed_text(self, v):
        self.speed_label.setText(f"{v} cycles/s  (×{v / 50:.2f} real time)")

    def _buttons(self):
        row = QHBoxLayout()
        combined = [n for n in self.stream.disturbances if n not in SINGLE]
        for title, names in (("Single disturbances", SINGLE), ("Combined disturbances", combined)):
            box, grid = QGroupBox(f"{title}  —  press and hold"), QGridLayout()
            for i, name in enumerate(names):
                b = QPushButton(pretty(name))
                b.setProperty("dist", True)
                b.setAutoRepeat(False)
                b.pressed.connect(lambda n=name: self._press(n))
                b.released.connect(lambda n=name: self._release(n))
                grid.addWidget(b, i // 4, i % 4)
            box.setLayout(grid)
            row.addWidget(box)
        return row

    # ------------------------------------------------------------ behaviour
    def _press(self, name):
        self.stream.press(name)
        self.held_since[name] = time.monotonic()
        self.holding.setText(f"● injecting {pretty(name)}")

    def _release(self, name):
        self.stream.release(name)
        held = time.monotonic() - self.held_since.pop(name, time.monotonic())
        if self.run and self.run["cls"] == name:
            self.run["held"] = held
        self.holding.setText("")

    def tick(self):
        dt = self.clock.restart() / 1000
        self.budget += min(dt, 0.2) * self.speed.value() * self.stream.cycle_len
        n = int(self.budget)
        self.budget -= n
        while n > 0:
            if self.scope.head_done():
                self._emit_cycle()
            n -= self.scope.reveal(n)
            if self.scope.head_done():                    # a cycle just finished
                self.panel.show_cycle(self.scope.cycles[-1][0])
        if self.stream.active:
            held = time.monotonic() - self.held_since.get(self.stream.active, time.monotonic())
            self.holding.setText(f"● injecting {pretty(self.stream.active)} — held {held:.1f} s")
        self.scope.update()

    def _emit_cycle(self):
        c = self.clf.classify(self.stream.next_cycle())
        key = "normal" if c.true_class == NORMAL else "dist"
        self.counts[key][0] += c.pred_class == c.true_class
        self.counts[key][1] += 1
        if self.run is None or self.run["cls"] != c.true_class:
            self._close_run()
            self.run = {"cls": c.true_class, "preds": Counter(), "held": None}
        self.run["preds"][c.pred_class] += 1
        self.scope.push(c)
        d, nrm = self.counts["dist"], self.counts["normal"]
        self.panel.stats.setText(
            f"Disturbed cycles correct: {d[0]}/{d[1]}" + (f" ({d[0] / d[1]:.0%})" if d[1] else "")
            + f"<br>Normal cycles correct: {nrm[0]}/{nrm[1]}" + (f" ({nrm[0] / nrm[1]:.0%})" if nrm[1] else ""))

    def _close_run(self):
        r = self.run
        if not r or r["cls"] == NORMAL:
            return
        n = sum(r["preds"].values())
        held = f"held {r['held']:.1f} s · " if r["held"] is not None else ""
        verdict = ", ".join(f"{pretty(k)} ×{v}" for k, v in r["preds"].most_common())
        self.log.insertItem(0, f"{time.strftime('%H:%M:%S')}  {pretty(r['cls'])}: {held}{n} cycle"
                               f"{'s' if n > 1 else ''} → model said {verdict}")


def run_demo(win, app, out):
    """Scripted presses for a reproducible screenshot (docs/figures/gui.png)."""
    script = [(400, "press", "Sag"), (1400, "release", "Sag"),
              (2300, "press", "Harmonics_with_Swell"), (3100, "release", "Harmonics_with_Swell"),
              (3900, "press", "Oscillatory_Transient"), (4500, "release", "Oscillatory_Transient"),
              (5000, "press", "Sag_with_Harmonics")]         # still held when the screenshot is taken
    for ms, action, name in script:
        QTimer.singleShot(ms, lambda a=action, n=name: win._press(n) if a == "press" else win._release(n))

    def shoot():
        win.timer.stop()
        win.grab().save(str(out))
        print("screenshot:", out)
        app.quit()
    QTimer.singleShot(6400, shoot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=GUI_CKPT, help="checkpoint relative to the repo root")
    ap.add_argument("--speed", type=int, default=6, help="playback speed in cycles per second (1-25)")
    ap.add_argument("--demo", metavar="PNG", help="run a scripted demo, save a screenshot and exit")
    ap.add_argument("--theme", choices=list(THEMES), help="colour theme (default: last used, else the OS setting)")
    a = ap.parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    if not (ROOT / a.ckpt).exists():
        QMessageBox.critical(None, "Checkpoint missing",
                             f"{a.ckpt} not found.\nTrain it with:\n\npython src/train.py --dataset xpqrs_sync "
                             "--models resnet_cam --seeds 0 --epochs 40 --patience 15\n\nthen copy "
                             "outputs/benchmark/ckpt/xpqrs_sync_resnet_cam_s0.pth to models/resnet_cam_xpqrs_sync.pth")
        return 1
    os_dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    theme = a.theme or QSettings("pqd-explainer", "gui").value("theme") or ("dark" if os_dark else "light")
    theme = theme if theme in THEMES else "light"   # ignore a corrupted settings value
    apply_theme(app, theme)
    win = MainWindow(a.ckpt, max(1, min(25, a.speed)), theme)
    win.show()
    if a.demo:
        run_demo(win, app, a.demo)
    code = app.exec()
    # explicit teardown: let Qt destroy the widget tree before Python's GC runs
    win.timer.stop()
    win.deleteLater()
    del win
    app.processEvents()
    app.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
