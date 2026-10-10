"""Physics-based power-quality features (IEEE 1159 vocabulary) for an
interpretable classifier. Every feature has a plain-language meaning, so SHAP
on these reads like an engineer's report, which is ideal input for the LLM.

Input is the *raw* (un-normalised) signal: 1.0 = nominal peak, 64 samples/cycle.
"""
import numpy as np

CYCLE = 64
N_CYC = 10


def _rms(x, axis=-1):
    return np.sqrt((x ** 2).mean(axis))


def extract(raw):
    """raw: (N, 640) -> (features (N, F), names list)."""
    X = np.asarray(raw, dtype=np.float64)
    n = len(X)
    f = {}

    # --- magnitude (sag / swell / interruption): RMS in p.u. (nominal = 1.0)
    cyc = _rms(X.reshape(n, N_CYC, CYCLE)) * np.sqrt(2)
    half = _rms(X.reshape(n, 2 * N_CYC, CYCLE // 2)) * np.sqrt(2)
    c = np.cumsum(np.pad(X ** 2, ((0, 0), (1, 0))), 1)
    slide = np.sqrt((c[:, CYCLE:] - c[:, :-CYCLE]) / CYCLE) * np.sqrt(2)  # 1-cycle sliding RMS
    for i in range(N_CYC):
        f[f"rms_cycle_{i + 1}"] = cyc[:, i]
    f["rms_min"] = slide.min(1)
    f["rms_max"] = slide.max(1)
    f["rms_halfcycle_min"] = half.min(1)
    f["rms_halfcycle_max"] = half.max(1)
    f["rms_range"] = slide.max(1) - slide.min(1)
    f["cycles_below_0.9pu"] = (slide < 0.9).sum(1) / CYCLE
    f["cycles_below_0.5pu"] = (slide < 0.5).sum(1) / CYCLE
    f["cycles_below_0.1pu"] = (slide < 0.1).sum(1) / CYCLE
    f["cycles_above_1.1pu"] = (slide > 1.1).sum(1) / CYCLE
    f["peak_abs"] = np.abs(X).max(1)
    f["crest_factor"] = np.abs(X).max(1) / (_rms(X) + 1e-9)
    # depth of the deepest dip relative to the other cycles (sudden vs gradual)
    f["rms_step_max"] = np.abs(np.diff(cyc, axis=1)).max(1)
    # minimum instantaneous envelope over the window (interruption ~ 0)
    seg = np.abs(X).reshape(n, 2 * N_CYC, CYCLE // 2).max(2)
    f["halfcycle_peak_min"] = seg.min(1)
    f["halfcycle_peak_max"] = seg.max(1)

    # --- spectrum (harmonics, flicker): fundamental sits at bin 10 (10 cycles)
    spec = np.abs(np.fft.rfft(X, axis=1)) / (len(X[0]) / 2)
    fund = spec[:, N_CYC] + 1e-9
    f["fundamental_mag"] = fund
    for h in (2, 3, 5, 7, 9, 11):
        f[f"harmonic_{h}_ratio"] = spec[:, N_CYC * h] / fund
    harm = np.sqrt((spec[:, [N_CYC * h for h in range(2, 26)]] ** 2).sum(1))
    f["THD"] = harm / fund
    # side-band energy around the fundamental = amplitude modulation (flicker)
    f["sideband_ratio"] = np.sqrt((spec[:, [N_CYC - 2, N_CYC - 1, N_CYC + 1, N_CYC + 2]] ** 2).sum(1)) / fund
    f["rms_cycle_std"] = cyc.std(1)
    env_spec = np.abs(np.fft.rfft(cyc - cyc.mean(1, keepdims=True), axis=1))
    f["rms_modulation_peak"] = env_spec[:, 1:].max(1)
    hf_bins = np.arange(len(spec[0])) > N_CYC * 26
    f["high_freq_energy"] = np.sqrt((spec[:, hf_bins] ** 2).sum(1)) / fund

    # --- transients (spike, impulse, oscillatory, notch): residual after
    # removing a fitted fundamental sine per cycle
    t = np.arange(CYCLE)
    basis = np.stack([np.sin(2 * np.pi * t / CYCLE), np.cos(2 * np.pi * t / CYCLE)], 1)
    cyc_x = X.reshape(n, N_CYC, CYCLE)
    coef = cyc_x @ basis / (CYCLE / 2)
    resid = (cyc_x - coef @ basis.T).reshape(n, -1)
    d2 = np.abs(np.diff(X, 2, axis=1))
    f["residual_rms"] = _rms(resid)
    f["residual_peak"] = np.abs(resid).max(1)
    f["residual_peak_to_rms"] = f["residual_peak"] / (f["residual_rms"] + 1e-9)
    hp = np.diff(resid, axis=1)
    f["hf_residual_peak"] = np.abs(hp).max(1)
    f["hf_residual_rms"] = _rms(hp)
    f["hf_residual_kurtosis"] = ((hp - hp.mean(1, keepdims=True)) ** 4).mean(1) / (hp.var(1) ** 2 + 1e-12)
    thr = 3 * np.median(np.abs(hp), 1, keepdims=True) + 1e-3
    active = np.abs(hp) > np.maximum(thr, 0.25 * np.abs(hp).max(1, keepdims=True))
    f["transient_active_samples"] = active.sum(1)
    idx = np.arange(hp.shape[1])
    f["transient_spread_samples"] = np.where(active.any(1), np.where(active, idx, -1).max(1)
                                             - np.where(active, idx, idx[-1]).min(1), 0)
    # sign changes in the residual around the largest transient -> oscillation
    k = np.abs(hp).argmax(1)
    hp_pad = np.pad(hp, ((0, 0), (8, 24)))
    win = hp_pad[np.arange(n)[:, None], k[:, None] + np.arange(32)]  # 8 before .. 24 after peak
    big = np.abs(win) > np.maximum(0.1 * np.abs(hp).max(1, keepdims=True), 1e-3)
    f["transient_zero_crossings"] = ((np.diff(np.sign(win), axis=1) != 0) & big[:, 1:]).sum(1)
    f["second_diff_peak"] = d2.max(1)
    # notch: repeated short events per cycle -> count cycles with a residual blip
    cyc_res_peak = np.abs(resid.reshape(n, N_CYC, CYCLE)).max(2)
    f["cycles_with_residual_blip"] = (cyc_res_peak > 0.05).sum(1)
    f["residual_blip_regularity"] = cyc_res_peak.std(1) / (cyc_res_peak.mean(1) + 1e-9)

    names = list(f)
    return np.stack([f[k] for k in names], 1).astype(np.float32), names


if __name__ == "__main__":
    # self-check: a pure sine should look nominal
    s = np.sin(2 * np.pi * np.arange(640) / CYCLE)[None]
    F, names = extract(s)
    d = dict(zip(names, F[0]))
    assert abs(d["rms_min"] - 1) < 1e-3 and d["THD"] < 1e-3 and d["cycles_below_0.9pu"] == 0 and d["transient_zero_crossings"] == 0, d
    sag = s.copy(); sag[:, 192:384] *= 0.5
    d = dict(zip(names, extract(sag)[0][0]))
    assert abs(d["rms_min"] - 0.5) < 1e-3 and 3 <= d["cycles_below_0.9pu"] <= 4, d
    print(f"ok: {len(names)} features")
