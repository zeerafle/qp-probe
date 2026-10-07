#!/usr/bin/env python3
"""Keyframe artefact: does it track fps/GOP, and can the bitstream remove it?

    keyframe.py extract --data ubfc-rppg --out DIR [--subjects 1,10] [--rate 800k]
    keyframe.py analyse --out DIR

Follow-up to the C1 pilot (Proposal/experiments/c1-pilot/results-2026-10-07),
where `-g 60` made POS lock onto ~59 bpm at every bitrate. Two questions:

1. Mechanism. Keyframes every G frames pump quality at fps/G Hz. If that is the
   cause, the spurious HR must sit on a harmonic k*fps/G, and move when G moves
   (G = 30, 60, 120; long GOP and x264 intra-refresh as references).
2. Receiver-side removal. The decoder knows which frames are keyframes and what
   QP each frame got, so it can regress that structure out of the RGB trace
   before POS, without touching the encoder. Methods, fixed before any run:
     none     - no correction
     notch    - notch filters at k*fps/P, P = keyframe period read from the
                frame types (bitstream-informed, but blind to amplitude)
     phase    - regress each channel on a Fourier basis of GOP phase (time since
                the last I-frame, K=6 harmonics); handles irregular GOPs
     qp       - regress each channel on the ROI-mean and frame-mean QP traces
     phase+qp - both regressor sets
   PRIMARY = phase (cleanest: needs only frame types).

Decision rule, fixed before the first run (7 October 2026):
  The fix WORKS for a GOP mode if, with the primary method,
    (a) median POS err <= long-GOP median err + 2 bpm, and
    (b) MAE falls by >= 50% of the gap (MAE_mode - MAE_long),
  on the pooled windows; per-subject values are reported alongside.
  It must also be SAFE: on long-GOP encodes (no pumping) the primary method
  may not raise median err by more than 0.5 bpm.
  Intra-refresh has no periodic I-frames, so phase/notch do not apply there;
  it is judged on `qp` alone and reported, not gated.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import extract as E   # noqa: E402  (rPPG, reference, QP helpers: one implementation)
import facebox        # noqa: E402

MODES = ["long", "g30", "g60", "g120", "intra-refresh"]
METHODS = ["none", "notch", "phase", "qp", "phase+qp"]
PRIMARY = "phase"
WIN_S, HOP_S = 10.0, 5.0
K_PHASE = 6


def encode(src, dst, rate, mode):
    # Same CBR/zerolatency/aq=1 recipe as the pilot; only the GOP differs.
    if mode == "long":
        g, params = "9999", "aq-mode=1"
    elif mode == "intra-refresh":
        g, params = "250", "aq-mode=1:intra-refresh=1"
    else:
        g, params = mode[1:], "aq-mode=1"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
         "-b:v", rate, "-minrate", rate, "-maxrate", rate, "-bufsize", rate,
         "-tune", "zerolatency", "-g", g, "-x264-params", params, str(dst)],
        check=True)


# ------------------------------------------------------------- extraction ----

def extract(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subs = E.find_subjects(a.data)
    if a.subjects:
        want = {int(s) for s in a.subjects.split(",")}
        subs = {k: v for k, v in subs.items() if k in want}
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    for sid, sdir in subs.items():
        npz = out / f"s{sid}.npz"
        if npz.exists():
            print(f"subject{sid}: done, skipping", flush=True)
            continue
        t0 = time.time()
        src = sdir / "vid.avi"
        w, h, fps = E.probe_video(src)
        box, note = facebox.face_box(str(src))
        if box is None:
            sys.exit(f"subject{sid}: face box failed - {note}")
        arrs = {"fps": fps, "box": np.array(box)}
        f = E.stream_features(src, box, w, h)
        arrs["rgb_source"] = f["rgb"]
        for mode in MODES:
            mp = tmp / f"s{sid}-{mode}.mp4"
            try:
                encode(src, mp, a.rate, mode)
                q = E.qp_summary(mp, box)
                f = E.stream_features(mp, box, w, h)
            finally:
                mp.unlink(missing_ok=True)
            arrs[f"rgb_{mode}"] = f["rgb"]
            arrs[f"isI_{mode}"] = (q.pict_type == "I").to_numpy()
            arrs[f"qroi_{mode}"] = q.roi_mean.to_numpy(float)
            arrs[f"qmean_{mode}"] = q.mean_qp.to_numpy(float)
            print(f"  s{sid} {mode:<14} I-frames {int(arrs[f'isI_{mode}'].sum()):3d}", flush=True)
        ref, usable = E.load_reference(sdir / "ground_truth.txt", fps, len(arrs["rgb_source"]))
        arrs["ref"], arrs["usable"] = ref, usable
        np.savez_compressed(npz, **arrs)
        print(f"subject{sid} done in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------- correction ----

def _bp(x, fps):
    # Regress in the band the estimator sees, so slow drift cannot soak up
    # the fit and leave the in-band pumping untouched.
    return E.bandpass(x, fps)


def keyframe_period(isI):
    idx = np.flatnonzero(isI)
    return float(np.median(np.diff(idx))) if len(idx) >= 3 else None


def phase_basis(isI):
    """Fourier basis of the position inside the current GOP. Tracks the
    actual I-frame positions, so scene-cut keyframes do not break it."""
    n = len(isI)
    idx = np.flatnonzero(isI)
    if len(idx) < 3:
        return None
    ph = np.zeros(n)
    bounds = list(idx) + [n + (n - idx[-1])]
    for s, e in zip(bounds[:-1], bounds[1:]):
        L = max(e - s, 1)
        seg = np.arange(s, min(e, n))
        ph[seg] = (seg - s) / L
    ph[:idx[0]] = np.nan
    cols = []
    for k in range(1, K_PHASE + 1):
        cols += [np.sin(2 * np.pi * k * ph), np.cos(2 * np.pi * k * ph)]
    X = np.column_stack(cols)
    return np.nan_to_num(X)


def regress_out(rgb, X, fps):
    """Per channel: relative trace (c / clip mean - 1), band-passed; subtract
    the least-squares fit on band-passed regressors; rebuild an RGB trace."""
    out = np.empty_like(rgb)
    Xb = np.column_stack([_bp(X[:, j], fps) for j in range(X.shape[1])])
    for c in range(3):
        m = rgb[:, c].mean()
        y = rgb[:, c] / m - 1
        yb = _bp(y, fps)
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        out[:, c] = m * (1 + y - Xb @ beta)
    return out


def notch(rgb, fps, period):
    f0 = fps / period
    out = rgb.copy()
    k = 1
    while k * f0 < min(fps / 2 - 0.1, E.BAND[1] + 0.5):
        if k * f0 > E.BAND[0] - 0.3:
            b, a_ = iirnotch(k * f0, Q=30, fs=fps)
            out = filtfilt(b, a_, out, axis=0)
        k += 1
    return out


def corrected(rgb, isI, qroi, qmean, fps, method):
    if method == "none":
        return rgb
    P = keyframe_period(isI)
    if method == "notch":
        return notch(rgb, fps, P) if P else None
    cols = []
    if "phase" in method:
        B = phase_basis(isI)
        if B is None:
            return None
        cols.append(B)
    if "qp" in method:
        q1 = np.nan_to_num(qroi - np.nanmean(qroi))
        q2 = np.nan_to_num(qmean - np.nanmean(qmean))
        cols.append(np.column_stack([q1, q2]))
    return regress_out(rgb, np.column_stack(cols), fps)


# --------------------------------------------------------------- analysis ----

def windows(bvp, ref, usable, fps):
    wn, hop = int(round(WIN_S * fps)), int(round(HOP_S * fps))
    for s in range(0, usable - wn + 1, hop):
        yield s, E.hr_bpm(bvp[s:s + wn], fps), E.hr_bpm(ref[s:s + wn], fps)


def analyse(a):
    out = pathlib.Path(a.out)
    rows = []
    for npz in sorted(out.glob("s*.npz")):
        d = np.load(npz)
        sid, fps, usable = int(npz.stem[1:]), float(d["fps"]), int(d["usable"])
        ref = E.bandpass(d["ref"], fps)
        conds = [("source", "none", d["rgb_source"])]
        for mode in MODES:
            for meth in METHODS:
                r = corrected(d[f"rgb_{mode}"], d[f"isI_{mode}"], d[f"qroi_{mode}"],
                              d[f"qmean_{mode}"], fps, meth)
                if r is not None:
                    conds.append((mode, meth, r))
        for mode, meth, rgb in conds:
            n = min(len(rgb), usable)
            bvp = E.pos(rgb[:n], fps)
            P = keyframe_period(d[f"isI_{mode}"]) if mode.startswith("g") else None
            for s, hp, hr in windows(bvp, ref, n, fps):
                harm = np.nan
                if P:
                    f0 = 60 * fps / P
                    harm = abs(hp - f0 * round(hp / f0))   # distance to nearest k*fps/P
                rows.append(dict(subject=sid, mode=mode, method=meth, win=s,
                                 hr_ref=hr, hr_pos=hp, err=abs(hp - hr), harm_dist=harm))
    df = pd.DataFrame(rows)
    df.to_csv(out / "windows.csv", index=False)

    L = [f"# Keyframe artefact test\n\n{df.subject.nunique()} subjects, POS, "
         f"{a_rate(out)} aq=1, 10 s windows / 5 s hop.\n"]
    L.append("## 1. Mechanism: does the POS peak sit on k*fps/G?\n")
    L.append("| mode | median err | share of windows within 1.5 bpm of a keyframe harmonic | same share if HR were the reference |")
    L.append("|---|---|---|---|")
    for mode in [m for m in MODES if m.startswith("g")]:
        x = df[(df["mode"] == mode) & (df.method == "none")]
        P = None
        sub = []
        for sid, g in x.groupby("subject"):
            dd = np.load(out / f"s{sid}.npz")
            P = keyframe_period(dd[f"isI_{mode}"])
            f0 = 60 * float(dd["fps"]) / P
            sub.append(np.abs(g.hr_ref - f0 * np.round(g.hr_ref / f0)) <= 1.5)
        chance = pd.concat(sub).mean()
        L.append(f"| {mode} | {x.err.median():.1f} | {100 * (x.harm_dist <= 1.5).mean():.0f}% | {100 * chance:.0f}% |")

    L.append("\n## 2. Removal: POS error by mode and method\n")
    L.append("| mode | method | median err | MAE | share > 5 bpm | per-subject median err |")
    L.append("|---|---|---|---|---|---|")
    src = df[df["mode"] == "source"]
    L.append(f"| source | - | {src.err.median():.1f} | {src.err.mean():.1f} | {100 * (src.err > 5).mean():.0f}% | "
             + ", ".join(f"{g.err.median():.1f}" for _, g in src.groupby("subject")) + " |")
    for mode in MODES:
        for meth in METHODS:
            x = df[(df["mode"] == mode) & (df.method == meth)]
            if len(x):
                L.append(f"| {mode} | {meth} | {x.err.median():.1f} | {x.err.mean():.1f} | "
                         f"{100 * (x.err > 5).mean():.0f}% | "
                         + ", ".join(f"{g.err.median():.1f}" for _, g in x.groupby("subject")) + " |")

    L.append(f"\n## 3. Verdict (primary method: {PRIMARY}; rule in keyframe.py docstring)\n")
    lg = df[(df["mode"] == "long") & (df.method == "none")]
    lp = df[(df["mode"] == "long") & (df.method == PRIMARY)]
    for mode in [m for m in MODES if m.startswith("g")]:
        x0 = df[(df["mode"] == mode) & (df.method == "none")]
        x1 = df[(df["mode"] == mode) & (df.method == PRIMARY)]
        a_ok = x1.err.median() <= lg.err.median() + 2
        gap = x0.err.mean() - lg.err.mean()
        b_ok = gap > 0 and (x0.err.mean() - x1.err.mean()) >= 0.5 * gap
        L.append(f"- {mode}: median {x0.err.median():.1f} -> {x1.err.median():.1f} (long {lg.err.median():.1f}); "
                 f"MAE {x0.err.mean():.1f} -> {x1.err.mean():.1f}, gap closed "
                 f"{100 * (x0.err.mean() - x1.err.mean()) / gap if gap > 0 else float('nan'):.0f}% "
                 f"-> **{'WORKS' if a_ok and b_ok else 'DOES NOT WORK'}**")
    if len(lp):
        safe = lp.err.median() <= lg.err.median() + 0.5
        L.append(f"- safety on long GOP: median {lg.err.median():.1f} -> {lp.err.median():.1f} -> "
                 f"**{'SAFE' if safe else 'UNSAFE'}**")
    else:
        L.append("- safety on long GOP: phase basis undefined (one I-frame) -> correction is a no-op there, SAFE by construction")
    ir0 = df[(df["mode"] == "intra-refresh") & (df.method == "none")]
    ir1 = df[(df["mode"] == "intra-refresh") & (df.method == "qp")]
    L.append(f"- intra-refresh (qp only, reported): median {ir0.err.median():.1f} -> {ir1.err.median():.1f}, "
             f"MAE {ir0.err.mean():.1f} -> {ir1.err.mean():.1f}")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def a_rate(out):
    p = out / "rate.json"
    return json.loads(p.read_text())["rate"] if p.exists() else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("extract")
    x.add_argument("--data", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--subjects")
    x.add_argument("--rate", default="800k")
    y = sp.add_parser("analyse")
    y.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / "rate.json").write_text(json.dumps({"rate": a.rate}))
        extract(a)
    else:
        analyse(a)


if __name__ == "__main__":
    main()
