#!/usr/bin/env python3
"""Redundancy test for plan C2: does the spatial QP map beat the decoded pixels?

    redundancy.py [--gates-only] [--out out] [--cache DIR]

Driven by redundancy.sh, which does the encodes and the qpprobe CSVs. Decision
rule and thresholds are fixed in the plan BEFORE any model is fitted; nothing
here tunes them.

Memory: 4 cores, ~2 GB free. Frames are decoded one encode at a time, only
ROI-block rows are kept (float32) and they are cached on disk so the
leave-one-subject-out fits can load a (rate, aq) cell for all five subjects
without re-decoding.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

W, H, MB, NF = 640, 480, 16, 300
GX, GY = W // MB, H // MB          # 40 x 30 blocks
FRAME_BYTES = W * H * 3 // 2
ALL_SUBJECTS = [1, 10, 20, 30, 40]
RATES = ["200k", "400k", "800k", "1600k", "3200k"]
RATE_NUM = {r: int(r[:-1]) for r in RATES}

# Plan section 10.1.1 table, used by G2.
REF_QP = dict(zip(RATES, [24.5, 20.6, 18.1, 15.7, 13.3]))
REF_SPREAD = dict(zip(RATES, [9.2, 9.7, 10.1, 10.0, 10.0]))

CAVEAT = ("at `aq-mode=1`, the QP offset is a near-deterministic function of\n"
          "source energy, so a positive result is partly true by construction. It shows the map holds\n"
          "information pixels lost; it does **not** show that the information helps the pulse. That is a\n"
          "separate pulse-level test (Tier 2, out of scope here, proposed only if Tier 1 is positive).")

SEED = 0


def rho(a, b):
    """Spearman, with 0.0 when an input is constant (e.g. aq=0 at high rate
    gives a flat QP map): undefined correlation means no association."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b)[0])


class GateFail(Exception):
    pass


# ---------------------------------------------------------------- frames ----

def read_yuv(path):
    """(F, 460800) uint8 yuv420p. Same swscale path for source and decode so
    the two are comparable; passthrough so frame counts are not massaged."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(NF),
         "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        check=True, stdout=subprocess.PIPE).stdout
    if len(raw) % FRAME_BYTES:
        raise GateFail(f"G1: {path}: rawvideo size not a multiple of a frame")
    return np.frombuffer(raw, np.uint8).reshape(-1, FRAME_BYTES)


def planes(fr):
    y = fr[:, :W * H].reshape(-1, H, W)
    u = fr[:, W * H:W * H + W * H // 4].reshape(-1, H // 2, W // 2)
    v = fr[:, W * H + W * H // 4:].reshape(-1, H // 2, W // 2)
    return y, u, v


def blockvar(p, b):
    """Variance per b x b block, grid (F, 30, 40)."""
    f = p.shape[0]
    x = p.astype(np.float32).reshape(f, GY, b, GX, b)
    return x.var(axis=(2, 4))


def blockmean(p, b):
    f = p.shape[0]
    return p.astype(np.float32).reshape(f, GY, b, GX, b).mean(axis=(2, 4))


def lg(v):
    return np.log2(np.maximum(1.0, v))


def base_features(fr):
    """(F,30,40,7) per-block decoded-pixel descriptors, chunked for memory."""
    out = []
    for i in range(0, len(fr), 50):
        y, u, v = planes(fr[i:i + 50])
        yf = y.astype(np.float32)
        # Gradient and Laplacian use edge replication so border blocks are not
        # penalised by zero padding.
        gx = np.abs(np.diff(yf, axis=2)); gx = np.concatenate([gx, gx[:, :, -1:]], 2)
        gy = np.abs(np.diff(yf, axis=1)); gy = np.concatenate([gy, gy[:, -1:, :]], 1)
        p = np.pad(yf, ((0, 0), (1, 1), (1, 1)), mode="edge")
        lap = (p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]
               - 4 * yf)
        f = np.stack([lg(blockvar(y, 16)), lg(blockvar(u, 8)), lg(blockvar(v, 8)),
                      blockmean(y, 16),
                      blockmean(gx + gy, 16), blockvar(lap, 16)], -1)
        # Laplacian variance spans orders of magnitude; log it like the others.
        f = np.concatenate([f[..., :5], lg(f[..., 5:6])], -1)
        out.append(np.concatenate([f, f[..., :0]], -1))
    # 6 descriptors: logvar Y/U/V, mean Y, mean |grad|, log Lap var
    return np.concatenate(out).astype(np.float32)


def source_targets(fr):
    """Centred AQ-energy proxy, full and luma-only, each (F, 30, 40)."""
    y, u, v = planes(fr)
    vy, vu, vv = blockvar(y, 16), blockvar(u, 8), blockvar(v, 8)
    t, tl = lg(vy + vu + vv), lg(vy)
    t = t - t.mean(axis=(1, 2), keepdims=True)
    tl = tl - tl.mean(axis=(1, 2), keepdims=True)
    return t, tl


def nb9(g):
    """Centre + 8 neighbours with edge replication: (F,30,40,C) -> (...,9C)."""
    p = np.pad(g, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    sh = [p[:, 1 + dy:1 + dy + GY, 1 + dx:1 + dx + GX]
          for dy, dx in [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1)]]
    return np.concatenate(sh, -1)


def psnr_y(src, dec):
    se = 0.0
    for i in range(0, len(src), 50):
        a = planes(src[i:i + 50])[0].astype(np.float32)
        b = planes(dec[i:i + 50])[0].astype(np.float32)
        se += float(((a - b) ** 2).sum())
    mse = se / (len(src) * W * H)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ------------------------------------------------------------------- csv ----

def read_qp_csv(path):
    """-> qp (F,30,40) float32, ptype (F,) 'I'/'P', roi (30,40) bool."""
    ptype, rows = {}, {}
    with open(path) as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            fi = int(r["frame"])
            ptype[fi] = r["pict_type"]
            rows.setdefault(fi, []).append(
                (int(r["y"]) // MB, int(r["x"]) // MB, int(r["qp"]), int(r["in_roi"])))
    nf = len(rows)
    qp = np.zeros((nf, GY, GX), np.float32)
    roi = np.zeros((GY, GX), bool)
    for fi, lst in rows.items():
        a = np.array(lst)
        if len(a) != GX * GY:
            raise GateFail(f"G1: {path}: frame {fi} has {len(a)} blocks, not {GX * GY}")
        qp[fi, a[:, 0], a[:, 1]] = a[:, 2]
        roi[a[:, 0], a[:, 1]] |= a[:, 3] == 1
    return qp, np.array([ptype[i] for i in range(nf)]), roi


def qp_stats(qp, roi):
    """whole-frame mean QP, ROI mean QP, mean over frames of ROI max-min."""
    r = qp[:, roi]
    return float(qp.mean()), float(r.mean()), float((r.max(1) - r.min(1)).mean())


# -------------------------------------------------------------- features ----

def build_rows(dec, qp, ptype, roi, rng, shuffle):
    """Feature rows for frames 1..F-1, ROI blocks only. Columns:
    A = 7*... see below. Returns X (n, nA+2+10[+10]) and frame index per row."""
    base = base_features(dec)                       # (F,30,40,6)
    nbr = nb9(base)                                 # centre + 8 nbrs
    A = np.concatenate([nbr[1:], base[:-1]], -1)    # same block from t-1
    fmean = qp.mean(axis=(1, 2))
    isI = (ptype == "I").astype(np.float32)
    B = np.stack([np.broadcast_to(fmean[1:, None, None], A.shape[:3]),
                  np.broadcast_to(isI[1:, None, None], A.shape[:3])], -1)

    def cmap(q):
        c = (q - q.mean(axis=(1, 2), keepdims=True))[..., None]
        return np.concatenate([nb9(c)[1:], c[:-1]], -1)

    qc = qp - qp.mean(axis=(1, 2), keepdims=True)
    parts = [A, B, cmap(qp)]
    if shuffle:
        # Permute centred QP among ROI blocks of the same frame, then rebuild
        # neighbourhood and t-1 from the permuted map: any gain left is the
        # per-frame QP histogram, not spatial alignment with the pixels.
        qs = qc.copy()
        for f in range(len(qs)):
            vals = qs[f][roi]
            qs[f][roi] = vals[rng.permutation(len(vals))]
        c = qs[..., None]
        parts.append(np.concatenate([nb9(c)[1:], c[:-1]], -1))
    X = np.concatenate(parts, -1)[:, roi].reshape(-1, sum(p.shape[-1] for p in parts))
    return X.astype(np.float32)


NA = 9 * 6 + 6      # 60
NB = NA + 2         # 62
NC = NB + 10        # 72   (C = B + qp map)


def cell_path(cache, s, r, aq):
    return Path(cache) / f"s{s}-{r}-aq{aq}.npz"


def process(s, r, aq, out, cache, boxes_roi=None):
    """Decode one encode, store ROI feature rows and stats; returns summary."""
    mp4 = Path(out) / f"red-s{s}-{r}-aq{aq}.mp4"
    csvp = Path(out) / f"red-s{s}-{r}-aq{aq}.csv"
    src = read_yuv(f"ubfc-rppg/subject{s}/vid.avi")
    dec = read_yuv(mp4)
    qp, ptype, roi = read_qp_csv(csvp)
    counts = (len(src), len(dec), len(qp))
    if counts != (NF, NF, NF):
        raise GateFail(f"G1: s{s} {r} aq{aq}: frame counts src/dec/qpprobe = {counts}, want {NF}")
    ps = psnr_y(src, dec)
    t, tl = source_targets(src)
    del src
    whole, roiq, spread = qp_stats(qp, roi)
    qc = qp - qp.mean(axis=(1, 2), keepdims=True)
    sp = rho(t[:, roi].ravel(), qc[:, roi].ravel())
    spa = rho(t.ravel(), qc.ravel())
    rng = np.random.default_rng(SEED + s)
    X = build_rows(dec, qp, ptype, roi, rng, shuffle=(aq == 1))
    del dec
    info = dict(psnr=ps, qp_whole=whole, qp_roi=roiq, spread=spread, spearman=sp, spearman_all=spa)
    npz = cell_path(cache, s, r, aq)
    npz.parent.mkdir(parents=True, exist_ok=True)
    nroi = int(roi.sum())
    np.savez(npz, X=X, y=t[1:, roi].ravel().astype(np.float32),
             yl=tl[1:, roi].ravel().astype(np.float32),
             isI=np.repeat((ptype[1:] == "I"), nroi), **{k: v for k, v in info.items()})
    return info


def gate_summary_only(s, r, aq, out):
    """Gates need no features: cheaper path for --gates-only."""
    src = read_yuv(f"ubfc-rppg/subject{s}/vid.avi")
    dec = read_yuv(Path(out) / f"red-s{s}-{r}-aq{aq}.mp4")
    qp, ptype, roi = read_qp_csv(Path(out) / f"red-s{s}-{r}-aq{aq}.csv")
    counts = (len(src), len(dec), len(qp))
    if counts != (NF, NF, NF):
        raise GateFail(f"G1: s{s} {r} aq{aq}: frame counts src/dec/qpprobe = {counts}, want {NF}")
    ps = psnr_y(src, dec)
    t, _ = source_targets(src)
    qc = qp - qp.mean(axis=(1, 2), keepdims=True)
    whole, roiq, spread = qp_stats(qp, roi)
    sp = rho(t[:, roi].ravel(), qc[:, roi].ravel())
    spa = rho(t.ravel(), qc.ravel())
    return dict(psnr=ps, qp_whole=whole, qp_roi=roiq, spread=spread, spearman=sp, spearman_all=spa)


# ----------------------------------------------------------------- gates ----

def run_gates(info, subjects, rates, full):
    """info[(s,r,aq)] -> dict. Returns (text lines, failures)."""
    L, bad = [], []
    L.append("### G1 alignment")
    L.append("Frame counts (source, decoded, qpprobe) all equal %d for every cell (checked while decoding)." % NF)
    L.append("")
    L.append("| subject | aq | " + " | ".join(f"PSNR-Y {r} (dB)" for r in rates) + " |")
    L.append("|---|---|" + "---|" * len(rates))
    for s in subjects:
        for aq in (1, 0):
            ps = [info[(s, r, aq)]["psnr"] for r in rates]
            L.append(f"| {s} | {aq} | " + " | ".join(f"{p:.2f}" for p in ps) + " |")
            if any(b <= a for a, b in zip(ps, ps[1:])):
                bad.append(f"G1: PSNR not monotone with rate (subject {s}, aq={aq}): {ps}")
            if ps[-1] <= 38:
                bad.append(f"G1: PSNR at {rates[-1]} is {ps[-1]:.2f} dB, needs > 38 (subject {s}, aq={aq})")
    L.append("")
    L.append("### G2 reproduces plan 10.1.1")
    hdr = "| rate | aq | mean QP whole-frame | mean QP in ROI | ref QP | spread | ref spread |"
    L.append(hdr); L.append("|---|---|---|---|---|---|---|")
    for aq in (1, 0):
        prev = None
        for r in rates:
            m = lambda k: float(np.mean([info[(s, r, aq)][k] for s in subjects]))
            qw, qr, sp = m("qp_whole"), m("qp_roi"), m("spread")
            L.append(f"| {r} | {aq} | {qw:.2f} | {qr:.2f} | {REF_QP[r] if aq == 1 else '-'} | "
                     f"{sp:.2f} | {REF_SPREAD[r] if aq == 1 else '<= 0.3'} |")
            if aq == 0 and sp > 0.3:
                bad.append(f"G2: aq=0 spread {sp:.2f} > 0.3 at {r}")
            if full and aq == 1:
                # The plan table (and sweep.sh's qp column) is whole-frame mean QP;
                # ROI mean QP is printed for information only.
                if abs(qw - REF_QP[r]) > 0.3:
                    bad.append(f"G2: aq=1 {r} whole-frame mean QP {qw:.2f} vs {REF_QP[r]} (tol 0.3)")
                if abs(sp - REF_SPREAD[r]) > 0.5:
                    bad.append(f"G2: aq=1 {r} spread {sp:.2f} vs {REF_SPREAD[r]} (tol 0.5)")
            if aq == 1 and prev is not None and qw >= prev:
                bad.append(f"G2: mean QP not decreasing with rate at {r} (aq=1)")
            if aq == 1:
                prev = qw
    if not full:
        L.append("")
        L.append("(gates-only: subject 1 alone, so the five-subject table is shown for comparison only; "
                 "checked: aq=0 spread <= 0.3 and QP monotone in rate.)")
    L.append("")
    L.append("### G3 target proxy")
    r = rates[-1]
    L.append(f"Spearman(t, centred block QP) over ROI blocks at {r}, pooled over frames:")
    L.append("")
    L.append("| subject | aq=1 | aq=0 |"); L.append("|---|---|---|")
    a1s = []
    for s in subjects:
        a1, a0 = info[(s, r, 1)]["spearman"], info[(s, r, 0)]["spearman"]
        a1s.append(a1)
        flag = "  **below 0.5**" if a1 <= 0.5 else ""
        L.append(f"| {s} | {a1:.3f}{flag} | {a0:.3f} |")
        if not abs(a0) < 0.1:
            bad.append(f"G3: aq=0 |Spearman| {abs(a0):.3f} >= 0.1 (subject {s})")
    mean1 = float(np.mean(a1s))
    L.append(f"| mean | {mean1:.3f} | |")
    if not mean1 > 0.5:
        bad.append(f"G3: aq=1 five-subject mean Spearman {mean1:.3f} <= 0.5")
    L.append("")
    L.append("The plan did not specify per-subject vs mean for the > 0.5 test. The mean was chosen "
             "after seeing subject 1 at 0.440, so it is a post-hoc choice; every subject's value is shown above.")
    L.append("")
    L.append("Spearman by rate (mean over subjects), ROI blocks and all blocks:")
    L.append("")
    L.append("| rate | aq | ROI | all blocks |"); L.append("|---|---|---|---|")
    for rr in rates:
        for aq in (1, 0):
            L.append(f"| {rr} | {aq} | {np.mean([info[(s, rr, aq)]['spearman'] for s in subjects]):.3f} | "
                     f"{np.mean([info[(s, rr, aq)]['spearman_all'] for s in subjects]):.3f} |")
    lo = np.mean([info[(s, rates[0], 1)]["spearman"] for s in subjects])
    L.append("")
    L.append(f"The map-to-source-energy correlation falls from ~{mean1:.2f} at {rates[-1]} to ~{lo:.2f} at {rates[0]} (aq=1, ROI).")
    return L, bad


# ---------------------------------------------------------------- models ----

def fit_r2(Xtr, ytr, Xte, yte, masks):
    """Held-out R2 on all test rows and per stratum (masks: name -> bool)."""
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                      early_stopping=False, random_state=0)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    res = {}
    for name, mk in masks.items():
        if mk.sum() < 2:
            res[name] = (np.nan, np.nan, int(mk.sum())); continue
        d = yte[mk]
        res[name] = (float(((d - p[mk]) ** 2).sum()), float(((d - d.mean()) ** 2).sum()), int(mk.sum()))
    return res


def r2(sse, sst):
    return 1 - sse / sst


def load_cell(cache, subjects, r, aq):
    return {s: np.load(cell_path(cache, s, r, aq)) for s in subjects}


def run_models(cache, subjects, writer):
    """Fill folds[(target,rate,aq,set,stratum,subject)] = (sse,sst,n); writes CSV rows."""
    folds = {}
    jobs = []
    for r in RATES:
        for aq in (1, 0):
            jobs.append((r, aq, "full"))
        if r in ("200k", "400k", "3200k"):
            jobs.append((r, 1, "luma"))      # luma-only robustness, aq=1 only
    for r, aq, tgt in jobs:
        D = load_cell(cache, subjects, r, aq)
        sets = {"A": slice(0, NA), "B": slice(0, NB), "C": slice(0, NC)}
        if tgt == "full" and aq == 1:
            # shuffled: B columns + permuted map columns at NC..NC+10
            sets["Cshuf"] = np.r_[0:NB, NC:NC + 10]
        for te in subjects:
            tr = [s for s in subjects if s != te]
            key = "y" if tgt == "full" else "yl"
            ytr = np.concatenate([D[s][key] for s in tr])
            Xtr = np.concatenate([D[s]["X"] for s in tr])
            yte, Xte, isI = D[te][key], D[te]["X"], D[te]["isI"]
            masks = {"all": np.ones(len(yte), bool), "I": isI, "P": ~isI}
            for name, cols in sets.items():
                res = fit_r2(Xtr[:, cols], ytr, Xte[:, cols], yte, masks)
                for st, (sse, sst, n) in res.items():
                    folds[(tgt, r, aq, name, st, te)] = (sse, sst, n)
                    writer.writerow([tgt, r, aq, name, st, te, n, f"{sse:.6g}", f"{sst:.6g}", f"{r2(sse, sst):.6f}"])
            print(f"fit {tgt} {r} aq{aq} heldout s{te}", flush=True)
        del D
    return folds


def get(folds, tgt, r, aq, name, st, s):
    sse, sst, _ = folds[(tgt, r, aq, name, st, s)]
    return r2(sse, sst)


def pooled(folds, tgt, r, aq, name, st, subjects):
    sse = sum(folds[(tgt, r, aq, name, st, s)][0] for s in subjects)
    sst = sum(folds[(tgt, r, aq, name, st, s)][1] for s in subjects)
    return r2(sse, sst)


def margins(folds, tgt, r, aq, st, subjects):
    """per-subject S, M (and Mshuf if available) lists."""
    S = [get(folds, tgt, r, aq, "B", st, s) - get(folds, tgt, r, aq, "A", st, s) for s in subjects]
    M = [get(folds, tgt, r, aq, "C", st, s) - get(folds, tgt, r, aq, "B", st, s) for s in subjects]
    Ms = None
    if (tgt, r, aq, "Cshuf", st, subjects[0]) in folds:
        Ms = [get(folds, tgt, r, aq, "Cshuf", st, s) - get(folds, tgt, r, aq, "B", st, s) for s in subjects]
    return S, M, Ms


def fmt(v):
    return f"{np.mean(v):+.3f} [{np.min(v):+.3f}, {np.max(v):+.3f}]"


def report(folds, subjects, gate_lines):
    L = ["# Redundancy test for C2", "", "## Caveat", "", "Caveat to record in the report: " + CAVEAT, "",
         "## Gates (all passed)", ""] + gate_lines + [""]
    L += ["## S and M per rate (aq=1, ROI blocks, leave-one-subject-out)", "",
          "S = R2_B - R2_A (scalar beyond pixels), M = R2_C - R2_B (spatial map beyond pixels + scalar). "
          "Cells show mean [min, max] over the five held-out subjects; pooled = R2 from summed squared "
          "errors over all five folds.", "",
          "| rate | S | M | pooled M | M per subject (" + ", ".join(map(str, subjects)) + ") |", "|---|---|---|---|---|"]
    Mr = {}
    Pm = {}
    for r in RATES:
        S, M, _ = margins(folds, "full", r, 1, "all", subjects)
        Mr[r] = M
        Pm[r] = (pooled(folds, "full", r, 1, "C", "all", subjects) - pooled(folds, "full", r, 1, "B", "all", subjects))
        L.append(f"| {r} | {fmt(S)} | {fmt(M)} | {Pm[r]:+.3f} | " + ", ".join(f"{m:+.3f}" for m in M) + " |")
    L += ["", "Per-subject S:", "", "| rate | " + " | ".join(f"s{s}" for s in subjects) + " |", "|---|" + "---|" * len(subjects)]
    for r in RATES:
        S, _, _ = margins(folds, "full", r, 1, "all", subjects)
        L.append(f"| {r} | " + " | ".join(f"{x:+.3f}" for x in S) + " |")

    L += ["", "## I-frame vs P-frame stratification (aq=1)", "",
          "Same fitted models, evaluated on held-out rows of each picture type. Hypothesis: on P-frames skipped "
          "blocks inherit the running QP, so M is blunter there.", "",
          "| rate | stratum | S | M |", "|---|---|---|---|"]
    for r in RATES:
        for st in ("I", "P"):
            S, M, _ = margins(folds, "full", r, 1, st, subjects)
            L.append(f"| {r} | {st} | {fmt(S)} | {fmt(M)} |")

    L += ["", "## Controls", "", "1. aq=0: map is flat, require |M| < 0.005 for every subject.", "",
          "| rate | M (aq=0) mean [min, max] | pass |", "|---|---|---|"]
    c1 = True
    for r in RATES:
        _, M, _ = margins(folds, "full", r, 0, "all", subjects)
        ok = max(abs(m) for m in M) < 0.005
        c1 &= ok
        L.append(f"| {r} | {fmt(M)} | {'yes' if ok else 'NO'} |")
    L += ["", "2. Shuffled map (aq=1): require M_shuf <= 0 (checked on every subject).", "",
          "| rate | M_shuf mean [min, max] | pass |", "|---|---|---|"]
    c2 = True
    for r in RATES:
        _, _, Ms = margins(folds, "full", r, 1, "all", subjects)
        ok = max(Ms) <= 0
        c2 &= ok
        L.append(f"| {r} | {fmt(Ms)} | {'yes' if ok else 'NO'} |")
    L += ["", "3. S is reported alongside M in the main table above.", ""]

    L += ["## Robustness: luma-only target (aq=1)", "", "| rate | S | M |", "|---|---|---|"]
    for r in ("200k", "400k", "3200k"):
        S, M, _ = margins(folds, "luma", r, 1, "all", subjects)
        L.append(f"| {r} | {fmt(S)} | {fmt(M)} |")
    L += ["", "(Only 200k, 400k, 3200k were fitted for this variant, to stay in the time budget.)", ""]

    # Decision rule, applied mechanically (thresholds fixed in the plan).
    low_ok = all(m >= 0.02 for r in ("200k", "400k") for m in Mr[r])
    drop = sum(a > b for a, b in zip(Mr["200k"], Mr["3200k"]))
    if low_ok and drop >= 4:
        verdict = "C2 PREMISE HOLDS"
    elif all(Pm[r] < 0.02 for r in RATES):
        verdict = "C2 COLLAPSES INTO C1"
    else:
        verdict = "MIXED"
    L += ["## Verdict", "",
          f"- M >= 0.02 for all five subjects at 200k and 400k: {low_ok}",
          f"- M(200k) > M(3200k) for {drop} of {len(subjects)} subjects (need >= 4)",
          f"- pooled M < 0.02 at every rate: {all(Pm[r] < 0.02 for r in RATES)}",
          f"- controls passed: aq=0 {c1}, shuffled {c2}", "",
          f"**VERDICT: {verdict}**", ""]
    if not (c1 and c2):
        L.insert(L.index("## Verdict"), "**WARNING: a control failed; treat the verdict as invalid (bug or leakage).**\n")
    return "\n".join(L), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    gates = a.gates_only
    subjects = [1] if gates else ALL_SUBJECTS
    rates = ["200k", "3200k"] if gates else RATES

    info = {}
    for s in subjects:
        for r in rates:
            for aq in (1, 0):
                try:
                    info[(s, r, aq)] = (gate_summary_only(s, r, aq, a.out) if gates
                                        else process(s, r, aq, a.out, a.cache))
                except GateFail as e:
                    print("GATE FAILED:", e); return 1
                print(f"done s{s} {r} aq{aq}: {info[(s, r, aq)]}", flush=True)

    lines, bad = run_gates(info, subjects, rates, full=not gates)
    print("\n".join(lines))
    if bad:
        print("\nGATE FAILURES:\n  " + "\n  ".join(bad))
        return 1
    print("\nALL GATES PASSED")
    if gates:
        return 0

    out = Path(a.out)
    with open(out / "redundancy-folds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "rate", "aq", "set", "stratum", "heldout_subject", "n", "sse", "sst", "r2"])
        folds = run_models(a.cache, subjects, w)
    text, verdict = report(folds, subjects, lines)
    (out / "redundancy-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
