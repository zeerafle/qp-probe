#!/usr/bin/env python3
"""C1 pilot, extraction: per-window rPPG error and features for UBFC-rPPG.

    extract.py --data <dir> --out <dir> --shard K --of N
               [--subjects 1,10,...] [--rates 100k,...] [--tmp <dir>]

Implements PLAN.md sections 3 and 4 (Proposal/experiments/c1-pilot). For each
subject: one face box from the source clip, then for every condition (source,
each rate at aq=1, 200k at aq=0) encode -> qpprobe summary -> decode once via
an ffmpeg rawvideo pipe -> per-frame ROI mean RGB / motion / blockiness ->
POS and CHROM -> 10 s windows. Decoded video is never stored, and the mp4 is
deleted as soon as its features exist.

Alignment of the contact PPG (ground_truth.txt line 1) to the video: line 3
holds the PPG sample times in seconds (a repeated timestamp keeps its first
sample). They are NOT uniform (UBFC logs them as
the sensor delivered them: the first gaps are 0.011 and 0.044 s), and the
clip's last timestamp sits a little short of the last video frame. So the PPG
is linearly interpolated onto the video frame times t_i = i / fps (fps from
ffprobe), with i = 0 the first frame and the PPG clock taken as the video clock.
Frames after the last PPG sample are not used for windows (np.interp would
extend the edge value, which is a flat segment, not a signal). The number of
windows is therefore set by min(video frames, frames covered by the PPG) and
both lengths are printed.

Resumable: each subject writes partial files under <out>/partial/ (atomic
rename). A rerun skips subjects whose partials exist; the shard files are
merged from the partials at the end.
"""
import argparse
import gzip
import io
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import facebox  # noqa: E402

QPPROBE = ROOT / "qpprobe"
DEFAULT_RATES = ["100k", "200k", "400k", "800k", "1600k"]
AQ0_RATE = "200k"
EXTRA_RATE = "800k"  # keyframe-structure conditions (intra-refresh, g60), C0 only
INNER = 0.6           # fraction of the face box (per side length) used for the RGB trace
BAND = (0.7, 3.0)     # Hz, rPPG band and HR search band
WIN_S, HOP_S = 10.0, 5.0
POS_WIN_S = 1.6
NFFT = 1 << 16        # zero padding: ~0.0005 Hz bins at 30 fps


# ------------------------------------------------------------- video I/O ----

def probe_video(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


GOP_MODES = ("g60", "intra-refresh", "long")


def ffmpeg_x264(src, dst, rate, aq, gop="long"):
    # "long" (one keyframe, at the start) is the main recipe since the PLAN
    # amendment of 7 October 2026. The redundancy-test recipe (-g 60) puts a
    # keyframe every ~2 s; its 2nd harmonic (~0.98 Hz, ~59 bpm) lands inside
    # the HR band and POS locked onto it at every rate in the smoke test, so it
    # survives only as an artefact condition at EXTRA_RATE, next to
    # intra-refresh (which showed an unexplained 65 bpm peak on subject 1).
    if gop == "g60":
        g, params = "60", f"aq-mode={aq}"
    elif gop == "intra-refresh":
        g, params = "250", f"aq-mode={aq}:intra-refresh=1"
    else:
        g, params = "9999", f"aq-mode={aq}"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-b:v", rate, "-minrate", rate, "-maxrate", rate,
         "-bufsize", rate, "-tune", "zerolatency", "-g", g,
         "-x264-params", params, str(dst)],
        check=True)


def qp_summary(path, box):
    out = subprocess.run(
        [str(QPPROBE), str(path), "--max-frames", "0", "--summary-csv",
         "--roi", ",".join(map(str, box))],
        capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(out))


# ------------------------------------------------------ per-frame features ----

def _blockiness(y, x0, y0, B):
    """Boundary / interior mean absolute gradient, averaged over both axes.
    Coordinates are absolute so the 16-aligned ROI keeps the block grid."""
    ratios = []
    for axis, off in ((1, x0), (0, y0)):
        g = np.abs(np.diff(y, axis=axis))
        idx = (np.arange(g.shape[axis]) + 1 + off) % B == 0
        sel = g.compress(idx, axis=axis)
        oth = g.compress(~idx, axis=axis)
        ratios.append(sel.mean() / (oth.mean() + 1e-6))
    return float(np.mean(ratios))


def stream_features(path, box, w, h):
    """One decode pass. Returns per-frame arrays: rgb (N,3) inner-ROI mean,
    ti (std of luma difference over the frame, P.910 style), rdiff (mean |luma
    difference| in the ROI box), blk8, blk16, hf (mean squared Laplacian in the
    ROI box). Frame 0 has no previous frame, so its motion entries are NaN."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ix0 = x0 + int(round(bw * (1 - INNER) / 2))
    ix1 = x1 - int(round(bw * (1 - INNER) / 2))
    iy0 = y0 + int(round(bh * (1 - INNER) / 2))
    iy1 = y1 - int(round(bh * (1 - INNER) / 2))

    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
        stdout=subprocess.PIPE, bufsize=1 << 22)
    fsz = w * h * 3
    rgb, ti, rdiff, blk8, blk16, hf = [], [], [], [], [], []
    prev = None
    buf = bytearray(fsz)
    view = memoryview(buf)
    while True:
        got = 0
        while got < fsz:
            n = proc.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        if got < fsz:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        rgb.append(fr[iy0:iy1, ix0:ix1].reshape(-1, 3).mean(axis=0))
        f32 = fr.astype(np.float32)
        luma = f32 @ np.array([0.299, 0.587, 0.114], np.float32)
        roi = luma[y0:y1, x0:x1]
        if prev is None:
            ti.append(np.nan)
            rdiff.append(np.nan)
        else:
            d = luma - prev
            ti.append(float(d.std()))
            rdiff.append(float(np.abs(d[y0:y1, x0:x1]).mean()))
        prev = luma
        blk8.append(_blockiness(roi, x0, y0, 8))
        blk16.append(_blockiness(roi, x0, y0, 16))
        lap = (roi[1:-1, 1:-1] * 4 - roi[:-2, 1:-1] - roi[2:, 1:-1]
               - roi[1:-1, :-2] - roi[1:-1, 2:])
        hf.append(float((lap ** 2).mean()))
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg decode of {path} failed ({rc})")
    return dict(rgb=np.array(rgb), ti=np.array(ti), rdiff=np.array(rdiff),
                blk8=np.array(blk8), blk16=np.array(blk16), hf=np.array(hf))


# ---------------------------------------------------------------- rPPG -------

def bandpass(x, fps):
    b, a = butter(3, BAND, btype="bandpass", fs=fps)
    return filtfilt(b, a, x)


def pos(rgb, fps):
    """Wang et al. 2016, 1.6 s sliding window, overlap-add, then band-pass."""
    n = len(rgb)
    l = int(np.ceil(POS_WIN_S * fps))
    P = np.array([[0, 1, -1], [-2, 1, 1]], float)
    H = np.zeros(n)
    for e in range(l, n + 1):
        C = rgb[e - l:e]
        Cn = C / (C.mean(axis=0) + 1e-9)
        S = P @ Cn.T
        hh = S[0] + S[0].std() / (S[1].std() + 1e-9) * S[1]
        H[e - l:e] += hh - hh.mean()
    return bandpass(H, fps)


def chrom(rgb, fps):
    """de Haan & Jeanne 2013. Same 1.6 s overlap-add framing as POS (the
    rPPG-toolbox convention) so the two differ only in the projection."""
    n = len(rgb)
    l = int(np.ceil(POS_WIN_S * fps))
    H = np.zeros(n)
    for e in range(l, n + 1):
        C = rgb[e - l:e]
        Cn = C / (C.mean(axis=0) + 1e-9)
        Xs = 3 * Cn[:, 0] - 2 * Cn[:, 1]
        Ys = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
        hh = Xs - Xs.std() / (Ys.std() + 1e-9) * Ys
        H[e - l:e] += hh - hh.mean()
    return bandpass(H, fps)


def spectrum(x, fps):
    """Zero-padded Hann periodogram restricted to the HR band. Shared by rPPG
    and the reference so the two estimates differ only in the signal."""
    x = x - x.mean()
    X = np.fft.rfft(x * np.hanning(len(x)), NFFT)
    f = np.fft.rfftfreq(NFFT, 1 / fps)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return f[m], np.abs(X[m]) ** 2


def hr_bpm(x, fps):
    f, p = spectrum(x, fps)
    return 60.0 * f[np.argmax(p)]


def signal_quality(x, fps):
    """The S set, computed on the POS window."""
    f, p = spectrum(x, fps)
    k = np.argmax(p)
    f0 = f[k]
    near = (np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.1)
    sig, tot = p[near].sum(), p.sum()
    snr = 10 * np.log10(sig / max(tot - sig, 1e-12 * tot) + 1e-12)
    away = np.abs(f - f0) > 0.15
    second = p[away].max() if away.any() else 1e-12 * p[k]
    pn = p / p.sum()
    ent = -(pn * np.log(pn + 1e-300)).sum() / np.log(len(pn))
    xc = x - x.mean()
    ac = np.correlate(xc, xc, "full")[len(xc) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(fps / BAND[1]), int(np.ceil(fps / BAND[0]))
    per = ac[lo:hi + 1].max()
    return dict(S_snr=snr, S_peak_ratio=p[k] / (second + 1e-12),
                S_entropy=ent, S_autocorr=per)


# ----------------------------------------------------------- ground truth ----

def load_reference(gt_path, fps, n_frames):
    """Contact PPG on the video frame times. Returns (signal, n_usable)."""
    lines = open(gt_path).read().split("\n")
    ppg = np.array(lines[0].split(), float)
    t = np.array(lines[2].split(), float)
    if len(ppg) != len(t):
        raise ValueError(f"{gt_path}: {len(ppg)} PPG samples vs {len(t)} timestamps")
    # UBFC logs the odd repeated timestamp (subjects 1 and 10 have one each).
    # Keep the first sample of any repeat; np.interp needs increasing x.
    keep = np.concatenate([[True], np.diff(t) > 0])
    ppg, t = ppg[keep], t[keep]
    ft = np.arange(n_frames) / fps
    usable = int(np.searchsorted(ft, t[-1], side="right"))
    ref = np.interp(ft, t, ppg)
    return ref, usable


# ------------------------------------------------------------ windowing ------

def slope(y, fps):
    if len(y) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(y)) / fps, y, 1)[0])


def window_rows(subject, cond, rate_k, aq, gop, feats, qp, ref_bvp, ref_n, fps):
    n = min(len(feats["rgb"]), ref_n)
    wn, hn = int(round(WIN_S * fps)), int(round(HOP_S * fps))
    bvp_pos = pos(feats["rgb"][:n], fps)
    bvp_chr = chrom(feats["rgb"][:n], fps)
    log_rate = 0.0 if rate_k == 0 else float(np.log(rate_k * 1000.0))
    rows = []
    for s in range(0, n - wn + 1, hn):
        sl = slice(s, s + wn)
        hr_ref = hr_bpm(ref_bvp[sl], fps)
        hr_p = hr_bpm(bvp_pos[sl], fps)
        hr_c = hr_bpm(bvp_chr[sl], fps)
        r = dict(subject=subject, cond=cond, rate_k=rate_k, aq=aq, gop=gop,
                 win_start=s, hr_ref=hr_ref, hr_pos=hr_p, hr_chrom=hr_c,
                 err=abs(hr_p - hr_ref), err_chrom=abs(hr_c - hr_ref))
        r.update(signal_quality(bvp_pos[sl], fps))
        r["S_hr_disagree"] = abs(hr_p - hr_c)
        r["M_ti"] = float(np.nanmax(feats["ti"][sl]))
        r["M_roi_diff"] = float(np.nanmean(feats["rdiff"][sl]))
        r["R_log_bitrate"] = log_rate
        r["P_blk8"] = float(feats["blk8"][sl].mean())
        r["P_blk16"] = float(feats["blk16"][sl].mean())
        r["P_hf"] = float(feats["hf"][sl].mean())
        q = qp.iloc[sl] if qp is not None else None
        nan = np.nan
        if q is None or len(q) < wn:
            r.update(Q_mean=nan, Q_std=nan, Q_max=nan, Q_slope=nan,
                     Q_ifrac=nan, Qmap_roi_std=nan, Qmap_roi_minus_frame=nan)
        else:
            # mean_qp, not frame_qp: libavcodec's frame-level qp is the PPS
            # init QP (constant 26 for these encodes), and the rate control
            # shows up only in the per-block deltas. mean_qp is the decoder's
            # real per-frame QP scalar.
            fq = q["mean_qp"].to_numpy(float)
            r["Q_mean"] = float(np.nanmean(fq))
            r["Q_std"] = float(np.nanstd(fq))
            r["Q_max"] = float(np.nanmax(fq))
            r["Q_slope"] = slope(fq, fps)
            r["Q_ifrac"] = float((q["pict_type"] == "I").mean())
            r["Qmap_roi_std"] = float(np.nanmean(q["roi_std"].to_numpy(float)))
            r["Qmap_roi_minus_frame"] = float(np.nanmean(
                q["roi_mean"].to_numpy(float) - q["mean_qp"].to_numpy(float)))
        rows.append(r)
    return rows


# ---------------------------------------------------------------- driver -----

def find_subjects(data):
    found = {}
    for p in pathlib.Path(data).rglob("vid.avi"):
        m = re.fullmatch(r"subject(\d+)", p.parent.name)
        if m and (p.parent / "ground_truth.txt").exists():
            found[int(m.group(1))] = p.parent
    return dict(sorted(found.items()))


def run_subject(sid, sdir, rates, tmp, partial):
    t0 = time.time()
    src = sdir / "vid.avi"
    w, h, fps = probe_video(src)
    box, note = facebox.face_box(str(src))
    if box is None:
        raise RuntimeError(f"subject{sid}: face box failed - {note}")
    print(f"  box {box} ({note})", flush=True)

    # (rate, rate_k, aq, gop). Main sweep and the aq=0 control are long-GOP;
    # the two keyframe-structure conditions sit at one rate, for C0, and are
    # never pooled into the main fit (analyse.py filters on gop == "long").
    conds = [("source", 0, -1, "none")] + [(r, int(r[:-1]), 1, "long") for r in rates]
    conds.append((AQ0_RATE, int(AQ0_RATE[:-1]), 0, "long"))
    conds += [(EXTRA_RATE, int(EXTRA_RATE[:-1]), 1, g) for g in ("intra-refresh", "g60")]

    rows, qp_rows = [], []
    ref_bvp = ref_n = None
    for name, rate_k, aq, gop in conds:
        tc = time.time()
        label = "source" if rate_k == 0 else f"{name}-aq{aq}-{gop}"
        if rate_k == 0:
            feats = stream_features(src, box, w, h)
            qp = None
            path = None
        else:
            path = tmp / f"s{sid}-{label}.mp4"
            try:
                ffmpeg_x264(src, path, name, aq, gop)
                qp = qp_summary(path, box)
                feats = stream_features(path, box, w, h)
            finally:
                path.unlink(missing_ok=True)
            if len(qp) != len(feats["rgb"]):
                print(f"  WARNING {label}: qpprobe {len(qp)} frames vs decode "
                      f"{len(feats['rgb'])}", flush=True)
        if ref_bvp is None:
            ref_raw, usable = load_reference(sdir / "ground_truth.txt", fps,
                                             len(feats["rgb"]))
            ref_bvp, ref_n = bandpass(ref_raw, fps), usable
            print(f"  frames {len(feats['rgb'])}, fps {fps:.3f}, PPG covers "
                  f"{usable} frames", flush=True)
        cond = label
        rows += window_rows(sid, cond, rate_k, aq, gop, feats, qp, ref_bvp, ref_n, fps)
        if qp is not None:
            q = qp.copy()
            q.insert(0, "cond", cond)
            q.insert(0, "subject", sid)
            qp_rows.append(q)
        print(f"  {label:<12} {time.time() - tc:6.1f}s", flush=True)

    feat_df = pd.DataFrame(rows)
    qp_df = pd.concat(qp_rows, ignore_index=True)
    partial.mkdir(parents=True, exist_ok=True)
    # qp first, features last: the features file is the "done" marker.
    for df, name in ((qp_df, f"s{sid}.qp.csv.gz"), (feat_df, f"s{sid}.feat.csv.gz")):
        tmpf = partial / (name + ".tmp")
        df.to_csv(tmpf, index=False, compression="gzip")
        os.replace(tmpf, partial / name)
    print(f"  subject{sid} done in {time.time() - t0:.0f}s", flush=True)


def env_info(tag):
    def first(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    ffv = first(["ffmpeg", "-version"]).split("\n")
    libavcodec = next((l.strip() for l in ffv if l.startswith("libavcodec")), "")
    # x264 writes its version into the stream as an SEI string; the ffmpeg
    # banner only names the library, not its build.
    x264 = ""
    with tempfile.TemporaryDirectory() as d:
        t = pathlib.Path(d) / "t.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=gray:s=64x64:d=0.2", "-c:v", "libx264", str(t)])
        m = re.search(rb"x264 - core \d+[ -~]*", t.read_bytes()) if t.exists() else None
        x264 = m.group(0).decode()[:120] if m else ""
    import scipy
    import sklearn
    return dict(
        tag=tag,
        git_commit=first(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) or "unknown",
        git_dirty=bool(first(["git", "-C", str(ROOT), "status", "--porcelain"])),
        ffmpeg=ffv[0] if ffv else "",
        libavcodec_banner=libavcodec,
        libavcodec_dev=first(["pkg-config", "--modversion", "libavcodec"]),
        x264=x264,
        python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__,
        pandas=pd.__version__)


def merge(out, tag, partial):
    feats = sorted(partial.glob("s*.feat.csv.gz"), key=lambda p: int(p.name[1:].split(".")[0]))
    if not feats:
        return
    fdf = pd.concat([pd.read_csv(p) for p in feats], ignore_index=True)
    try:
        import pyarrow  # noqa: F401
        fdf.to_parquet(out / f"features-{tag}.parquet", index=False)
        kind = "parquet"
    except ImportError:
        fdf.to_csv(out / f"features-{tag}.csv", index=False)
        kind = "csv"
    qpf = sorted(partial.glob("s*.qp.csv.gz"), key=lambda p: int(p.name[1:].split(".")[0]))
    pd.concat([pd.read_csv(p) for p in qpf], ignore_index=True).to_csv(
        out / f"qp-{tag}.csv.gz", index=False, compression="gzip")
    print(f"merged {len(feats)} subjects -> features-{tag}.{kind}, qp-{tag}.csv.gz "
          f"({len(fdf)} window rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--subjects", help="comma list of subject numbers (before sharding)")
    ap.add_argument("--rates", default=",".join(DEFAULT_RATES))
    ap.add_argument("--tmp", help="scratch dir for encodes (default: system temp)")
    a = ap.parse_args()

    if not QPPROBE.exists():
        sys.exit(f"{QPPROBE} missing: run `make qpprobe` (or ./ensure-build.sh logic) first")
    rates = a.rates.split(",")
    tag = f"{a.shard}of{a.of}"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f"partial-{tag}"

    subs = find_subjects(a.data)
    if a.subjects:
        want = {int(s) for s in a.subjects.split(",")}
        subs = {k: v for k, v in subs.items() if k in want}
    if not subs:
        sys.exit(f"no subject*/vid.avi with ground_truth.txt under {a.data}")
    mine = list(subs.items())[a.shard::a.of]
    print(f"shard {tag}: subjects {[k for k, _ in mine]} of {list(subs)}", flush=True)

    (out / f"env-{tag}.json").write_text(json.dumps(env_info(tag), indent=2))

    tmp = pathlib.Path(a.tmp) if a.tmp else pathlib.Path(tempfile.mkdtemp(prefix="c1pilot-"))
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for sid, sdir in mine:
        if (partial / f"s{sid}.feat.csv.gz").exists():
            print(f"subject{sid}: already done, skipping", flush=True)
            continue
        print(f"subject{sid} ({sdir})", flush=True)
        run_subject(sid, sdir, rates, tmp, partial)
    merge(out, tag, partial)
    print(f"shard {tag} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
