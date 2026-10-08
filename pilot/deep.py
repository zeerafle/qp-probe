#!/usr/bin/env python3
"""Deep-model keyframe test: are TS-CAN and PhysNet fooled by periodic keyframes?

    deep.py extract --data DIR --out DIR --shard K --of N [--subjects 1,10]
                    [--rates 3200k,800k,400k] [--modes long,g30,g60,g120,intra-refresh]
                    [--toolbox PATH] [--tmp DIR]
    deep.py md5check --data DIR --subject S --out DIR [--tmp DIR]
    deep.py analyse --in DIR [DIR ...] --out DIR [--smoke]

Follow-up to keyframe.py (POS locks onto the keyframe harmonic with -g 60).
Same question for the models people deploy: rPPG-Toolbox TS-CAN and PhysNet,
released PURE-trained weights (never saw UBFC-rPPG), tested on UBFC-rPPG with
POS and CHROM re-run on the very same encodes.

Everything the toolbox does to the video is the toolbox's own code, imported
from the cloned repo: BaseLoader.crop_face_resize (box enlarged 1.5x about its
centre, INTER_AREA resize to 72x72), diff_normalize_data, standardized_data and
chunk (180 frames for TS-CAN, 128 for PhysNet), the model classes, and
evaluation.post_process._detrend. Only the face box differs: facebox.face_box
on the source clip, once per subject, applied to every encode of that subject
(the subjects do not move; the toolbox detects on frame 0 and holds it too).

Encodes are the pilot recipe plus threads=1, because default-thread x264 is
not reproducible run to run (every earlier number is one draw). Gate G2
(md5check) proves that the threads=1 encodes are.

Pre-registered rule, fixed before the first run (amended 8 Oct 2026, before any
42-subject run: 3200k added after the one-subject smoke showed TS-CAN and
PhysNet already failing at 800k with a single keyframe, which makes the g60
comparison meaningless there). Per model, pooled over 42 subjects, at each
gated rate (3200k and 800k) where the model's BASELINE IS HEALTHY, i.e. its
long-GOP median err <= its uncompressed median err + 3 bpm; where it is not,
the cell reads "BASELINE FAILS, no verdict" and the failure itself is reported.
Long GOP is the reference:
  Fooled  if median err(g60) >= median err(long) + 10 bpm AND the share of
          g60 windows within 1.5 bpm of a keyframe harmonic is >= 2x the
          chance share (from hr_ref, as keyframe.py)
  Immune  if median err(g60) <= median err(long) + 3 bpm
  Partial otherwise. Numbers are reported, no relabelling.
Gate G1: uncompressed TS-CAN / PhysNet MAE <= 6 bpm and Pearson r >= 0.8
(window HR vs contact PPG HR, pooled over source windows). Gate G2:
deterministic encodes (md5check). A gate failure stops the analysis.
g30 / g120 and 400k are reported, not gated. A verdict at one healthy rate is
never carried over to another rate.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import extract as E   # noqa: E402  (rPPG, reference, windows: one implementation)
import keyframe as K  # noqa: E402  (keyframe_period)
import facebox        # noqa: E402

MODES = ["long", "g30", "g60", "g120", "intra-refresh"]
RATES = ["3200k", "800k", "400k"]
GATED_RATES = (3200, 800)
BASELINE_TOL = 3.0   # bpm: long-GOP median err vs uncompressed median err
METHODS = ["POS", "CHROM", "TSCAN", "PHYSNET"]
DEEP = ["TSCAN", "PHYSNET"]
WEIGHTS = {"TSCAN": "PURE_TSCAN.pth", "PHYSNET": "PURE_PhysNet_DiffNormalized.pth"}
CHUNK = {"TSCAN": 180, "PHYSNET": 128}
FRAME_DEPTH = 10
SIZE = 72
BOX_COEF = 1.5
BATCH = 4                      # chunks per forward pass (toolbox INFERENCE.BATCH_SIZE)
HARM_TOL = 1.5                 # bpm
N_SUBJECTS = 42
N_BOOT, BOOT_SEED = 1000, 0
WEIGHTS_REPO = "https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox"
WIN_S, HOP_S = E.WIN_S, E.HOP_S


# ---------------------------------------------------------------- encode -----

def encode(src, dst, rate, mode):
    # Pilot / keyframe.py recipe; the only addition is threads=1 so the
    # bitstream does not depend on thread timing.
    if mode == "long":
        g, params = "9999", "aq-mode=1:threads=1"
    elif mode == "intra-refresh":
        g, params = "250", "aq-mode=1:threads=1:intra-refresh=1"
    else:
        g, params = mode[1:], "aq-mode=1:threads=1"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
         "-b:v", rate, "-minrate", rate, "-maxrate", rate, "-bufsize", rate,
         "-tune", "zerolatency", "-g", g, "-x264-params", params, str(dst)],
        check=True)


# --------------------------------------------------------------- toolbox -----

def load_toolbox(path):
    """Put the cloned toolbox on sys.path (last, so it cannot shadow ours) and
    import the pieces we use."""
    path = pathlib.Path(path).resolve()
    if not (path / "dataset" / "data_loader" / "BaseLoader.py").exists():
        sys.exit(f"{path} is not an rPPG-Toolbox clone (git clone "
                 "https://github.com/ubicomplab/rPPG-Toolbox)")
    sys.path.append(str(path))
    import torch
    from dataset.data_loader.BaseLoader import BaseLoader
    from evaluation.post_process import _detrend
    from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
    from neural_methods.model.TS_CAN import TSCAN
    return dict(path=path, torch=torch, BL=BaseLoader, detrend=_detrend,
                PhysNet=PhysNet_padding_Encoder_Decoder_MAX, TSCAN=TSCAN)


def toolbox_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def fetch_weights(tb):
    """Released weights from the pinned toolbox commit, cached in the clone."""
    d = tb["path"] / "final_model_release"
    d.mkdir(exist_ok=True)
    commit = toolbox_commit(tb["path"])
    ref = commit if commit != "unknown" else "main"
    for name in WEIGHTS.values():
        p = d / name
        if p.exists() and p.stat().st_size > 1000:
            continue
        url = f"{WEIGHTS_REPO}/{ref}/final_model_release/{name}"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, p.with_suffix(".tmp"))
        os.replace(p.with_suffix(".tmp"), p)
    return {k: d / v for k, v in WEIGHTS.items()}


def load_models(tb):
    torch = tb["torch"]
    paths = fetch_weights(tb)
    models = {}
    for key, cls in (("TSCAN", lambda: tb["TSCAN"](frame_depth=FRAME_DEPTH, img_size=SIZE)),
                     ("PHYSNET", lambda: tb["PhysNet"](frames=CHUNK["PHYSNET"]))):
        m = cls()
        sd = torch.load(paths[key], map_location="cpu")
        # Saved from nn.DataParallel: keys carry a "module." prefix.
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        m.load_state_dict(sd)
        models[key] = m.eval()
    return models


class FixedBox:
    """Stand-in for a BaseLoader instance: crop_face_resize only needs
    self.face_detection, and here the (enlarged) box is decided once."""

    def __init__(self, box):
        self.box = box

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        return list(self.box)


def toolbox_box(box, w, h):
    """facebox's 16-aligned rectangle -> the toolbox's face box: a square
    about the same centre (the Haar box is square), enlarged by BOX_COEF about
    that centre, origin clipped at 0 (the far edge is clipped when slicing)."""
    x0, y0, x1, y1 = box
    s = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x, y = cx - s / 2, cy - s / 2
    x = max(0, x - (BOX_COEF - 1.0) / 2 * s)
    y = max(0, y - (BOX_COEF - 1.0) / 2 * s)
    return [x, y, BOX_COEF * s, BOX_COEF * s]


# ----------------------------------------------------------------- decode ----

def decode_pass(path, w, h, box, tb):
    """One ffmpeg rawvideo pass feeding both consumers: the RGB trace (inner
    ROI mean, exactly extract.stream_features' rgb) and the toolbox-cropped
    72x72 frames, kept as uint8 (~1800 x 72 x 72 x 3 = 28 MB). cv2.VideoCapture
    is not used: it segfaults on the raw bgr24 AVIs."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ix0 = x0 + int(round(bw * (1 - E.INNER) / 2))
    ix1 = x1 - int(round(bw * (1 - E.INNER) / 2))
    iy0 = y0 + int(round(bh * (1 - E.INNER) / 2))
    iy1 = y1 - int(round(bh * (1 - E.INNER) / 2))
    stub = FixedBox(toolbox_box(box, w, h))

    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
        stdout=subprocess.PIPE, bufsize=1 << 22)
    fsz = w * h * 3
    buf = bytearray(fsz)
    view = memoryview(buf)
    rgb, crops = [], []
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
        # One frame at a time: the whole decoded clip (1.6 GB) never exists.
        c = tb["BL"].crop_face_resize(stub, fr[None], True, "HC", True, BOX_COEF,
                                      False, 30, False, SIZE, SIZE)
        crops.append(c[0].astype(np.uint8))
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg decode of {path} failed ({rc})")
    return np.array(rgb), np.stack(crops)


# -------------------------------------------------------------- inference ----

def run_deep(crops, models, tb, fps):
    """Per-frame BVP for each deep model. Frames past the last whole chunk get
    no output (the toolbox drops them too), so the BVP is shorter than the
    clip by N % chunk frames."""
    torch = tb["torch"]
    BL = tb["BL"]
    n = len(crops)
    out = {}
    for key in DEEP:
        f = crops.astype(np.float64)    # the toolbox works on float64 frames
        diff = BL.diff_normalize_data(f.copy())
        if key == "TSCAN":
            data = np.concatenate([diff, BL.standardized_data(f.copy())], axis=-1)
        else:
            data = diff
        del f, diff
        clips, _ = BL.chunk(None, data, np.zeros(n), CHUNK[key])
        del data
        if key == "TSCAN":
            x = np.transpose(clips, (0, 1, 4, 2, 3)).astype(np.float32, order="C")   # N D C H W
        else:
            x = np.transpose(clips, (0, 4, 1, 2, 3)).astype(np.float32, order="C")   # N C D H W
        del clips
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), BATCH):
                xb = torch.from_numpy(x[i:i + BATCH])
                if key == "TSCAN":
                    N, D, C, H, W = xb.shape
                    y = models[key](xb.reshape(N * D, C, H, W)).reshape(-1)
                else:
                    y = models[key](xb)[0].reshape(-1)
                preds.append(y.numpy().astype(np.float64))
        del x
        pred = np.concatenate(preds)
        # Toolbox post-processing for diff-normalised outputs: cumulative sum
        # over the whole video, then its lambda=100 detrend.
        bvp = tb["detrend"](np.cumsum(pred), 100)
        out[key] = E.bandpass(bvp, fps)
    return out


# ---------------------------------------------------------------- windows ----

def condition_rows(sid, mode, rate_k, rgb, deep_bvp, ref_bvp, ref_n, fps, f0):
    """Same 10 s / 5 s windows and hr_bpm as the pilot. A deep model's BVP
    stops at its last whole chunk, so its last window or two can be missing
    where POS/CHROM have them; those rows are simply absent (not padded)."""
    n = min(len(rgb), ref_n)
    wn, hn = int(round(WIN_S * fps)), int(round(HOP_S * fps))
    sigs = {"POS": E.pos(rgb[:n], fps), "CHROM": E.chrom(rgb[:n], fps)}
    for k, b in (deep_bvp or {}).items():
        sigs[k] = b[:n]
    rows = []
    for meth in METHODS:
        if meth not in sigs:
            continue
        bvp = sigs[meth]
        for s in range(0, len(bvp) - wn + 1, hn):
            hr_ref = E.hr_bpm(ref_bvp[s:s + wn], fps)
            hr_est = E.hr_bpm(bvp[s:s + wn], fps)
            harm = np.nan
            if f0:
                harm = abs(hr_est - f0 * round(hr_est / f0))   # to nearest k*fps/P
            rows.append(dict(subject=sid, mode=mode, rate_k=rate_k, method=meth,
                             win=s, hr_ref=hr_ref, hr_est=hr_est,
                             err=abs(hr_est - hr_ref), harm_dist=harm,
                             f0=f0 if f0 else np.nan))
    return rows


# ---------------------------------------------------------------- extract ----

def run_subject(sid, sdir, rates, modes, tmp, partial, tb, models):
    t0 = time.time()
    src = sdir / "vid.avi"
    w, h, fps = E.probe_video(src)
    box, note = facebox.face_box(str(src))
    if box is None:
        raise RuntimeError(f"subject{sid}: face box failed - {note}")
    print(f"  box {box} ({note}) -> toolbox box {toolbox_box(box, w, h)}", flush=True)

    conds = [("source", 0)] + [(m, int(r[:-1])) for r in rates for m in modes]
    rows = []
    ref_bvp = ref_n = None
    for mode, rate_k in conds:
        tc = time.time()
        f0 = None
        if mode == "source":
            rgb, crops = decode_pass(src, w, h, box, tb)
        else:
            path = tmp / f"s{sid}-{mode}-{rate_k}k.mp4"
            try:
                encode(src, path, f"{rate_k}k", mode)
                if mode.startswith("g"):
                    # Keyframe period from the real frame types, as keyframe.py.
                    q = E.qp_summary(path, box)
                    P = K.keyframe_period((q.pict_type == "I").to_numpy())
                    f0 = 60 * fps / P if P else None
                rgb, crops = decode_pass(path, w, h, box, tb)
            finally:
                path.unlink(missing_ok=True)
        if ref_bvp is None:
            ref_raw, ref_n = E.load_reference(sdir / "ground_truth.txt", fps, len(rgb))
            ref_bvp = E.bandpass(ref_raw, fps)
            print(f"  frames {len(rgb)}, fps {fps:.3f}, PPG covers {ref_n} frames", flush=True)
        tdec = time.time() - tc
        deep_bvp = run_deep(crops, models, tb, fps)
        del crops
        rows += condition_rows(sid, mode, rate_k, rgb, deep_bvp, ref_bvp, ref_n, fps, f0)
        print(f"  {mode:<14}{rate_k:>5}k {time.time() - tc:6.1f}s "
              f"(encode+decode {tdec:.1f}s, models {time.time() - tc - tdec:.1f}s)", flush=True)

    partial.mkdir(parents=True, exist_ok=True)
    tmpf = partial / f"s{sid}.feat.csv.gz.tmp"
    pd.DataFrame(rows).to_csv(tmpf, index=False, compression="gzip")
    os.replace(tmpf, partial / f"s{sid}.feat.csv.gz")
    print(f"  subject{sid} done in {time.time() - t0:.0f}s", flush=True)


def env_info(tag, tb):
    d = E.env_info(tag)
    d.update(torch=tb["torch"].__version__, toolbox_commit=toolbox_commit(tb["path"]),
             weights=WEIGHTS, rates=RATES, threads_x264=1)
    return d


def cmd_extract(a):
    if not E.QPPROBE.exists():
        sys.exit(f"{E.QPPROBE} missing: run `make qpprobe` first")
    rates = a.rates.split(",")
    modes = a.modes.split(",")
    tag = f"{a.shard}of{a.of}"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f"partial-{tag}"

    subs = E.find_subjects(a.data)
    if a.subjects:
        want = {int(s) for s in a.subjects.split(",")}
        subs = {k: v for k, v in subs.items() if k in want}
    if not subs:
        sys.exit(f"no subject*/vid.avi with ground_truth.txt under {a.data}")
    mine = list(subs.items())[a.shard::a.of]
    print(f"shard {tag}: subjects {[k for k, _ in mine]} of {list(subs)}", flush=True)

    tb = load_toolbox(a.toolbox)
    models = load_models(tb)
    (out / f"env-{tag}.json").write_text(json.dumps(env_info(tag, tb), indent=2))

    tmp = pathlib.Path(a.tmp) if a.tmp else pathlib.Path(tempfile.mkdtemp(prefix="deepkf-"))
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for sid, sdir in mine:
        if (partial / f"s{sid}.feat.csv.gz").exists():
            print(f"subject{sid}: already done, skipping", flush=True)
            continue
        print(f"subject{sid} ({sdir})", flush=True)
        run_subject(sid, sdir, rates, modes, tmp, partial, tb, models)

    feats = sorted(partial.glob("s*.feat.csv.gz"), key=lambda p: int(p.name[1:].split(".")[0]))
    if feats:
        df = pd.concat([pd.read_csv(p) for p in feats], ignore_index=True)
        df.to_csv(out / f"features-{tag}.csv", index=False)
        print(f"merged {len(feats)} subjects -> features-{tag}.csv ({len(df)} window rows)")
    print(f"shard {tag} finished in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------- md5check --

def md5(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def cmd_md5check(a):
    """Gate G2: encode the same clip twice with threads=1, compare md5. Writes
    md5.txt (first line PASS or FAIL) into --out for `analyse` to read."""
    subs = E.find_subjects(a.data)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(a.tmp) if a.tmp else pathlib.Path(tempfile.mkdtemp(prefix="deepkf-md5-"))
    tmp.mkdir(parents=True, exist_ok=True)
    lines, ok = [], True
    for sid in [int(s) for s in str(a.subject).split(",")]:
        if sid not in subs:
            sys.exit(f"subject{sid} not found under {a.data}")
        src = subs[sid] / "vid.avi"
        for mode in a.modes.split(","):
            h = []
            for rep in (1, 2):
                p = tmp / f"md5-s{sid}-{mode}-{rep}.mp4"
                try:
                    encode(src, p, a.rate, mode)
                    h.append(md5(p))
                finally:
                    p.unlink(missing_ok=True)
            same = h[0] == h[1]
            ok &= same
            lines.append(f"subject{sid} {mode} {a.rate} {h[0]} {h[1]} "
                         f"{'equal' if same else 'DIFFERENT'}")
            print(lines[-1], flush=True)
    (out / "md5.txt").write_text(("PASS" if ok else "FAIL") + "\n" + "\n".join(lines) + "\n")
    print("G2:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------- analyse ---

def chance_share(x):
    """Share of windows whose hr_ref alone would sit within HARM_TOL of a
    harmonic of that subject's keyframe rate (keyframe.py analyse)."""
    d = np.abs(x.hr_ref - x.f0 * np.round(x.hr_ref / x.f0))
    return float((d <= HARM_TOL).mean())


def pooled_median(groups, idx):
    return float(np.median(np.concatenate([groups[i] for i in idx])))


def bootstrap_diff(by_subj_a, by_subj_b, subjects):
    """Subject-level bootstrap of median(a) - median(b), pooled windows of the
    resampled subjects. Fixed seed -> identical numbers on rerun."""
    rng = np.random.default_rng(BOOT_SEED)
    subs = [s for s in subjects if s in by_subj_a and s in by_subj_b]
    A = [by_subj_a[s] for s in subs]
    B = [by_subj_b[s] for s in subs]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(subs), len(subs))
        d[i] = pooled_median(A, idx) - pooled_median(B, idx)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(med_mode, med_long, share, chance):
    ratio = share / chance if chance > 0 else np.inf
    if med_mode >= med_long + 10 and ratio >= 2:
        return "Fooled", ratio
    if med_mode <= med_long + 3:
        return "Immune", ratio
    return "Partial", ratio


def read_md5(dirs):
    for d in dirs:
        for p in [pathlib.Path(d) / "md5.txt"] + sorted(pathlib.Path(d).rglob("md5.txt")):
            if p.exists():
                return p.read_text().split("\n")[0].strip(), p
    return None, None


def cmd_analyse(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for d in a.inp:
        files += sorted(pathlib.Path(d).rglob("features-*of*.csv"))
    if not files:
        sys.exit("no features-*of*.csv under " + " ".join(a.inp))
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df.sort_values(["subject", "rate_k", "mode", "method", "win"]).reset_index(drop=True)
    df.to_csv(out / "windows.csv", index=False)
    subjects = sorted(df.subject.unique())
    L = [f"# Deep-model keyframe test\n\n{len(subjects)} subjects, {len(files)} feature files, "
         "10 s windows / 5 s hop, threads=1 encodes, PURE-trained TS-CAN / PhysNet.\n"]
    blocked = []

    # ---- gate G1: uncompressed deep models
    L.append("## Gate G1: uncompressed clips (MAE <= 6 bpm and r >= 0.8)\n")
    L.append("| method | windows | MAE | Pearson r | gated | result |")
    L.append("|---|---|---|---|---|---|")
    src = df[df["mode"] == "source"]
    for m in METHODS:
        x = src[src.method == m]
        if not len(x):
            continue
        mae = x.err.mean()
        r = float(np.corrcoef(x.hr_est, x.hr_ref)[0, 1]) if len(x) > 2 else np.nan
        gated = m in DEEP
        res = ""
        if gated:
            ok = bool(mae <= 6 and r >= 0.8)
            res = "PASS" if ok else "FAIL"
            if not ok:
                blocked.append(f"G1 {m}: MAE {mae:.2f}, r {r:.3f}")
        L.append(f"| {m} | {len(x)} | {mae:.2f} | {r:.3f} | {'yes' if gated else 'context'} | {res} |")
    missing = [m for m in DEEP if not len(src[src.method == m])]
    if missing:
        blocked.append(f"G1: no source rows for {missing}")

    # ---- gate G2
    status, p = read_md5(a.inp)
    L.append("\n## Gate G2: deterministic encodes (md5check)\n")
    if status is None:
        L.append("NOT RUN (no md5.txt in --in).")
        if not a.smoke:
            blocked.append("G2: NOT RUN")
    else:
        L.append(f"{status} ({p})")
        if status != "PASS" and not a.smoke:
            blocked.append(f"G2: {status}")
    if a.smoke:
        L.append("(--smoke: G2 not required)")
    if len(subjects) < N_SUBJECTS:
        L.append(f"\nSubject count {len(subjects)} < {N_SUBJECTS}.")
        if not a.smoke:
            blocked.append(f"only {len(subjects)} of {N_SUBJECTS} subjects")

    if blocked:
        L.append("\n## VERDICT BLOCKED\n")
        L += [f"- {b}" for b in blocked]
        if not a.smoke:
            (out / "report.md").write_text("\n".join(L) + "\n")
            print("\n".join(L))
            return 1
        L.append("\n(--smoke: continuing for inspection only; this is not a verdict)")

    # ---- the rule, per method
    for rate_k in (3200, 800, 400):
        gate = rate_k in GATED_RATES
        L.append(f"\n## {rate_k}k{'' if gate else ' (reported, not gated)'}\n")
        d = df[df.rate_k == rate_k]
        for m in METHODS:
            x = d[d.method == m]
            if not len(x):
                continue
            L.append(f"### {m} @ {rate_k}k\n")
            L.append("| mode | windows | median err | share > 5 bpm | harmonic share (<= 1.5 bpm) | chance | bootstrap 95% CI of median(mode) - median(long) |")
            L.append("|---|---|---|---|---|---|---|")
            by = {md: {s: g.err.to_numpy() for s, g in x[x["mode"] == md].groupby("subject")}
                  for md in MODES}
            for md in MODES:
                y = x[x["mode"] == md]
                if not len(y):
                    continue
                hs = ch = ci = "-"
                if md.startswith("g") and y.f0.notna().any():
                    yy = y[y.f0.notna()]
                    hs = f"{100 * (yy.harm_dist <= HARM_TOL).mean():.0f}%"
                    ch = f"{100 * chance_share(yy):.0f}%"
                if md != "long" and by["long"]:
                    lo, hi = bootstrap_diff(by[md], by["long"], subjects)
                    ci = f"[{lo:.1f}, {hi:.1f}]"
                L.append(f"| {md} | {len(y)} | {y.err.median():.1f} | "
                         f"{100 * (y.err > 5).mean():.0f}% | {hs} | {ch} | {ci} |")
            L.append("\nPer-subject median err (bpm), rows = subject:\n")
            piv = x.groupby(["subject", "mode"]).err.median().unstack("mode")
            piv = piv[[md for md in MODES if md in piv.columns]]
            L.append("| subject | " + " | ".join(piv.columns) + " |")
            L.append("|---|" + "---|" * len(piv.columns))
            for sid, row in piv.iterrows():
                L.append(f"| {sid} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
            L.append("")
            long = x[x["mode"] == "long"]
            for md in ("g60", "g30", "g120"):
                y = x[(x["mode"] == md) & x.f0.notna()]
                if not len(y) or not len(long):
                    continue
                share = float((y.harm_dist <= HARM_TOL).mean())
                chance = chance_share(y)
                src_med = src[src.method == m].err.median()
                if gate and long.err.median() > src_med + BASELINE_TOL:
                    L.append(f"VERDICT {m} {rate_k}k {md}: **BASELINE FAILS, no verdict** "
                             f"(long-GOP median err {long.err.median():.1f} vs uncompressed "
                             f"{src_med:.1f}; healthy needs <= +{BASELINE_TOL:g}). "
                             f"{md} median err {y.err.median():.1f}.\n")
                    continue
                v, ratio = verdict(y.err.median(), long.err.median(), share, chance)
                tag = ("GATED" if gate and md == "g60" and m in DEEP
                       else "reference method, not a deep-model gate" if gate and md == "g60"
                       else "reported, not gated")
                L.append(f"VERDICT {m} {rate_k}k {md}: **{v}** ({tag}). "
                         f"median err {md} {y.err.median():.1f} vs long {long.err.median():.1f} "
                         f"(diff {y.err.median() - long.err.median():+.1f}; Fooled >= +10, Immune <= +3); "
                         f"harmonic share {100 * share:.0f}% vs chance {100 * chance:.0f}% "
                         f"(ratio {ratio:.2f}; Fooled needs >= 2).\n")

    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--shard", type=int, required=True)
    e.add_argument("--of", type=int, required=True)
    e.add_argument("--subjects", help="comma list of subject numbers (before sharding)")
    e.add_argument("--rates", default=",".join(RATES))
    e.add_argument("--modes", default=",".join(MODES))
    e.add_argument("--toolbox", default="rPPG-Toolbox", help="clone of ubicomplab/rPPG-Toolbox")
    e.add_argument("--tmp", help="scratch dir for encodes (default: system temp)")

    m = sub.add_parser("md5check")
    m.add_argument("--data", required=True)
    m.add_argument("--subject", required=True, help="subject number (or comma list)")
    m.add_argument("--out", required=True, help="dir that receives md5.txt")
    m.add_argument("--rate", default="800k")
    m.add_argument("--modes", default="long,g60,intra-refresh")
    m.add_argument("--tmp")

    n = sub.add_parser("analyse")
    n.add_argument("--in", dest="inp", nargs="+", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--smoke", action="store_true",
                   help="local smoke only: skip the 42-subject and G2 requirements")

    a = ap.parse_args()
    if a.cmd == "extract":
        cmd_extract(a)
    elif a.cmd == "md5check":
        sys.exit(cmd_md5check(a))
    else:
        sys.exit(cmd_analyse(a))


if __name__ == "__main__":
    main()
