#!/usr/bin/env python3
"""Rate-control control for the keyframe artefact (supervisor review, Oct 2026).

    rc.py --data DIR --out DIR [--shard K --of N] [--subjects 1,10] [--tmp DIR] [--jobs 4]

Is the keyframe error the I-frame jumps themselves, or CBR starving the frames
between keyframes of bits? Same x264 recipe as deep.py (zerolatency, aq-mode=1,
threads=1); only the rate control varies: CBR 800k (the pilot), constant QP 18
and 24, CRF 23 and 28. Modes long, g30, g60, g120.

Saves per-frame traces, one npz per subject, in the keyframe.py layout with the
condition as `<rc>_<mode>` (rgb_, isI_, qroi_, qmean_, pkt_, kbps_), so any
receiver-side method can be run later without re-encoding. Analysis and the
decision rule (committed before the first run): rppg-f0, README section
"Rate-control control".
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import extract as E   # noqa: E402
import facebox        # noqa: E402

RCS = {
    "cbr800": ["-b:v", "800k", "-minrate", "800k", "-maxrate", "800k", "-bufsize", "800k"],
    "qp18": ["-qp", "18"],
    "qp24": ["-qp", "24"],
    "crf23": ["-crf", "23"],
    "crf28": ["-crf", "28"],
}
GOPS = {"long": "9999", "g30": "30", "g60": "60", "g120": "120"}


def encode(src, dst, rc, mode):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", *RCS[rc],
         "-tune", "zerolatency", "-g", GOPS[mode], "-x264-params", "aq-mode=1:threads=1", str(dst)],
        check=True)


def pkt_sizes(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pkt_size",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return np.array([int(x.strip(",")) for x in out.split()], float)


def condition(src, sid, rc, mode, box, w, h, fps, tmp):
    p = tmp / f"s{sid}-{rc}-{mode}.mp4"
    try:
        encode(src, p, rc, mode)
        q = E.qp_summary(p, box)
        rgb = E.stream_features(p, box, w, h)["rgb"]
        pkt = pkt_sizes(p)
        kbps = p.stat().st_size * 8 / 1000 / (len(rgb) / fps)
    finally:
        p.unlink(missing_ok=True)
    return {"rgb": rgb, "isI": (q.pict_type == "I").to_numpy(), "qroi": q.roi_mean.to_numpy(float),
            "qmean": q.mean_qp.to_numpy(float), "pkt": pkt, "kbps": kbps}


def run_subject(sid, sdir, out, tmp, jobs):
    t0 = time.time()
    src = sdir / "vid.avi"
    w, h, fps = E.probe_video(src)
    box, note = facebox.face_box(str(src))
    if box is None:
        raise RuntimeError(f"subject{sid}: face box failed - {note}")
    arrs = {"fps": fps, "box": np.array(box)}
    arrs["rgb_source"] = E.stream_features(src, box, w, h)["rgb"]
    n = len(arrs["rgb_source"])
    conds = [(rc, m) for rc in RCS for m in GOPS]
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(lambda c: condition(src, sid, *c, box, w, h, fps, tmp), conds))
    for (rc, m), r in zip(conds, res):
        c = f"{rc}_{m}"
        assert len(r["rgb"]) == n and len(r["isI"]) == n, (sid, c, len(r["rgb"]), len(r["isI"]), n)
        for k, v in r.items():
            arrs[f"{k}_{c}"] = v
        print(f"  s{sid} {c:<12} I-frames {int(r['isI'].sum()):3d}  {r['kbps']:7.0f} kbps", flush=True)
    arrs["ref"], arrs["usable"] = E.load_reference(sdir / "ground_truth.txt", fps, n)
    tmpf = out / f"s{sid}.tmp.npz"
    np.savez_compressed(tmpf, **arrs)
    tmpf.rename(out / f"s{sid}.npz")
    print(f"subject{sid} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--subjects")
    ap.add_argument("--tmp")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if not E.QPPROBE.exists():
        sys.exit(f"{E.QPPROBE} missing: run `make qpprobe` first")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subs = E.find_subjects(a.data)
    if a.subjects:
        want = {int(s) for s in a.subjects.split(",")}
        subs = {k: v for k, v in subs.items() if k in want}
    if not subs:
        sys.exit(f"no subject*/vid.avi with ground_truth.txt under {a.data}")
    mine = list(subs.items())[a.shard::a.of]
    tag = f"{a.shard}of{a.of}"
    print(f"shard {tag}: subjects {[k for k, _ in mine]} of {list(subs)}", flush=True)
    env = E.env_info(tag)
    env.update(rcs=RCS, gops=GOPS, x264_params="aq-mode=1:threads=1", tune="zerolatency")
    (out / f"env-{tag}.json").write_text(json.dumps(env, indent=2))
    tmp = pathlib.Path(a.tmp) if a.tmp else pathlib.Path(tempfile.mkdtemp(prefix="rc-"))
    tmp.mkdir(parents=True, exist_ok=True)
    for sid, sdir in mine:
        if (out / f"s{sid}.npz").exists():
            print(f"subject{sid}: done, skipping", flush=True)
            continue
        print(f"subject{sid} ({sdir})", flush=True)
        run_subject(sid, sdir, out, tmp, a.jobs)


if __name__ == "__main__":
    main()
