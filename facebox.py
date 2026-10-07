#!/usr/bin/env python3
"""Derive a per-subject face ROI for sweep.sh / qpprobe --roi.

    facebox.py <video> [video ...]

Prints one "<path>  <x0,y0,x1,y1>" line per video, ready to paste after
./sweep.sh. Reusing one hand-drawn box across subjects does not work: the box
in plan v3 §10.1.1 was drawn for one clip, and a box that misses the face makes
the within-ROI spread meaningless rather than obviously wrong.

Median over sampled frames, not per-frame tracking -- UBFC-rPPG is a fixed
camera with a seated subject, so one box for the clip is the honest summary and
a bad detection cannot drag the median.

Frames come from the system ffmpeg, not cv2.VideoCapture: OpenCV's bundled
FFmpeg segfaults on UBFC's rawvideo/bgr24 AVIs. cv2 is only the cascade here.

The snap to 16 is not cosmetic: qpprobe marks blocks whose CENTRE falls inside
the box, so an unaligned ROI half-includes macroblocks along every edge and
dilutes the spread statistic C2 is being judged on.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np

SAMPLES = 40  # frames probed per clip
MB = 16       # H.264 macroblock size, and so the ROI quantum
MIN_FACE = 96 # px; UBFC faces fill much of a 640x480 frame, so this only cuts noise


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_frames"])


def sample_frames(path, step, dest):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-vf", rf"select=not(mod(n\,{step}))",
         "-fps_mode", "passthrough", str(dest / "f%04d.png")],
        check=True)
    return sorted(dest.glob("*.png"))


def face_box(path):
    w, h, n = probe(path)
    if n <= 0:
        return None, "no frames"
    step = max(1, n // SAMPLES)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    hits = []
    with tempfile.TemporaryDirectory() as tmp:
        frames = sample_frames(path, step, pathlib.Path(tmp))
        for f in frames:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            found = cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5,
                minSize=(MIN_FACE, MIN_FACE))
            if len(found):
                # Largest detection: cascades occasionally fire on a hand or a
                # background edge, never on something bigger than the face.
                hits.append(max(found, key=lambda r: r[2] * r[3]))
        total = len(frames)

    if not total:
        return None, "ffmpeg extracted no frames"
    if len(hits) < total // 4:
        return None, f"only {len(hits)}/{total} frames detected"

    x, y, bw, bh = np.median(np.array(hits), axis=0)
    x0 = max(0, int(x) // MB * MB)
    y0 = max(0, int(y) // MB * MB)
    x1 = min(w, -(-int(x + bw) // MB) * MB)
    y1 = min(h, -(-int(y + bh) // MB) * MB)
    return (x0, y0, x1, y1), f"{len(hits)}/{total} frames"


def main(argv):
    if not argv:
        print("usage: facebox.py <video> [video ...]", file=sys.stderr)
        return 2
    rc = 0
    for path in argv:
        try:
            box, note = face_box(path)
        except subprocess.CalledProcessError as e:
            box, note = None, f"ffmpeg/ffprobe failed ({e.returncode})"
        if box is None:
            print(f"{path}\tFAILED - {note}", file=sys.stderr)
            rc = 1
        else:
            print(f"{path}\t{box[0]},{box[1]},{box[2]},{box[3]}\t({note})",
                  flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
