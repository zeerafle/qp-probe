#!/usr/bin/env bash
# Redundancy test for plan C2: does the spatial QP map carry anything the
# decoded pixels (plus the frame-level QP scalar) do not?
#
#   ./redundancy.sh [--gates-only]
#
# --gates-only: subject 1, 3200k + 200k, aq=1 and aq=0, gates G1-G3, no models.
# Run it first; a failed gate means the full run would be uninterpretable.
#
# Env overrides:  OUT=out  VENV=<venv dir>  CACHE=<feature cache dir>
# The venv and cache live outside this directory on purpose: `make push`
# uploads it, and the cache is hundreds of MB.

set -u
cd "$(dirname "$0")"
OUT=${OUT:-out}
SCR=/tmp/claude-1000/-home-zeerafle-WorkFolder-Documents-College-3rd-Semester-Thesis/0decb12f-a827-4268-8219-5bf3015b633e/scratchpad
VENV=${VENV:-$SCR/venv}
CACHE=${CACHE:-$(dirname "$VENV")/redcache}

. ./ensure-build.sh

mkdir -p "$OUT" "$CACHE"   # after the build: `make clean` removes out/

GATES=0
[ "${1:-}" = "--gates-only" ] && GATES=1

if [ $GATES = 1 ]; then
  SUBJECTS="1"; RATES="200k 3200k"
else
  SUBJECTS="1 10 20 30 40"; RATES="200k 400k 800k 1600k 3200k"
fi

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV" >/dev/null || exit 1
  uv pip install -q --python "$VENV/bin/python" numpy scikit-learn "opencv-python-headless<5" || exit 1
fi
PY="$VENV/bin/python"

# Face ROI per subject, cached: the cascade pass is slow and deterministic.
BOXES="$OUT/boxes.tsv"
touch "$BOXES"
for s in $SUBJECTS; do
  v="ubfc-rppg/subject$s/vid.avi"
  grep -q "^$v	" "$BOXES" && continue
  line=$("$PY" facebox.py "$v") || { echo "facebox failed for $v"; exit 1; }
  printf '%s\t%s\n' "$v" "$(cut -f2 <<<"$line")" >> "$BOXES"
done

for s in $SUBJECTS; do
  v="ubfc-rppg/subject$s/vid.avi"
  box=$(grep "^$v	" "$BOXES" | cut -f2)
  for r in $RATES; do
    for aq in 1 0; do
      f="$OUT/red-s$s-$r-aq$aq.mp4"
      c="$OUT/red-s$s-$r-aq$aq.csv"
      if [ ! -s "$f" ]; then
        # Same flags as sweep.sh, plus the 300-frame cap qpprobe applies anyway.
        ffmpeg -y -loglevel error -i "$v" -frames:v 300 -c:v libx264 \
          -b:v "$r" -minrate "$r" -maxrate "$r" -bufsize "$r" \
          -tune zerolatency -g 60 -x264-params "aq-mode=$aq" "$f" \
          || { echo "encode failed: $f"; rm -f "$f"; exit 1; }
      fi
      [ -s "$c" ] || ./qpprobe "$f" --csv --roi "$box" > "$c" || { rm -f "$c"; exit 1; }
    done
  done
done

if [ $GATES = 1 ]; then
  exec "$PY" redundancy.py --gates-only --out "$OUT" --cache "$CACHE"
else
  exec "$PY" redundancy.py --out "$OUT" --cache "$CACHE"
fi
