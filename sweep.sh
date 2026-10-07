#!/usr/bin/env bash
# Does the ROI-background QP gap move with bitrate, and does it survive aq-mode=0?
#
# Follow-up to run.sh, which answered only half of the plan §10.1 pass
# condition: the gap is non-zero at one bitrate with x264's default AQ on.
# "Moving with bitrate" is the other half, and a gap that exists only because
# aq-mode defaults to 1 is a gap the plan cannot assume.
#
#   ./sweep.sh <source> <roi_x0,y0,x1,y1>
#
# H.264 CBR only -- it is the one cell of the run.sh matrix that PASSes, so it
# is the only place a sweep can tell you anything.
#
# Env overrides:  RATES="200k 400k 800k 1600k 3200k"  OUT=out

set -u
cd "$(dirname "$0")"
OUT=${OUT:-out}

. ./ensure-build.sh

mkdir -p "$OUT"   # after the build: `make clean` removes out/

SRC=${1:-}
ROI=${2:-}
[ -n "$SRC" ] && [ -r "$SRC" ] || { echo "usage: ./sweep.sh <source> <x0,y0,x1,y1>"; exit 1; }
[ -n "$ROI" ] || { echo "a real face ROI is the whole point of this script"; exit 1; }

RATES=${RATES:-"200k 400k 800k 1600k 3200k"}

cell() {
  local br=$1 aq=$2
  local f="$OUT/sweep-$br-aq$aq.mp4"
  ffmpeg -y -loglevel error -i "$SRC" -c:v libx264 \
    -b:v "$br" -minrate "$br" -maxrate "$br" -bufsize "$br" \
    -tune zerolatency -g 60 -x264-params "aq-mode=$aq" \
    "$f" 2>"$OUT/sweep-$br-aq$aq.err" || { printf '%-8s %-4s %s\n' "$br" "$aq" "ENCODE FAILED"; return; }

  local report qp roiqp roispr roibg
  report=$(./qpprobe "$f" --roi "$ROI")
  qp=$(sed    -n 's/.*mean QP (whole frame) *: *\([0-9.]*\).*/\1/p'      <<<"$report")
  roiqp=$(sed -n 's/.*mean QP inside ROI *: *\([0-9.]*\).*/\1/p'         <<<"$report")
  roispr=$(sed -n 's/.*mean within-ROI spread *: *\([0-9.]*\).*/\1/p'    <<<"$report")
  roibg=$(sed -n 's/.*mean ROI-background QP *: *\([+-][0-9.]*\).*/\1/p' <<<"$report")
  printf '%-8s %-4s %-7s %-8s %-9s %s\n' \
    "$br" "$aq" "${qp:-n/a}" "${roiqp:-n/a}" "${roispr:-n/a}" "${roibg:-NOT MEASURED}"
}

echo
echo "source: $SRC"
echo "roi: $ROI"
echo "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
echo "============================================================"
printf '%-8s %-4s %-7s %-8s %-9s %s\n' rate aq qp roi-qp roi-spr roi-bg
echo "------------------------------------------------------------"
for br in $RATES; do cell "$br" 1; done
echo "------------------------------------------------------------"
for br in $RATES; do cell "$br" 0; done

cat <<'TXT'

============================================================
Reading the result
  aq=1 rows are the default encoder. aq=0 turns adaptive quantisation off.

  roi-spr is the one that matters. A cropped rPPG model never sees the
  background, so roi-bg is a constant to it; the variation INSIDE the face is
  the only spatial signal C2 can condition on.

  roi-spr large and rate-dependent -> C2 has something to learn (premise holds)
  roi-spr ~0                       -> the face is quantised uniformly; the map
                                      degenerates to the scalar VP9 already
                                      gives, and C2 collapses into C1
  roi-spr ~0 only at aq=0          -> signal exists but is AQ's doing; the plan
                                      must declare the dependency, since an
                                      encoder with AQ off emits nothing usable

  qp vs roi-qp shows whether the face is being punished relative to the frame.
  Interesting for motivation (§6.3), but not a conditioning feature.
TXT
