#!/usr/bin/env bash
# Week-1 go/no-go for rppg_codec_research_plan_v3.md §10.
#
# Encodes one source across the planned encoders x {CRF-ish, CBR} and asks each
# decoder for a quantisation map. Prints the matrix the plan needs.
#
#   ./run.sh <source> [roi_x0,y0,x1,y1]
#
# <source> should be a NEAR-LOSSLESS clip -- a UBFC-rPPG .avi or PURE PNG
# sequence. Passing an already-compressed file measures double compression and
# tells you nothing (plan §8, "Hard constraint").
#
# Without a source it falls back to a synthetic clip: a static flat "skin" patch
# next to a moving block. That proves what the decoder exposes, NOT whether the
# map is informative over a real face -- only real video answers that.
#
# Env overrides:  BR=600k  OUT=/kaggle/working/out  FRAMES=300

set -u
cd "$(dirname "$0")"
OUT=${OUT:-out}

. ./ensure-build.sh

mkdir -p "$OUT"   # after the build: `make clean` removes out/

command -v ffmpeg >/dev/null || { echo "ffmpeg not on PATH"; exit 1; }
HAVE=$(ffmpeg -hide_banner -encoders 2>/dev/null)

SRC=${1:-}
ROI=${2:-}
if [ -z "$SRC" ]; then
  SRC=$OUT/synthetic.y4m
  echo "No source given -> synthetic fallback (decoder support only, not informativeness)."
  ffmpeg -y -loglevel error \
    -f lavfi -i "color=c=0x9a7060:size=320x240:rate=30:duration=4" \
    -f lavfi -i "testsrc2=size=96x96:rate=30:duration=4" \
    -filter_complex "[0:v][1:v]overlay=x='160+60*sin(t)':y=70" \
    -pix_fmt yuv420p "$SRC"
  ROI=${ROI:-0,0,120,240}   # the flat "skin" half
elif [ ! -r "$SRC" ]; then
  echo "cannot read source: $SRC"; exit 1
fi

BR=${BR:-600k}
ROIARG=""; [ -n "$ROI" ] && ROIARG="--roi $ROI"

run_one() {
  local enc=$1 label=$2 ext=$3; shift 3
  if ! grep -q " $enc " <<<"$HAVE"; then
    printf '%-22s %s\n' "$label" "skipped - $enc not in this ffmpeg build"; return
  fi
  local f="$OUT/$label.$ext"
  if ! ffmpeg -y -loglevel error -i "$SRC" -c:v "$enc" "$@" "$f" 2>"$OUT/$label.err"; then
    printf '%-22s %s\n' "$label" "ENCODE FAILED (see $OUT/$label.err)"; return
  fi
  local report
  report=$(./qpprobe "$f" $ROIARG)
  printf '%-22s %s\n' "$label" "$(sed -n 's/.*VERDICT *: //p' <<<"$report")"
  local extra
  extra=$(grep -E "mean within-frame spread|mean ROI-background" <<<"$report" | tr '\n' ' ' | tr -s ' ')
  [ -n "$extra" ] && printf '%-22s   %s\n' "" "$extra"
}

echo
echo "source: $SRC"
echo "roi: ${ROI:-none}   target bitrate: $BR"
echo "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
echo "============================================================"
echo "-- offline rate control (CRF / CQ), what all prior work used"
run_one libx264    h264-crf    mp4  -crf 28
run_one libx265    h265-crf    mp4  -crf 28
run_one libvpx-vp9 vp9-crf     webm -crf 40 -b:v 0
run_one libaom-av1 av1aom-crf  mp4  -crf 40 -b:v 0 -cpu-used 8
run_one libsvtav1  av1svt-crf  mp4  -crf 40

echo
echo "-- low-latency CBR, the regime the plan claims as novel (§6.4)"
run_one libx264    h264-cbr    mp4  -b:v $BR -minrate $BR -maxrate $BR -bufsize $BR -tune zerolatency -g 60
run_one libx265    h265-cbr    mp4  -b:v $BR -x265-params "vbv-maxrate=${BR%k}:vbv-bufsize=${BR%k}" -g 60
run_one libvpx-vp9 vp9-cbr     webm -b:v $BR -minrate $BR -maxrate $BR -lag-in-frames 0 -g 60
run_one libaom-av1 av1aom-cbr  mp4  -b:v $BR -minrate $BR -maxrate $BR -lag-in-frames 0 -cpu-used 8 -g 60
run_one libsvtav1  av1svt-cbr  mp4  -b:v $BR -svtav1-params "pred-struct=1:rc=2" -g 60

cat <<'EOF'

============================================================
Reading the result
  PASS    per-block map that varies spatially -> C1 and C2 both viable
  WEAK    per-block map but flat              -> C2 has no signal to condition on
  PARTIAL frame-level QP scalar only          -> C1 viable, C2 dead for that codec
  FAIL    decoder exposes nothing             -> both dead for that codec

Go/no-go (plan §10): C2 needs PASS on at least one codec the plan will ship.
If only H.264 passes, C1/C2 become H.264-only and the "modern codecs" framing
in §7.1 has to be cut to a C0-characterisation claim.

With a real face ROI, the number that matters is "mean ROI-background QP".
Consistently non-zero and moving with bitrate = the §6.3 mechanism is real.
Flat = C2 loses its premise even though extraction works.
EOF
