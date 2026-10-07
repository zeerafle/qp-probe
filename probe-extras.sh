#!/usr/bin/env bash
# Two follow-ups to §10.1 that run.sh does not cover.
#
#   ./probe-extras.sh <source> <roi_x0,y0,x1,y1>
#
# 1. DECODER-AGE HYPOTHESIS. §10.1 finding 3 says H.265/AV1 expose no QP map.
#    The proposed explanation is that libavcodec's export tracks decoder
#    codebase age, not codec capability: h264 and the mpegvideo family inherited
#    it from the deprecated AVFrame.qscale_table, and decoders written after that
#    era never got it. If true, mpeg2video and mpeg4 -- older and far less
#    capable than HEVC -- must PASS where HEVC fails. If they fail too, the
#    explanation is wrong and §10.1 should stop asserting it.
#
# 2. VP9 CONFOUND. §10.1 records VP9 as a frame-level scalar with nb_blocks=0
#    and attributes that to the decoder. But VP9's only spatial QP mechanism is
#    segmentation, and libvpx leaves aq-mode=0 by default -- so the encoder may
#    simply not have written any spatial variation to find. Forcing aq-mode=3
#    separates "decoder will not export it" from "encoder never encoded it".

set -u
cd "$(dirname "$0")"
OUT=${OUT:-out}

. ./ensure-build.sh
mkdir -p "$OUT"

SRC=${1:-}
ROI=${2:-}
[ -n "$SRC" ] && [ -r "$SRC" ] || { echo "usage: ./probe-extras.sh <source> <x0,y0,x1,y1>"; exit 1; }
ROIARG=""; [ -n "$ROI" ] && ROIARG="--roi $ROI"

# Short clips: these questions are about whether side data appears at all, which
# the first GOP settles. The bitrate grid is sweep.sh's job.
FRAMES=${FRAMES:-150}

cell() {
  local label=$1; shift
  local f="$OUT/x-$label.${EXT:-mkv}"
  if ! ffmpeg -y -loglevel error -i "$SRC" -frames:v "$FRAMES" "$@" "$f" \
       2>"$OUT/x-$label.err"; then
    printf '%-26s %s\n' "$label" "ENCODE FAILED (see $OUT/x-$label.err)"; return
  fi
  local report
  report=$(./qpprobe "$f" $ROIARG 2>&1)
  printf '%-26s %s\n' "$label" \
    "$(sed -n 's/.*VERDICT *: //p' <<<"$report" | head -1)"
  # One sample frame line, so a PASS/FAIL can be read back to nb_blocks.
  sed -n 's/^  \(frame .*nb_blocks=.*\)/      \1/p' <<<"$report" | sed -n 2p
}

echo
echo "source: $SRC   frames: $FRAMES   roi: ${ROI:-none}"
echo "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
echo "libavcodec: $(pkg-config --modversion libavcodec 2>/dev/null)"
echo "==========================================================="
echo "1. decoder-age hypothesis -- older codecs should PASS"
EXT=mkv cell "mpeg2video"  -c:v mpeg2video -b:v 800k
EXT=mkv cell "mpeg4"       -c:v mpeg4      -b:v 800k
EXT=mkv cell "h264 (control)" -c:v libx264 -b:v 800k

echo
echo "2. VP9 -- is nb_blocks=0 the decoder or the encoder?"
EXT=webm cell "vp9 aq-mode default" -c:v libvpx-vp9 -b:v 800k
EXT=webm cell "vp9 aq-mode=3"       -c:v libvpx-vp9 -b:v 800k -aq-mode 3

cat <<'TXT'

===========================================================
Reading the result
  mpeg2/mpeg4 PASS, hevc FAIL  -> decoder-age hypothesis holds; §10.1 may state
                                  it as the reason rather than a guess
  mpeg2/mpeg4 FAIL             -> hypothesis is wrong; the H.265 gap needs a
                                  different explanation and §10.1 should say
                                  only "not exported", with no cause attached

  vp9 aq-mode=3 still scalar   -> genuinely a decoder limit, §10.1 stands
  vp9 aq-mode=3 gives blocks   -> §10.1's VP9 row was measuring an encoder
                                  default, not a decoder limit, and VP9 may be
                                  a second C2 codec after all
TXT
