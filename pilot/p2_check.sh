#!/usr/bin/env bash
# Gate P2 (PLAN.md section 8): does THIS machine's ffmpeg/x264 reproduce the
# plan section 10.1.1 whole-frame mean QP table?
#
#   pilot/p2_check.sh [data_dir]      (default: ubfc-rppg, searched recursively)
#
# Subjects 1/10/20/30/40, aq=1, the sweep.sh recipe (full-length encode, then
# qpprobe's default first 300 frames), five-subject mean QP per rate against
# 24.5 / 20.6 / 18.1 / 15.7 within 0.5. The tolerance is 0.5 rather than the 0.3
# used locally because Kaggle's ffmpeg is a different version.
#
# Env: TMP=<scratch dir for encodes>   PY=<python with cv2 for facebox.py>

set -u
cd "$(dirname "$0")/.."
DATA=${1:-ubfc-rppg}
TMP=${TMP:-$(mktemp -d)}
PY=${PY:-python3}
RATES="200k 400k 800k 1600k"
REF=(24.5 20.6 18.1 15.7)
TOL=0.5
SUBJECTS="1 10 20 30 40"

. ./ensure-build.sh
mkdir -p "$TMP"

declare -A SUM
for r in $RATES; do SUM[$r]=0; done
n=0
for s in $SUBJECTS; do
  src=$(find "$DATA" -path "*subject$s/vid.avi" | head -1)
  [ -n "$src" ] || { echo "subject$s: vid.avi not found under $DATA"; exit 1; }
  roi=$($PY facebox.py "$src" | cut -f2)
  [ -n "$roi" ] || { echo "subject$s: facebox failed"; exit 1; }
  for br in $RATES; do
    f="$TMP/p2-s$s-$br.mp4"
    ffmpeg -y -loglevel error -i "$src" -c:v libx264 \
      -b:v "$br" -minrate "$br" -maxrate "$br" -bufsize "$br" \
      -tune zerolatency -g 60 -x264-params "aq-mode=1" "$f" || { echo "encode failed: s$s $br"; exit 1; }
    qp=$(./qpprobe "$f" --roi "$roi" | sed -n 's/.*mean QP (whole frame) *: *\([0-9.]*\).*/\1/p')
    rm -f "$f"
    [ -n "$qp" ] || { echo "no QP for s$s $br"; exit 1; }
    SUM[$br]=$(awk -v a="${SUM[$br]}" -v b="$qp" 'BEGIN{print a+b}')
  done
  n=$((n+1))
  echo "subject$s done" >&2
done

echo
echo "ffmpeg: $(ffmpeg -version | head -1)"
printf '%-8s %-10s %-10s %-8s %s\n' rate measured plan diff verdict
fail=0
i=0
for br in $RATES; do
  m=$(awk -v a="${SUM[$br]}" -v n="$n" 'BEGIN{printf "%.2f", a/n}')
  ref=${REF[$i]}
  d=$(awk -v m="$m" -v r="$ref" 'BEGIN{printf "%+.2f", m-r}')
  ok=$(awk -v d="$d" -v t="$TOL" 'BEGIN{print (d<0?-d:d)<=t ? "ok" : "OUT"}')
  [ "$ok" = ok ] || fail=1
  printf '%-8s %-10s %-10s %-8s %s\n' "$br" "$m" "$ref" "$d" "$ok"
  i=$((i+1))
done
echo
[ $fail -eq 0 ] && echo "P2: PASS (all within $TOL)" || echo "P2: FAIL (see OUT rows)"
exit $fail
