# Kaggle notebooks for the C1 pilot

All notebooks: CPU, internet ON. Add the dataset `malekdinarito/ubfc-rppg-dataset`
as an input. Its mount layout is not known in advance, so `--data` takes any
directory and searches `subject*/vid.avi` recursively; the first cell prints
what it finds. Replace `<commit>` with the pinned commit hash of
`zeerafle/qp-probe` (the same in every notebook).

## Extraction notebooks (run three, `K` = 0, 1, 2)

Cell 1, setup and layout check:

```python
!ls /kaggle/input
!find /kaggle/input -maxdepth 3 -name "subject1" | head
```

Cell 2, build tools (only if headers are missing):

```bash
%%bash
pkg-config --exists libavformat libavcodec libavutil || \
  (apt-get update && apt-get install -y libavformat-dev libavcodec-dev libavutil-dev pkg-config)
python -c "import cv2; cv2.CascadeClassifier" 2>/dev/null || pip install "opencv-python-headless<5"
python -c "import pyarrow" 2>/dev/null || pip install pyarrow   # optional; extract.py falls back to csv
```

Cell 3, code at a pinned commit:

```bash
%%bash
cd /kaggle/working
git clone https://github.com/zeerafle/qp-probe qp-probe
cd qp-probe && git checkout <commit> && make qpprobe
```

Cell 4, gate P2 (once per notebook; stop if it prints FAIL):

```bash
%%bash
cd /kaggle/working/qp-probe
TMP=/kaggle/temp pilot/p2_check.sh /kaggle/input/ubfc-rppg-dataset
```

Cell 5, extraction. Set `K` to this notebook's shard (0, 1 or 2). Resumable: if
the session dies, rerun cells 2, 3 and 5 with the same `--out`
(partial results live in `/kaggle/working/partial-<K>of3`).

```bash
%%bash
cd /kaggle/working/qp-probe
mkdir -p /kaggle/temp
K=0
python pilot/extract.py --shard $K --of 3 \
  --data /kaggle/input/ubfc-rppg-dataset --out /kaggle/working --tmp /kaggle/temp
```

Outputs in `/kaggle/working`: `features-<K>of3.parquet` (or `.csv`),
`qp-<K>of3.csv.gz`, `env-<K>of3.json`. Save a version of the notebook so the
outputs can be attached to the analysis notebook.

## Analysis notebook

Add the three extraction notebooks' outputs as inputs (Add Input -> Notebook
Output). Cell 1 and 3 as above (clone, `make` not required); then:

```bash
%%bash
pip install scikit-learn pandas scipy matplotlib 2>&1 | tail -1
cd /kaggle/working
git clone https://github.com/zeerafle/qp-probe qp-probe
cd qp-probe && git checkout <commit>
ls /kaggle/input
python pilot/analyse.py \
  --in /kaggle/input/<extract-notebook-0> /kaggle/input/<extract-notebook-1> /kaggle/input/<extract-notebook-2> \
  --out /kaggle/working/report
```

Copy `report/report.md`, `report/folds.csv` and the PNGs to
`Proposal/experiments/c1-pilot/`. The script exits with status 1 and a clear
message if gate P0, P1 or P3 fails; it does not fit in that case.
