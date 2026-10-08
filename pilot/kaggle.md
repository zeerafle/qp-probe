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

## Deep-model keyframe test

Pre-registered rule and gates are in the docstring of `pilot/deep.py`. Three
notebooks (`K` = 0, 1, 2), CPU, internet ON, same dataset input as above, and
the same pinned `<commit>` of `zeerafle/qp-probe` (the one containing
`pilot/deep.py`). The weights (`PURE_TSCAN.pth`,
`PURE_PhysNet_DiffNormalized.pth`) are downloaded by `deep.py` from the pinned
toolbox commit into the clone, so no extra input is needed. No GPU.

Cell 1, setup and layout check:

```python
!ls /kaggle/input
!find /kaggle/input -maxdepth 3 -name "subject1" | head
```

Cell 2, build tools and Python dependencies. The toolbox's `data_loader`
package imports every dataset loader at import time, hence the extra
packages; the full `requirements.txt` is not needed.

```bash
%%bash
pkg-config --exists libavformat libavcodec libavutil || \
  (apt-get update && apt-get install -y libavformat-dev libavcodec-dev libavutil-dev pkg-config)
python -c "import cv2; cv2.CascadeClassifier" 2>/dev/null || pip install "opencv-python-headless<5"
python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-image h5py mat73 neurokit2 matplotlib tqdm scikit-learn
```

Cell 3, code at pinned commits (this repo, then the toolbox):

```bash
%%bash
cd /kaggle/working
git clone https://github.com/zeerafle/qp-probe qp-probe
cd qp-probe && git checkout <commit> && make qpprobe
cd /kaggle/working
git clone https://github.com/ubicomplab/rPPG-Toolbox rPPG-Toolbox
cd rPPG-Toolbox && git checkout <toolbox-commit>
```

Cell 4, gate G2 (once, in one notebook; stop if it prints FAIL). Re-encodes
one clip twice per mode with `threads=1` and compares md5; writes
`/kaggle/working/md5.txt`, which `analyse` reads.

```bash
%%bash
cd /kaggle/working/qp-probe
mkdir -p /kaggle/temp
python pilot/deep.py md5check --data /kaggle/input/ubfc-rppg-dataset \
  --subject 1 --out /kaggle/working --tmp /kaggle/temp
```

Cell 5, extraction. Set `K` to this notebook's shard (0, 1 or 2). Resumable: if
the session dies, rerun cells 2, 3 and 5 with the same `--out` (partial results
live in `/kaggle/working/partial-<K>of3`). Roughly 2-3 h per shard.

```bash
%%bash
cd /kaggle/working/qp-probe
mkdir -p /kaggle/temp
K=0
python pilot/deep.py extract --shard $K --of 3 \
  --data /kaggle/input/ubfc-rppg-dataset --out /kaggle/working \
  --toolbox /kaggle/working/rPPG-Toolbox --tmp /kaggle/temp
```

Outputs in `/kaggle/working`: `features-<K>of3.csv`, `env-<K>of3.json`, and
`md5.txt` from the notebook that ran cell 4. Save a version so the outputs can
be attached to the analysis notebook.

Analysis notebook (attach the three extraction outputs; `md5.txt` must be among
them). Exits 1 with the reason if G1 or G2 fails or fewer than 42 subjects are
present, and prints no verdict in that case:

```bash
%%bash
cd /kaggle/working/qp-probe
ls /kaggle/input
python pilot/deep.py analyse \
  --in /kaggle/input/<deep-notebook-0> /kaggle/input/<deep-notebook-1> /kaggle/input/<deep-notebook-2> \
  --out /kaggle/working/report
```

Copy `report/report.md` and `report/windows.csv` to
`Proposal/experiments/deep-keyframe-<date>/`.
