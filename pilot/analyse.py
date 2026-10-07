#!/usr/bin/env python3
"""C1 pilot, analysis: gates P0/P1/P3, nested-set models, decision rule.

    analyse.py --in <dir> [<dir> ...] --out <dir> [--smoke]

Implements PLAN.md sections 4, 5 and 8 (Proposal/experiments/c1-pilot). Gate
P2 (encoder reproduces the plan's QP table) is pilot/p2_check.sh, run on the
extraction machine, not here. The decision rule is applied mechanically at the
end; nothing below is tuned after seeing results.

--smoke is for tiny local runs only: it relaxes the fold count to the number of
subjects and does not abort on gate failures. Verdicts printed under --smoke
mean nothing.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

# ---- fixed by the plan; do not tune ------------------------------------
N_FOLDS = 7
FAIL_BPM = 5.0
RELIABLE_BPM, UNRELIABLE_BPM = 2.0, 6.0     # Arevalillo scheme
BOOT_N, BOOT_SEED = 1000, 0
C1_MIN_REL_GAIN = 0.05                      # AURC reduction needed (relative)
FREE_GATE_TOL = 0.10                        # Q-only vs S-only AURC
P0_MIN_FAIL_FRAC, P0_MIN_SUBJECTS = 0.15, 20
P1_MAX_MAE, P1_MIN_R = 6.0, 0.8
COVERAGES = np.linspace(1.0, 0.1, 91)       # 100% -> 10%
HGB = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
           early_stopping=False, random_state=0)
# Published number to compare P1 against: the rPPG-Toolbox UBFC-rPPG table
# (POS MAE ~4.0 bpm on its protocol). Not re-derived here; check the source.
TOOLBOX_NOTE = "rPPG-Toolbox UBFC-rPPG POS: MAE about 4 bpm (cite from their table)"

PREFIX = {"S": ["S_"], "M": ["M_"], "R": ["R_"], "P": ["P_"],
          "Q": ["Q_"], "Qmap": ["Qmap_"]}
SETS = {
    "S": ["S"],
    "S+M": ["S", "M"],
    "S+M+R": ["S", "M", "R"],
    "S+M+R+P": ["S", "M", "R", "P"],
    "S+M+R+P+Q": ["S", "M", "R", "P", "Q"],
    "S+M+R+P+Q+Qmap": ["S", "M", "R", "P", "Q", "Qmap"],
    "Q-only": ["Q", "Qmap", "R"],
}


# ------------------------------------------------------------------ data ----

def load(dirs):
    parts = []
    for d in dirs:
        d = pathlib.Path(d)
        files = sorted(d.glob("features-*.parquet")) + sorted(d.glob("features-*.csv"))
        for f in files:
            parts.append(pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f))
    if not parts:
        sys.exit(f"no features-*.parquet/csv under {dirs}")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["subject", "cond", "win_start"])
    return df


def columns(df, set_name):
    cols = []
    for g in SETS[set_name]:
        cols += [c for c in df.columns if any(c.startswith(p) for p in PREFIX[g])]
    return cols


def assign_folds(subjects, k):
    # Sorted, round-robin: deterministic, balanced (42 subjects -> 6 per fold),
    # independent of window counts, shared by every set and every target.
    subjects = sorted(subjects)
    return {s: i % k for i, s in enumerate(subjects)}


# ----------------------------------------------------------------- gates ----

def gates(df, smoke, log):
    ok = True
    comp = df[(df.rate_k > 0) & (df.aq == 1) & (df.gop == "long")]
    fail = comp.err > FAIL_BPM
    n_sub = comp.loc[fail, "subject"].nunique()
    frac = fail.mean()
    p0 = frac >= P0_MIN_FAIL_FRAC and n_sub >= P0_MIN_SUBJECTS
    log(f"P0 enough failures: {100 * frac:.1f}% of compressed POS windows have err > "
        f"{FAIL_BPM:g} bpm (need >= {100 * P0_MIN_FAIL_FRAC:g}%), from {n_sub} subjects "
        f"(need >= {P0_MIN_SUBJECTS}) -> {'PASS' if p0 else 'FAIL'}")

    src = df[df.rate_k == 0]
    per = []
    for s, g in src.groupby("subject"):
        r = pearsonr(g.hr_pos, g.hr_ref)[0] if len(g) > 2 and g.hr_ref.std() > 0 else np.nan
        per.append((s, (g.hr_pos - g.hr_ref).abs().mean(), r, len(g)))
    mae = (src.hr_pos - src.hr_ref).abs().mean()
    r_all = pearsonr(src.hr_pos, src.hr_ref)[0]
    p1 = mae <= P1_MAX_MAE and r_all >= P1_MIN_R
    log(f"P1 sane baseline (uncompressed POS): MAE {mae:.2f} bpm (need <= {P1_MAX_MAE:g}), "
        f"Pearson r {r_all:.3f} (need >= {P1_MIN_R:g}) -> {'PASS' if p1 else 'FAIL'}")
    log(f"   reference: {TOOLBOX_NOTE}")
    log("   per subject:  " + ", ".join(f"s{s}: MAE {m:.2f} r {r:.2f} (n={n})"
                                       for s, m, r, n in per))

    rates = sorted(comp.rate_k.unique(), reverse=True)     # high rate first
    med = [comp[comp.rate_k == r].err.median() for r in rates]
    p3 = all(b >= a - 1e-12 for a, b in zip(med, med[1:]))
    log("P3 monotone damage: median POS err by rate (high -> low): "
        + ", ".join(f"{r}k: {m:.2f}" for r, m in zip(rates, med))
        + f" -> {'PASS' if p3 else 'FAIL'}")
    if (df.aq == 0).any():
        a0 = df[df.aq == 0]
        log(f"   aq=0 control: median err {a0.err.median():.2f} bpm at {int(a0.rate_k.iloc[0])}k "
            f"(aq=1 same rate: {comp[comp.rate_k == a0.rate_k.iloc[0]].err.median():.2f})")
    ok = p0 and p1 and p3
    if not ok and not smoke:
        log("\nGATE FAILURE: not fitting. "
            + ("P0: add 50k and re-check before fitting; if still failing, report "
               "'UBFC-rPPG too easy' and move to PURE. " if not p0 else "")
            + ("P1: the rPPG baseline is broken; fix extraction before reading anything. "
               if not p1 else "")
            + ("P3: damage is not monotone in rate; investigate encodes. " if not p3 else ""))
    return ok


# --------------------------------------------------------------- metrics ----

def aurc_and_risk80(err, score):
    """Retain the windows with the lowest predicted error; risk = mean true err
    of the retained set; AURC = mean risk over the 100% -> 10% coverage grid."""
    order = np.argsort(score, kind="stable")
    cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    n = len(err)
    risks = np.array([cum[max(1, int(np.ceil(c * n))) - 1] for c in COVERAGES])
    r80 = cum[max(1, int(np.ceil(0.8 * n))) - 1]
    return risks.mean(), r80, risks


def auc(y, score):
    return roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan


def metrics(err, score):
    a, r80, _ = aurc_and_risk80(err, score)
    m = (err < RELIABLE_BPM) | (err > UNRELIABLE_BPM)
    return dict(aurc=a, risk80=r80, auc5=auc(err > FAIL_BPM, score),
                auc_arev=auc(err[m] > UNRELIABLE_BPM, score[m]), n=len(err))


def oof_predict(df, cols, target, fold_of):
    y = np.log1p(df[target].to_numpy())
    folds = df.subject.map(fold_of).to_numpy()
    pred = np.full(len(df), np.nan)
    for k in sorted(set(folds)):
        te = folds == k
        m = HistGradientBoostingRegressor(**HGB)
        m.fit(df.loc[~te, cols], y[~te])
        pred[te] = m.predict(df.loc[te, cols])
    return pred


def bootstrap(df, preds, target, pairs):
    """Subject-level resampling of the held-out predictions. Returns, per
    (base, new) pair, the point estimate and 95% CI of AURC(base) - AURC(new)
    (positive = new is better), plus the relative gain."""
    rng = np.random.default_rng(BOOT_SEED)
    subj = df.subject.to_numpy()
    uniq = np.unique(subj)
    idx = {s: np.flatnonzero(subj == s) for s in uniq}
    err = df[target].to_numpy()
    diffs = {p: [] for p in pairs}
    rels = {p: [] for p in pairs}
    for _ in range(BOOT_N):
        pick = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq))])
        a = {n: aurc_and_risk80(err[pick], preds[n][pick])[0]
             for n in {x for p in pairs for x in p}}
        for b, n in pairs:
            diffs[(b, n)].append(a[b] - a[n])
            rels[(b, n)].append((a[b] - a[n]) / a[b])
    out = {}
    for p in pairs:
        point = (aurc_and_risk80(err, preds[p[0]])[0] - aurc_and_risk80(err, preds[p[1]])[0])
        d = np.array(diffs[p])
        out[p] = dict(diff=point, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                      rel=point / aurc_and_risk80(err, preds[p[0]])[0],
                      rel_lo=np.percentile(rels[p], 2.5), rel_hi=np.percentile(rels[p], 97.5))
    return out


# ------------------------------------------------------------------ main ----

def evaluate(df, target, fold_of, log, label):
    """Fit every set on df (pooled rows), return (preds, per-set metrics)."""
    preds, rows = {}, []
    for name in SETS:
        cols = columns(df, name)
        preds[name] = oof_predict(df, cols, target, fold_of)
        err = df[target].to_numpy()
        rows.append(dict(set=name, scope="pooled", **metrics(err, preds[name])))
        for r in sorted(df.rate_k.unique(), reverse=True):
            mk = (df.rate_k == r).to_numpy()
            rows.append(dict(set=name, scope=f"{int(r)}k",
                             **metrics(err[mk], preds[name][mk])))
    log(f"  fitted {len(SETS)} sets for {label}")
    return preds, pd.DataFrame(rows)


def md_table(t):
    t = t.copy()
    for c in ("aurc", "risk80", "auc5", "auc_arev"):
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    return "```\n" + t.to_string(index=False) + "\n```"


def verdicts(df, preds, target, label, log):
    pairs = [("S+M+R+P", "S+M+R+P+Q"), ("S+M+R+P+Q", "S+M+R+P+Q+Qmap")]
    bs = bootstrap(df, preds, target, pairs)
    err = df[target].to_numpy()
    a = {n: aurc_and_risk80(err, preds[n])[0] for n in SETS}
    lines = []
    c1 = bs[pairs[0]]
    c1_ok = c1["rel"] >= C1_MIN_REL_GAIN and c1["lo"] > 0
    lines.append(f"[{label}] C1 {'HOLDS' if c1_ok else 'NULL'}: adding Q to S+M+R+P changes AURC "
                 f"{a['S+M+R+P']:.3f} -> {a['S+M+R+P+Q']:.3f} ({100 * c1['rel']:+.1f}% gain, "
                 f"diff {c1['diff']:+.3f} bpm, 95% CI [{c1['lo']:+.3f}, {c1['hi']:+.3f}])")
    sp = bs[pairs[1]]
    sp_ok = sp["rel"] >= C1_MIN_REL_GAIN and sp["lo"] > 0
    lines.append(f"[{label}] SPATIAL MAP {'ADDS' if sp_ok else 'NULL'}: adding Qmap after Q changes AURC "
                 f"{a['S+M+R+P+Q']:.3f} -> {a['S+M+R+P+Q+Qmap']:.3f} ({100 * sp['rel']:+.1f}% gain, "
                 f"diff {sp['diff']:+.3f} bpm, 95% CI [{sp['lo']:+.3f}, {sp['hi']:+.3f}])")
    ratio = a["Q-only"] / a["S"]
    free = abs(ratio - 1) <= FREE_GATE_TOL or ratio < 1
    lines.append(f"[{label}] FREE-GATE claim {'STANDS' if free else 'DOES NOT STAND'}: Q-only AURC "
                 f"{a['Q-only']:.3f} vs S-only {a['S']:.3f} ({100 * (ratio - 1):+.1f}%, "
                 f"tolerance {100 * FREE_GATE_TOL:g}%)")
    for ln in lines:
        log(ln)
    return lines, bs, a


def plot(df, preds, target, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping plot")
        return False
    err = df[target].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in SETS:
        ax.plot(100 * COVERAGES, aurc_and_risk80(err, preds[name])[2], label=name)
    ax.set_xlabel("coverage (%)")
    ax.set_ylabel("risk: mean |HR error| of retained windows (bpm)")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny local runs only: folds=min(7,n_subjects), no gate abort")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    report = []

    def log(s=""):
        print(s, flush=True)
        report.append(s)

    df = load(a.inp)
    subjects = sorted(df.subject.unique())
    k = min(N_FOLDS, len(subjects)) if a.smoke else N_FOLDS
    if len(subjects) < k:
        sys.exit(f"{len(subjects)} subjects < {k} folds; use --smoke for tiny runs")
    fold_of = assign_folds(subjects, k)
    pd.DataFrame({"subject": subjects, "fold": [fold_of[s] for s in subjects]}).to_csv(
        out / "folds.csv", index=False)

    log("# C1 pilot report" + ("  (SMOKE RUN - verdicts are meaningless)" if a.smoke else ""))
    log(f"\n{len(subjects)} subjects, {len(df)} window rows, {k} folds.\n")
    for d in a.inp:
        for e in sorted(pathlib.Path(d).glob("env-*.json")):
            env = json.loads(e.read_text())
            log(f"- {e.name}: commit {env.get('git_commit', '?')[:10]}, {env.get('ffmpeg', '')}, "
                f"libavcodec {env.get('libavcodec_dev', '')}, {env.get('x264', '')[:40]}")
    log("\n## Gates\n")
    ok = gates(df, a.smoke, log)
    if not ok and not a.smoke:
        (out / "report.md").write_text("\n".join(report) + "\n")
        sys.exit(1)

    main_df = df[(df.rate_k > 0) & (df.aq == 1) & (df.gop == "long")].reset_index(drop=True)
    log("\n## Main: POS, compressed conditions pooled (aq=1)\n")
    preds, tab = evaluate(main_df, "err", fold_of, log, "POS")
    log("\n" + md_table(tab) + "\n")
    lines, bs, aur = verdicts(main_df, preds, "err", "POS", log)
    plot(main_df, preds, "err", out / "risk_coverage_pos.png", "POS, pooled compressed")

    log("\n## Robustness: CHROM target\n")
    preds_c, tab_c = evaluate(main_df, "err_chrom", fold_of, log, "CHROM")
    log("\n" + md_table(tab_c) + "\n")
    verdicts(main_df, preds_c, "err_chrom", "CHROM", log)
    plot(main_df, preds_c, "err_chrom", out / "risk_coverage_chrom.png", "CHROM, pooled compressed")

    ctl = df[df.aq == 0].reset_index(drop=True)
    if len(ctl):
        log("\n## Control: aq=0 (separate fit on these rows only)\n")
        preds_0, tab_0 = evaluate(ctl, "err", fold_of, log, "aq=0 control")
        log("\n" + md_table(tab_0[tab_0.scope == "pooled"]) + "\n")
        verdicts(ctl, preds_0, "err", "aq0-control", log)
        log("Spatial-map features should carry nothing here; a Qmap gain at aq=0 "
            "would mean the map is read as something other than AQ.")

    # Keyframe structure at one rate: descriptive only, never fitted (PLAN amendment
    # 7 October 2026). Shows the -g 60 harmonic artefact against long GOP for C0.
    ks = df[(df.rate_k > 0) & (df.aq == 1)]
    ks = ks[ks.rate_k == ks[ks.gop != "long"].rate_k.min()] if (ks.gop != "long").any() else ks.iloc[:0]
    if len(ks):
        log(f"\n## Keyframe structure at {int(ks.rate_k.iloc[0])}k (descriptive, C0)\n")
        for g, d in ks.groupby("gop"):
            log(f"- {g}: POS median err {d.err.median():.2f} bpm, MAE {d.err.mean():.2f}, "
                f"median HR_POS {d.hr_pos.median():.1f} vs ref {d.hr_ref.median():.1f}, "
                f"share err>{FAIL_BPM:g}: {100 * (d.err > FAIL_BPM).mean():.0f}% (n={len(d)})")

    log("\n## Verdict (POS, pooled, decision rule from PLAN.md section 5)\n")
    for ln in lines:
        log(ln)
    (out / "report.md").write_text("\n".join(report) + "\n")
    print(f"\nwrote {out}/report.md, folds.csv")


if __name__ == "__main__":
    main()
