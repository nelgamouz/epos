# =====================================================================================================
# stats_paysim.py - statistical tests for the PaySim results ONLY (stand-alone; just run:  !python stats_paysim.py)
#   1. First merge the per-run files:  !python merge_outer_results.py paysim
#      -> merged_results/PaySim_baselines_outer_test_results.csv and merged_results/PaySim_EPOS_outer_test_results.csv
#   2. Then:  !pip install -q scipy matplotlib      then      !python stats_paysim.py
#   Results -> stats_paysim/  (friedman.csv, wilcoxon_vs_epos.csv, ranks.csv, stats_tables.tex, cd_*.png)
#   Command-line options of the general script still work, e.g.  !python stats_paysim.py --metrics F1 MCC --alpha 0.10
# =====================================================================================================
import sys
# in a notebook sys.argv survives between cells (e.g. leftovers of stats_churn) -> start clean there
if 'ipykernel' in sys.modules or 'google.colab' in sys.modules or any(a == '-f' for a in sys.argv):
    sys.argv = [sys.argv[0]]
sys.argv = [a for i, a in enumerate(sys.argv) if not (a == '-f' or (i > 0 and sys.argv[i - 1] == '-f'))]
DATASET = "PaySim"
OUT_DIR = "stats_paysim"
K_FOLDS = 5                     # outer folds per repeat (Nadeau-Bengio correction)
# the two merged files made by  !python merge_outer_results.py paysim
# (searched in merged_results/ of the working dir first, then anywhere in /kaggle/input)
import glob as _g
def _find(name):
    hits = _g.glob(f"merged_results/{name}") + _g.glob(f"/kaggle/input/**/{name}", recursive=True) + _g.glob(name)
    return hits[0] if hits else name
FILES = [_find("PaySim_baselines_outer_test_results.csv"), _find("PaySim_EPOS_outer_test_results.csv")]
if "--dataset" not in sys.argv:
    sys.argv += ["--dataset", DATASET]
if "--files" not in sys.argv:
    sys.argv += ["--files", *FILES]
if "--out" not in sys.argv:
    sys.argv += ["--out", OUT_DIR]
if "--k" not in sys.argv:
    sys.argv += ["--k", str(K_FOLDS)]

# =====================================================================================================
# stats_tests.py - Friedman / Nemenyi / Wilcoxon-Holm (+ Nadeau-Bengio corrected t-test) for the EPOS study
#
# Input : the PER-FOLD result files (outer_test_results.csv) of the baseline and EPOS runs of ONE dataset.
#         Every file needs the columns classifier, method, repeat, fold and the metric columns (F1, MCC, ...).
#         Files are read in the given order; if the same (classifier, method, repeat, fold) appears twice the LATER
#         file wins -> list the old baseline files first and the re-runs (e.g. *lgbm_stable*) after them.
#         The EPOS method (method == "EPOS") is renamed to "EPOS (ours)".
# Tests (per classifier and metric, blocks = the paired outer test folds):
#   (1) Friedman test over all methods (+ Iman-Davenport F), average ranks (1 = best)
#   (2) Nemenyi critical difference (CD) + CD diagram (Demsar 2006)
#   (3) EPOS vs every baseline: two-sided Wilcoxon signed-rank test, Holm-corrected over the baselines,
#       wins/ties/losses over the folds, and the Nadeau-Bengio corrected resampled t-test (repeated k-fold CV
#       folds share training data, so the plain tests are optimistic - the corrected test is the conservative check)
#   (4) POOLED over classifiers: blocks = (classifier, fold) -> one Friedman / CD diagram per dataset
# Usage (Kaggle: add the result folders as input):
#   pip install -q scipy matplotlib
#   python stats_tests.py --dataset "Credit Card" --k 5 \
#       --files "/kaggle/input/**/baselines_cc_results/outer_test_results.csv" \
#               "/kaggle/input/**/baseline_cc_*_results/outer_test_results.csv" \
#               "/kaggle/input/**/baselines_cc_lgbm_stable_results/outer_test_results.csv" \
#               "/kaggle/input/**/epos_creditcard*_results/outer_test_results.csv"
#   options: --metrics F1 MCC PR_AUC   --main F1   --alpha 0.05   --out stats_creditcard
# Outputs (--out): friedman.csv, wilcoxon_vs_epos.csv, ranks.csv, stats_tables.tex, cd_<classifier>_<metric>.png
# =====================================================================================================
import argparse, glob, os, re, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--files", nargs="+", required=True, help="per-fold CSVs or glob patterns (later files win)")
ap.add_argument("--metrics", nargs="+", default=["F1", "MCC", "PR_AUC"])
ap.add_argument("--main", default="F1", help="metric of the LaTeX Wilcoxon table")
ap.add_argument("--methods", nargs="*", default=None, help="keep only these methods (default: all found)")
ap.add_argument("--ref", default="EPOS (ours)")
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--k", type=int, default=5, help="outer folds per repeat (for the Nadeau-Bengio correction)")
ap.add_argument("--out", default=None)
ap.add_argument("--rope", type=float, default=0.01, help="region of practical equivalence for the Bayesian test, in metric units (0.01 = 1 percentage point)")
A, _unknown = ap.parse_known_args()            # ignores Jupyter/Colab's own '-f kernel.json' argument
OUT = A.out or "stats_" + re.sub(r"\W+", "_", A.dataset.lower()).strip("_")
os.makedirs(OUT, exist_ok=True)

NAMES = {"lr": "LR", "rf": "RF", "lgbm": "LightGBM", "xgb": "XGBoost"}
ORDER = ["No oversampling", "Class weights", "SMOTE", "ADASYN", "Borderline-SMOTE", "SMOTE-ENN", A.ref]

# ------------------------------------------------ load ------------------------------------------------
parts = []
for pat in A.files:
    hits = sorted(glob.glob(pat, recursive=True)) or ([pat] if os.path.exists(pat) else [])
    if not hits:
        print(f"WARNING: no file matches {pat}")
    for p in hits:
        d = pd.read_csv(p)
        need = {"classifier", "method", "repeat", "fold"}
        if not need <= set(d.columns):
            print(f"WARNING: {p} skipped (missing {need - set(d.columns)})"); continue
        d["_src"] = p; parts.append(d); print(f"read {len(d):5d} rows  {p}")
if not parts:
    sys.exit("no input files")
df = pd.concat(parts, ignore_index=True)
df["method"] = df["method"].replace({"EPOS": A.ref})
# sanity check: F1 / MCC / PR_AUC must lie in [-1, 1]; anything else means a broken input file
_bad = pd.Series(False, index=df.index)
for _m in [m for m in A.metrics if m in df.columns]:
    _v = pd.to_numeric(df[_m], errors="coerce")
    _bad |= _v.isna() | (_v.abs() > 1.0 + 1e-9)
if _bad.any():
    _src = "source_file" if "source_file" in df.columns else "_src"
    print("\nERROR: impossible metric values (outside [-1, 1]) in these rows:")
    print(df[_bad].groupby(["classifier", "method", _src]).size().rename("rows").to_string())
    print("\n-> those rows are excluded; fix the listed result files and re-run merge_outer_results.py")
    df = df[~_bad].copy()
df = df.drop_duplicates(["classifier", "method", "repeat", "fold"], keep="last")
if A.methods:
    df = df[df.method.isin(A.methods + [A.ref])]
missing = [m for m in A.metrics if m not in df.columns]
if missing:
    sys.exit(f"metric columns not found: {missing}")
CLFS = [c for c in ["lr", "rf", "lgbm", "xgb"] if c in set(df.classifier)] + \
       sorted(set(df.classifier) - {"lr", "rf", "lgbm", "xgb"})

# Nemenyi critical values q_alpha (Demsar 2006, Table 5), index = number of methods k
Q = {0.05: [0, 0, 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
     0.10: [0, 0, 1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920]}

def block_matrix(d, metric, block_cols):
    M = d.pivot_table(index=block_cols, columns="method", values=metric, aggfunc="first")
    meths = [m for m in ORDER if m in M.columns] + sorted(set(M.columns) - set(ORDER))
    M = M[meths]
    full = M.dropna()
    return full, len(M) - len(full)

def friedman(M):
    X = M.values; n, k = X.shape
    R = stats.rankdata(-X, axis=1)                                   # 1 = best, ties -> average rank
    avg = R.mean(0)
    if k < 3 or n < 2:
        return avg, np.nan, np.nan, np.nan, np.nan
    chi2, p = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    ff = (n - 1) * chi2 / max(n * (k - 1) - chi2, 1e-12)              # Iman-Davenport
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (n - 1))
    return avg, chi2, p, ff, p_ff

def nemenyi_cd(k, n, alpha):
    q = Q.get(alpha, Q[0.05])
    return q[k] * np.sqrt(k * (k + 1) / (6.0 * n)) if 2 <= k < len(q) else np.nan

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj

def wilcoxon_p(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="zsplit", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0

def nb_corrected_t(d, k):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV (n_test/n_train = 1/(k-1))"""
    d = np.asarray(d, float); J = len(d); v = d.var(ddof=1)
    if J < 2 or v == 0:
        return 0.0 if np.allclose(d, 0) else np.inf, 1.0 if np.allclose(d, 0) else 0.0
    t = d.mean() / np.sqrt((1.0 / J + 1.0 / (k - 1)) * v)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))

def corrected_scale(d, k):
    """std of the mean difference with the Nadeau-Bengio / Corani-Benavoli correlation correction (rho = 1/k)"""
    d = np.asarray(d, float); J = len(d)
    return np.sqrt((1.0 / J + 1.0 / (k - 1)) * d.var(ddof=1)) if J > 1 else 0.0

def corrected_ci(d, k, level=0.95):
    """confidence interval of the mean difference, corrected for overlapping training sets"""
    d = np.asarray(d, float); sc = corrected_scale(d, k)
    if sc == 0:
        return float(d.mean()), float(d.mean())
    h = stats.t.ppf(0.5 + level / 2, len(d) - 1) * sc
    return float(d.mean() - h), float(d.mean() + h)

def bayes_corr_t(d, k, rope):
    """Bayesian correlated t-test (Corani & Benavoli 2015; Benavoli et al. 2017, JMLR 18(77)).
    Posterior of the mean difference = Student t(df = J-1, loc = mean, scale = corrected std).
    Returns P(EPOS worse), P(practically equivalent: |diff| < rope), P(EPOS better)."""
    d = np.asarray(d, float); mu, sc = d.mean(), corrected_scale(d, k)
    if sc == 0:
        return (float(mu < -rope), float(abs(mu) <= rope), float(mu > rope))
    T = stats.t(len(d) - 1, loc=mu, scale=sc)
    p_l, p_r = float(T.cdf(-rope)), float(T.sf(rope))
    return p_l, max(0.0, 1.0 - p_l - p_r), p_r

def bayes_verdict(pl, pe, pr, thr=0.95):
    if pr > thr: return "EPOS better"
    if pl > thr: return "EPOS worse"
    if pe > thr: return "practically equivalent"
    if pr + pe > thr: return "EPOS not worse"
    if pl + pe > thr: return "EPOS not better"
    return "inconclusive"

def rank_biserial(d):
    """matched-pairs rank-biserial correlation (effect size of the Wilcoxon signed-rank test), in [-1, 1]"""
    d = np.asarray(d, float); d = d[np.abs(d) > 1e-12]
    if not len(d): return 0.0
    r = stats.rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())

def cliffs_delta(a, b):
    """Cliff's delta between the two score distributions, in [-1, 1] (> 0: EPOS scores tend to be higher)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((np.sign(a[:, None] - b[None, :])).mean())

def cliff_label(x):
    x = abs(x)            # Romano et al. (2006) thresholds
    return "negligible" if x < 0.147 else "small" if x < 0.33 else "medium" if x < 0.474 else "large"

def cd_diagram(names, avg, cd, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = len(names); o = np.argsort(avg); names = [names[i] for i in o]; avg = np.asarray(avg)[o]
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.32 * k)); ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0, 1)
    ax.axis("off"); y0 = 0.82
    ax.hlines(y0, lo, hi, color="k", lw=1)
    for r in range(lo, hi + 1):
        ax.vlines(r, y0, y0 + 0.03, color="k", lw=1); ax.text(r, y0 + 0.05, str(r), ha="center", fontsize=8)
    step = 0.62 / max(1, (k + 1) // 2)
    for i, (nm, a) in enumerate(zip(names, avg)):
        left = i < (k + 1) // 2
        yy = y0 - 0.08 - step * (i if left else k - 1 - i)
        xe = lo - 0.25 if left else hi + 0.25
        ax.plot([a, a, xe], [y0, yy, yy], color="k", lw=0.8)
        bold = "bold" if nm == A.ref else "normal"
        ax.text(xe + (-0.03 if left else 0.03), yy, f"{nm} ({a:.2f})", ha="right" if left else "left",
                va="center", fontsize=8, fontweight=bold)
    # cliques: maximal groups whose rank range < CD (not significantly different)
    if np.isfinite(cd):
        ax.plot([lo, lo + cd], [y0 + 0.13, y0 + 0.13], color="k", lw=2)
        ax.text(lo + cd / 2, y0 + 0.155, f"CD = {cd:.2f}", ha="center", fontsize=8)
        cl = []
        for i in range(k):
            j = max(jj for jj in range(i, k) if avg[jj] - avg[i] < cd)
            if j > i and not any(a_ <= i and j <= b_ for a_, b_ in cl):
                cl.append((i, j))
        for c_, (i, j) in enumerate(cl):
            yy = y0 - 0.03 - 0.025 * c_
            ax.plot([avg[i] - 0.03, avg[j] + 0.03], [yy, yy], color="tab:red", lw=2.5)
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------ tests ------------------------------------------------
fr_rows, wx_rows, rk_rows = [], [], []
groups = [(c, df[df.classifier == c], ["repeat", "fold"]) for c in CLFS]
groups.append(("ALL (pooled)", df, ["classifier", "repeat", "fold"]))
for clf, d, bcols in groups:
    for metric in A.metrics:
        M, dropped = block_matrix(d, metric, bcols)
        if M.shape[1] < 2 or len(M) < 2:
            continue
        meths = list(M.columns); n, k = M.shape
        avg, chi2, p, ff, p_ff = friedman(M)
        cd = nemenyi_cd(k, n, A.alpha)
        if dropped:
            print(f"   note: {clf} {metric}: {dropped} fold(s) without all methods were dropped")
        best = meths[int(np.argmin(avg))]
        ref_rank = float(avg[meths.index(A.ref)]) if A.ref in meths else np.nan
        fr_rows.append({"classifier": clf, "metric": metric, "folds": n, "methods": k, "chi2": chi2, "p_friedman": p,
                        "F_iman_davenport": ff, "p_iman_davenport": p_ff, "CD_nemenyi": cd, "best_by_rank": best,
                        "ref_avg_rank": ref_rank,
                        "ref_position": (int(stats.rankdata(avg)[meths.index(A.ref)]) if A.ref in meths else None)})
        for m_, a_ in zip(meths, avg):
            rk_rows.append({"classifier": clf, "metric": metric, "method": m_, "avg_rank": a_,
                            "mean": M[m_].mean(), "std": M[m_].std()})
        if clf.startswith("ALL") or metric in A.metrics:
            nm = clf.split()[0].lower()
            cd_diagram(meths, avg, cd, f"{A.dataset} - {NAMES.get(clf, clf)} - {metric}: Friedman p = {p:.3g} "
                       f"(N = {n} folds, k = {k})", f"{OUT}/cd_{nm}_{metric}.png")
        if A.ref in meths and not clf.startswith("ALL"):
            base = [m_ for m_ in meths if m_ != A.ref]
            raw, recs = [], []
            for b in base:
                dd = (M[A.ref] - M[b]).values
                t, p_nb = nb_corrected_t(dd, A.k)
                raw.append(wilcoxon_p(dd))
                recs.append({"classifier": clf, "metric": metric, "baseline": b, "folds": n,
                             "mean_diff": dd.mean(), "median_diff": float(np.median(dd)),
                             "wins": int((dd > 1e-12).sum()), "ties": int((np.abs(dd) <= 1e-12).sum()),
                             "losses": int((dd < -1e-12).sum()), "nb_t": t, "p_nb_corrected": p_nb})
                lo, hi = corrected_ci(dd, A.k); pl, pe, pr = bayes_corr_t(dd, A.k, A.rope)
                cdl = cliffs_delta(M[A.ref].values, M[b].values)
                recs[-1].update(ci95_low=lo, ci95_high=hi, rank_biserial=rank_biserial(dd), cliffs_delta=cdl,
                                cliffs_magnitude=cliff_label(cdl), rope=A.rope, P_epos_worse=pl,
                                P_equivalent=pe, P_epos_better=pr, bayes_verdict=bayes_verdict(pl, pe, pr))
            for r_, p_raw, p_h in zip(recs, raw, holm(raw)):
                r_.update(p_wilcoxon=p_raw, p_wilcoxon_holm=p_h,
                          verdict=("EPOS better" if p_h < A.alpha and r_["mean_diff"] > 0 else
                                   "EPOS worse" if p_h < A.alpha else "no sig. difference"))
                wx_rows.append(r_)

fr, wx, rk = pd.DataFrame(fr_rows), pd.DataFrame(wx_rows), pd.DataFrame(rk_rows)
fr.to_csv(f"{OUT}/friedman.csv", index=False); wx.to_csv(f"{OUT}/wilcoxon_vs_epos.csv", index=False)
rk.to_csv(f"{OUT}/ranks.csv", index=False)

# ------------------------------------------------ console ------------------------------------------------
pd.set_option("display.width", 200)
print(f"\n===== {A.dataset}: Friedman test (alpha = {A.alpha}) =====")
if len(fr):
    show = fr.copy(); show["classifier"] = show.classifier.map(lambda c: NAMES.get(c, c))
    print(show[["classifier", "metric", "folds", "methods", "chi2", "p_friedman", "CD_nemenyi", "best_by_rank",
                "ref_avg_rank", "ref_position"]].round(4).to_string(index=False))
if len(wx):
    print(f"\n===== {A.dataset}: {A.ref} vs each baseline ({A.main}) - Wilcoxon signed-rank, Holm-corrected =====")
    w = wx[wx.metric == A.main].copy(); w["classifier"] = w.classifier.map(lambda c: NAMES.get(c, c))
    w["mean_diff_pp"] = 100 * w.mean_diff
    print(w[["classifier", "baseline", "mean_diff_pp", "wins", "ties", "losses", "p_wilcoxon", "p_wilcoxon_holm",
             "p_nb_corrected", "verdict"]].round(4).to_string(index=False))
    print(f"\n===== {A.dataset}: effect sizes + Bayesian correlated t-test ({A.main}, ROPE = +-{100 * A.rope:g} pp) =====")
    w["CI95_pp"] = [f"[{100 * a:+.2f}, {100 * b:+.2f}]" for a, b in zip(w.ci95_low, w.ci95_high)]
    print(w[["classifier", "baseline", "mean_diff_pp", "CI95_pp", "rank_biserial", "cliffs_delta", "cliffs_magnitude",
             "P_epos_worse", "P_equivalent", "P_epos_better", "bayes_verdict"]].round(3).to_string(index=False))

# ------------------------------------------------ LaTeX ------------------------------------------------
def fp(p):
    return "--" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")

lab = re.sub(r"\W+", "", A.dataset.lower())
L = [r"% requires \usepackage{booktabs}", r"\begin{table}[!t]", r"\centering", r"\scriptsize",
     rf"\caption{{Friedman test over the paired outer test folds on the {A.dataset} dataset ($\alpha$ = {A.alpha}). "
     rf"Rank: average rank of EPOS (1 = best) and its position among the $k$ methods; CD: Nemenyi critical difference.}}",
     rf"\label{{tab:friedman_{lab}}}", r"\begin{tabular}{llrrrll}", r"\toprule",
     r"Classifier & Metric & $\chi^2_F$ & $p$ & CD & EPOS rank (pos.) & Best by rank \\", r"\midrule"]
for _, r in fr.iterrows():
    pos = f"{r.ref_avg_rank:.2f} ({int(r.ref_position)}/{int(r.methods)})" if np.isfinite(r.ref_avg_rank) else "--"
    L.append(f"{NAMES.get(r.classifier, r.classifier)} & {r.metric.replace('_', '-')} & {r.chi2:.2f} & "
             f"{fp(r.p_friedman)} & {r.CD_nemenyi:.2f} & {pos} & {r.best_by_rank} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
if len(wx):
    w = wx[wx.metric == A.main]
    bases = [b for b in ORDER if b in set(w.baseline)] + sorted(set(w.baseline) - set(ORDER))
    clfs = [c for c in CLFS if c in set(w.classifier)]
    L += [r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
          rf"\caption{{EPOS vs.\ each baseline on the {A.dataset} dataset ({A.main.replace('_', '-')}): mean paired "
          rf"difference in percentage points, Holm-corrected two-sided Wilcoxon signed-rank $p$-value and wins/ties/"
          rf"losses over the outer folds. $^{{+}}$/$^{{-}}$: EPOS significantly better/worse ($\alpha$ = {A.alpha}); "
          rf"$^{{\dagger}}$: also significant with the Nadeau--Bengio corrected resampled $t$-test.}}",
          rf"\label{{tab:wilcoxon_{lab}}}", r"\begin{tabular}{l" + "c" * len(bases) + "}", r"\toprule",
          "Classifier & " + " & ".join(bases) + r" \\", r"\midrule"]
    for c in clfs:
        cells = []
        for b in bases:
            r = w[(w.classifier == c) & (w.baseline == b)]
            if not len(r):
                cells.append("--"); continue
            r = r.iloc[0]; sig = r.p_wilcoxon_holm < A.alpha
            mark = ("$^{+}$" if r.mean_diff > 0 else "$^{-}$") if sig else ""
            mark += r"$^{\dagger}$" if (sig and r.p_nb_corrected < A.alpha) else ""
            cells.append(f"{100 * r.mean_diff:+.2f}{mark} ({fp(r.p_wilcoxon_holm)}; {r.wins}/{r.ties}/{r.losses})")
        L.append(f"{NAMES.get(c, c)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
if len(wx):
    w = wx[wx.metric == A.main]
    L += ["", r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
          rf"\caption{{Effect sizes and Bayesian analysis of EPOS vs.\ each baseline on the {A.dataset} dataset "
          rf"({A.main.replace('_', '-')}). $\Delta$: mean paired difference (pp) with 95\% confidence interval corrected "
          rf"for overlapping training sets; $\delta$: Cliff's delta (N/S/M/L = negligible/small/medium/large); "
          rf"$P(<)$, $P(=)$, $P(>)$: posterior probabilities of the Bayesian correlated $t$-test that EPOS is worse, "
          rf"practically equivalent (ROPE $\pm${100 * A.rope:g} pp) or better.}}",
          rf"\label{{tab:bayes_{lab}}}", r"\begin{tabular}{llrcrccc}", r"\toprule",
          r"Classifier & Baseline & $\Delta$ (pp) & 95\% CI & $\delta$ & $P(<)$ & $P(=)$ & $P(>)$ \\", r"\midrule"]
    for c in [c for c in CLFS if c in set(w.classifier)]:
        for _, r in w[w.classifier == c].iterrows():
            L.append(f"{NAMES.get(c, c)} & {r.baseline} & {100 * r.mean_diff:+.2f} & "
                     f"[{100 * r.ci95_low:+.2f}, {100 * r.ci95_high:+.2f}] & {r.cliffs_delta:+.2f} "
                     f"({r.cliffs_magnitude[0].upper()}) & {r.P_epos_worse:.2f} & {r.P_equivalent:.2f} & "
                     f"{r.P_epos_better:.2f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
open(f"{OUT}/stats_tables.tex", "w").write("\n".join(L) + "\n")
print(f"\nsaved: {OUT}/friedman.csv, wilcoxon_vs_epos.csv, ranks.csv, stats_tables.tex, cd_*.png")
