# =====================================================================================================
# EPOS - REVISION EXPERIMENTS - CREDIT CARD (reviewer comments R2-3, R2-3b, R1-2, R2-1e, R1-3a, R2-4, R1-1)
#   Same nested protocol, folds, seeds, preprocessing, classifiers and threshold rule as epos_creditcard_lgbm.py / baselines_creditcard.py
#   -> every row is paired with the earlier EPOS / baseline results of this dataset.
#
#   Methods (METHODS env, comma separated):
#     EPOS-orig  : original EPOS, (mu+lambda) EA over n in [50, 300] (absolute) x sigma in [0.01, 0.20]
#     EPOS-wide  : same EA, but DATASET-SCALED range: r = n_syn / n_fraud_train in [0, R_MAX] (default 5, = the
#                  largest ratio of the tuned SMOTE-family grid). r = 0 (no synthetic rows) is inside the space.
#     RS-orig / RS-wide     : random search, same space, budget = distinct configs EPOS used in the SAME fold
#     Grid-orig / Grid-wide : grid search (G_SIGMA sigma values x n values), same budget
#     Curve      : sensitivity curve. sigma fixed (CURVE_SIGMA = 0.05, original EPOS value), r in CURVE_R
#                  (r = 0 = threshold tuning only, i.e. the n = 0 ablation). Each point: threshold from inner CV,
#                  final model on the full outer-train, one test evaluation. Run on CURVE_REPEATS repeats.
#   Runtime: wall-clock search time and final-fit time per (classifier, method, fold) + hardware.json.
#
#   Every method has its OWN inner cache -> its runtime is not reduced by configurations of another method.
#   Selection: best inner FITNESS, ties -> fewer synthetic rows. Final: full outer-train + synthetic rows (from
#   outer-train frauds only) -> real test fold at the real fraud rate, evaluated ONCE.
#
# Run on Kaggle (CPU):  !pip install -q deap
#   !python epos_revision_creditcard.py                       (default classifiers lr,lgbm,xgb)
#   !CLASSIFIERS=rf OUT_DIR=rev_cc_rf python epos_revision_creditcard.py  (RF: reduced EA budget 8/8/6, max 30
#        configs, = the original RF EPOS run; RS / Grid get the same budget; ~6-8 h per Kaggle session, resumable)
#   Env: CLASSIFIERS (lr,rf,lgbm,xgb), METHODS, R_MAX, CURVE_R, CURVE_REPEATS, TIME_BUDGET_H, OUT_DIR, CSV path.
#   The script stops cleanly before TIME_BUDGET_H and RESUMES (add the previous output folder as Kaggle input,
#   keep the same OUT_DIR name).
#   ANALYZE_ONLY=1 -> only rebuild tables / figures from the saved results.
# Outputs (OUT_DIR): revision_results.csv, generation_log.csv, hardware.json, revision_summary.csv,
#   revision_tables.tex, fig_f1_vs_ratio.png, fig_selected_n.png
# =====================================================================================================
import os, glob, time, random, json, platform, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

# ------------------------------------------------ settings ------------------------------------------------
SEED = 42
DATASET = "creditcard"
OUT_DIR = os.environ.get("OUT_DIR", f"epos_revision_{DATASET}"); os.makedirs(OUT_DIR, exist_ok=True)
CLASSIFIERS = os.environ.get("CLASSIFIERS", "lr,lgbm,xgb").split(",")
METHODS = os.environ.get("METHODS", "EPOS-orig,EPOS-wide,RS-orig,RS-wide,Grid-orig,Grid-wide,Curve").split(",")
OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 3, 2                 # same as the baselines / EPOS runs
SEARCH_MAJ_MAX = 60_000
FITNESS = os.environ.get("FITNESS", "F1")
N_RANGE = (50, 300)                                            # original EPOS space (absolute)
R_MAX = float(os.environ.get("R_MAX", 5.0))                    # wide space: r = n / n_fraud_train in [0, R_MAX]
SIGMA_RANGE = (0.01, 0.20)
MU, LAMBDA, NGEN, CXPB, MUTPB = 12, 24, 10, 0.7, 0.3
PATIENCE, MAX_EVALS = 3, 100
# per-classifier EA settings (mu, lambda, ngen, patience, max_evals). RF on Credit Card used a REDUCED budget in the
# original EPOS run (epos_creditcard_rf_fast.py: 8, 8, 6, 2, 30) -> same here, so EPOS-orig reproduces that run.
EA_CFG = {"default": (MU, LAMBDA, NGEN, PATIENCE, MAX_EVALS)}
EA_CFG["rf"] = (8, 8, 6, 2, int(os.environ.get("RF_MAX_EVALS", 30)))
G_SIGMA = 4                                                    # sigma levels of the grid search
CURVE_SIGMA = 0.05
CURVE_R = [float(v) for v in os.environ.get("CURVE_R", "0,0.05,0.1,0.25,0.5,1,2,3,5").split(",")]
CURVE_REPEATS = int(os.environ.get("CURVE_REPEATS", 1))        # 1 -> 5 outer folds for the curve
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))
N_JOBS = -1
LR_PARAMS = {"max_iter": 2000, "C": 1.0}
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"}
RF_MAX_SAMPLES = 300_000
LGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "subsample_freq": 1,
               "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 1.0}
XGB_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "logloss"}
METRICS = ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC", "PR_AUC", "Specificity", "G_mean", "Balanced_Acc", "MCC"]
RES_CSV = f"{OUT_DIR}/revision_results.csv"
NAMES = {"lr": "LR", "rf": "RF", "lgbm": "LightGBM", "xgb": "XGBoost"}

# ------------------------------------------------ analysis (tables + figures) ------------------------------------------------
def analyze(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    res = res.copy()
    main = res[res.method != "Curve"]
    g = main.groupby(["classifier", "method"])
    summ = g.agg(folds=("F1", "size"), F1_mean=("F1", "mean"), F1_std=("F1", "std"), MCC_mean=("MCC", "mean"),
                 PR_AUC_mean=("PR_AUC", "mean"), configs=("evaluated_configs", "mean"), sel_n_median=("sel_n", "median"),
                 sel_r_median=("sel_r", "median"), at_upper=("at_upper_bound", "sum"), at_lower=("at_lower_bound", "sum"),
                 search_min=("search_s", lambda s: s.mean() / 60), final_min=("final_s", lambda s: s.mean() / 60),
                 total_min=("total_s", lambda s: s.mean() / 60)).reset_index()
    # paired Wilcoxon: EPOS vs random / grid in the same space, and wide vs original EPOS
    pairs = [("EPOS-orig", "RS-orig"), ("EPOS-orig", "Grid-orig"), ("EPOS-wide", "RS-wide"),
             ("EPOS-wide", "Grid-wide"), ("EPOS-wide", "EPOS-orig")]
    wl = []
    for c in summ.classifier.unique():
        for a, b in pairs:
            A = main[(main.classifier == c) & (main.method == a)].set_index(["repeat", "fold"])["F1"]
            B = main[(main.classifier == c) & (main.method == b)].set_index(["repeat", "fold"])["F1"]
            j = A.index.intersection(B.index)
            if len(j) < 5:
                continue
            d = 100 * (A[j] - B[j])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            wl.append({"classifier": c, "A": a, "B": b, "folds": len(j), "dF1_pp": d.mean(), "wins": int((d > 0).sum()),
                       "losses": int((d < 0).sum()), "p": p})
    wil = pd.DataFrame(wl)
    summ.to_csv(f"{OUT_DIR}/revision_summary.csv", index=False); wil.to_csv(f"{OUT_DIR}/revision_wilcoxon.csv", index=False)
    print("\n" + summ.round(3).to_string(index=False)); print("\n" + wil.round(4).to_string(index=False) if len(wil) else "")
    # LaTeX
    order = ["EPOS-orig", "RS-orig", "Grid-orig", "EPOS-wide", "RS-wide", "Grid-wide"]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Search strategy and search range (" + DATASET + r", outer test folds). F1 in \%, configs = distinct "
         r"$(n,\sigma)$ configurations evaluated in the inner CV, time = mean wall-clock minutes per outer fold.}",
         r"\label{tab:search_" + DATASET + "}", r"\begin{tabular}{llcccccc}\hline",
         r"Clf. & Method & F1 & PR-AUC & Configs & median $n$ & bound hits (low/up) & time (min)\\\hline"]
    for c in summ.classifier.unique():
        for m in order:
            r = summ[(summ.classifier == c) & (summ.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            L.append(f"{NAMES.get(c, c)} & {m} & {100 * r.F1_mean:.2f}$\\pm${100 * r.F1_std:.2f} & {100 * r.PR_AUC_mean:.2f} & "
                     f"{r.configs:.0f} & {r.sel_n_median:.0f} & {int(r.at_lower)}/{int(r.at_upper)} & {r.total_min:.1f}\\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table}"]
    if len(wil):
        L += ["", r"\begin{table}[t]\centering\small", r"\caption{Paired Wilcoxon tests on F1 (" + DATASET + r").}",
              r"\label{tab:search_wil_" + DATASET + "}", r"\begin{tabular}{lllccc}\hline",
              r"Clf. & A & B & $\Delta$F1 (pp) & W/L & $p$\\\hline"]
        for _, r in wil.iterrows():
            L.append(f"{NAMES.get(r.classifier, r.classifier)} & {r.A} & {r.B} & {r.dF1_pp:+.2f} & {r.wins}/{r.losses} & "
                     + ("$<$0.001" if r.p < 0.001 else f"{r.p:.3f}") + r"\\")
        L += [r"\hline\end{tabular}", r"\end{table}"]
    open(f"{OUT_DIR}/revision_tables.tex", "w").write("\n".join(L))
    # Figure 1: F1 vs ratio (sensitivity curve)
    cv = res[res.method == "Curve"]
    if len(cv):
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in cv.classifier.unique():
            s = cv[cv.classifier == c].groupby("sel_r_target")["F1"].agg(["mean", "std"]).reset_index()
            ax.errorbar(s.sel_r_target, 100 * s["mean"], yerr=100 * s["std"], marker="o", capsize=3, label=NAMES.get(c, c))
        nmin = cv.n_fraud_train.median()
        ax.axvspan(N_RANGE[0] / nmin, N_RANGE[1] / nmin, color="grey", alpha=0.2, label="original range 50-300")
        ax.set_xscale("symlog", linthresh=0.05); ax.set_xlabel(r"$r = n_{syn} / n_{fraud,train}$  ($\sigma$ = %.2f)" % CURVE_SIGMA)
        ax.set_ylabel("test F1 (%)"); ax.set_title(f"{DATASET}: F1 vs number of synthetic frauds"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_f1_vs_ratio.png", dpi=200); plt.close(fig)
    # Figure 2: selected n (boundary hits)
    e = main[main.method.isin(["EPOS-orig", "EPOS-wide"])]
    if len(e):
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, mk in [("EPOS-orig", "o"), ("EPOS-wide", "s")]:
            s = e[e.method == m]
            ax.scatter(s.classifier.map(lambda c: list(e.classifier.unique()).index(c)) + (0.12 if m == "EPOS-wide" else -0.12),
                       s.sel_r, marker=mk, label=m, alpha=.7)
        nmin = e.n_fraud_train.median()
        for v in N_RANGE:
            ax.axhline(v / nmin, ls="--", c="grey", lw=1)
        ax.set_xticks(range(e.classifier.nunique())); ax.set_xticklabels([NAMES.get(c, c) for c in e.classifier.unique()])
        ax.set_ylabel(r"selected $r = n/n_{fraud,train}$"); ax.set_title(f"{DATASET}: selected oversampling ratio (dashed = 50 / 300)")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_selected_n.png", dpi=200); plt.close(fig)
    print(f"\ntables / figures written to {OUT_DIR}/")

if os.environ.get("ANALYZE_ONLY") == "1":
    analyze(pd.read_csv(RES_CSV)); raise SystemExit(0)

from deap import base, creator, tools, algorithms

# ------------------------------------------------ hardware / software ------------------------------------------------
def hardware():
    import sklearn, scipy
    info = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "scipy": scipy.__version__}
    try:
        info["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l][0]
        info["ram_gb"] = round(int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) / 1e6, 1)
    except Exception:
        pass
    for mod in ("lightgbm", "xgboost", "deap"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info
HW = hardware(); json.dump(HW, open(f"{OUT_DIR}/hardware.json", "w"), indent=1); print("hardware:", HW)

# ------------------------------------------------ data (identical to the EPOS / baseline scripts) ------------------------------------------------
def find_csv(env, fname, kaggle_ds):
    if os.environ.get(env):
        return os.environ[env]
    hits = sorted(glob.glob(f"/kaggle/input/**/{fname}", recursive=True)) + ([fname] if os.path.exists(fname) else [])
    if hits:
        return hits[0]
    import kagglehub
    return os.path.join(kagglehub.dataset_download(kaggle_ds), fname)

CSV = find_csv("CREDITCARD_CSV", "creditcard.csv", "mlg-ulb/creditcardfraud")
df = pd.read_csv(CSV).drop_duplicates().reset_index(drop=True)
y_all = df["Class"].astype(np.int8).values
X_df = df.drop(columns=["Class"]); X_df["Amount"] = np.log1p(X_df["Amount"].clip(lower=0))
X_all = X_df.values.astype(np.float64); N_CONT = X_all.shape[1]
OUTLIER = np.zeros(len(y_all), bool); del df, X_df
print(f"{DATASET}: n={len(y_all)}, d={X_all.shape[1]}, frauds={int(y_all.sum())} ({100 * y_all.mean():.3f}%) | "
      f"classifiers {CLASSIFIERS} | methods {METHODS} | R_MAX {R_MAX}", flush=True)

print("EA settings (mu, lambda, ngen, patience, max_evals):", {c: EA_CFG.get(c, EA_CFG["default"]) for c in CLASSIFIERS}, flush=True)

# ------------------------------------------------ model ------------------------------------------------
def make_clf(name, n_jobs=1):
    if name == "lr":
        from sklearn.linear_model import LogisticRegression; return LogisticRegression(**LR_PARAMS)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=SEED, n_jobs=n_jobs, **RF_PARAMS)
    if name == "lgbm":
        from lightgbm import LGBMClassifier; return LGBMClassifier(random_state=SEED, n_jobs=n_jobs, verbose=-1, **LGBM_PARAMS)
    if name == "xgb":
        from xgboost import XGBClassifier; return XGBClassifier(random_state=SEED, n_jobs=n_jobs, **XGB_PARAMS)
    raise ValueError(name)

def scale(tr, va, w_tr):
    sc = StandardScaler().fit(X_all[tr, :N_CONT], sample_weight=w_tr)
    tf = lambda idx: np.hstack([sc.transform(X_all[idx, :N_CONT]), X_all[idx, N_CONT:]]).astype(np.float32)
    return tf(tr), tf(va)

def generate_synthetic(Xmin, n, sigma, rng):
    S = Xmin[rng.integers(0, len(Xmin), int(n))].astype(np.float64)
    S[:, :N_CONT] += rng.normal(0.0, sigma, size=(int(n), N_CONT))
    return S.astype(np.float32)

def fit_predict(clf_name, Xtr, ytr, wtr, Xva, n, sigma, seed, n_jobs=1):
    if n > 0:
        S = generate_synthetic(Xtr[ytr == 1], n, sigma, np.random.default_rng(seed))
        Xtr = np.vstack([Xtr, S]); ytr = np.concatenate([ytr, np.ones(len(S), np.int8)])
        wtr = np.concatenate([wtr, np.ones(len(S))])
    clf = make_clf(clf_name, n_jobs)
    if clf_name == "rf":
        clf.set_params(max_samples=min(1.0, RF_MAX_SAMPLES / len(ytr)))
    return clf.fit(Xtr, ytr, sample_weight=wtr).predict_proba(Xva)[:, 1].astype(np.float64)

THR_GRID = np.unique(np.concatenate([np.linspace(0.001, 0.05, 50), np.linspace(0.05, 0.99, 95)]))
def counts(y, p, w, ts):
    o = np.argsort(p); ps, w1, w0 = p[o], (w * (y == 1))[o], (w * (y == 0))[o]
    s1 = np.concatenate([np.cumsum(w1[::-1])[::-1], [0.0]]); s0 = np.concatenate([np.cumsum(w0[::-1])[::-1], [0.0]])
    k = np.searchsorted(ps, ts, side="left"); tp, fp = s1[k], s0[k]
    return tp, fp, s1[0] - tp, s0[0] - fp
def from_counts(tp, fp, fn, tn):
    e = 1e-12; prec, rec, spec = tp / np.maximum(tp + fp, e), tp / np.maximum(tp + fn, e), tn / np.maximum(tn + fp, e)
    mcc = (tp * tn - fp * fn) / np.maximum(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), e)
    return {"Precision": prec, "Recall": rec, "F1": 2 * tp / np.maximum(2 * tp + fp + fn, e),
            "Accuracy": (tp + tn) / np.maximum(tp + fp + fn + tn, e), "Specificity": spec, "G_mean": np.sqrt(rec * spec),
            "Balanced_Acc": (rec + spec) / 2, "MCC": mcc}
def best_threshold(y, p, w):
    return float(THR_GRID[int(np.argmax(from_counts(*counts(y, p, w, THR_GRID))[FITNESS]))])
def metrics(y, p, t):
    w = np.ones(len(y)); c = counts(y, p, w, np.array([t])); m = {k: float(v[0]) for k, v in from_counts(*c).items()}
    m["ROC_AUC"] = roc_auc_score(y, p); m["PR_AUC"] = average_precision_score(y, p)
    m.update(TP=float(c[0][0]), FP=float(c[1][0]), FN=float(c[2][0]), TN=float(c[3][0])); return m

# ------------------------------------------------ inner problem (own cache per method) ------------------------------------------------
def snap_sigma(s):
    return round(round(float(np.clip(s, *SIGMA_RANGE)) / 0.005) * 0.005, 4)

class InnerProblem:
    def __init__(self, clf_name, folds, seed):
        self.clf, self.folds, self.seed, self.cache = clf_name, folds, seed, {}
    def evaluate_many(self, keys):          # key = (n_inner, sigma)
        new = [k for k in dict.fromkeys(keys) if k not in self.cache]
        out = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(fit_predict)(self.clf, f["Xtr"], f["ytr"], f["wtr"], f["Xva"], k[0], k[1], self.seed + j)
            for k in new for j, f in enumerate(self.folds))
        J = len(self.folds)
        for i, k in enumerate(new):
            ps = out[i * J:(i + 1) * J]; ys, ws = [f["yva"] for f in self.folds], [f["wva"] for f in self.folds]
            t = best_threshold(np.concatenate(ys), np.concatenate(ps), np.concatenate(ws))
            fv = [from_counts(*counts(a, b, c, np.array([t])))[FITNESS][0] for a, b, c in zip(ys, ps, ws)]
            self.cache[k] = {"fit": float(np.mean(fv)), "thr": t}
        return [self.cache[k]["fit"] for k in keys]
    def best(self):
        k = max(self.cache, key=lambda k: (round(self.cache[k]["fit"], 6), -k[0])); return k, self.cache[k]

# space: gene0 = n (orig, absolute) or r (wide, ratio); key maps the gene to the inner number of synthetic rows
class Space:
    def __init__(self, kind, n_inner_ref):
        self.kind, self.nref = kind, n_inner_ref
        self.lo, self.hi = (N_RANGE if kind == "orig" else (0.0, R_MAX))
    def key(self, g0, s):
        g0 = float(np.clip(g0, self.lo, self.hi))
        return (int(round(g0)) if self.kind == "orig" else int(round(g0 * self.nref))), snap_sigma(s)
    def final_n(self, n_inner, n_outer_ref):     # orig: same absolute n (original EPOS); wide: same ratio
        return n_inner if self.kind == "orig" else int(round(n_inner / self.nref * n_outer_ref))

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)); creator.create("Individual", list, fitness=creator.FitnessMax)

def run_ea(prob, sp, seed, log, tag):
    MU, LAMBDA, NGEN, PATIENCE, MAX_EVALS = EA_CFG.get(prob.clf, EA_CFG["default"])
    random.seed(seed); np.random.seed(seed % (2 ** 32 - 1)); rnd = random.Random(seed)
    tb = base.Toolbox()
    tb.register("individual", lambda: creator.Individual([rnd.uniform(sp.lo, sp.hi), rnd.uniform(*SIGMA_RANGE)]))
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxBlend, alpha=0.3)
    tb.register("mutate", tools.mutPolynomialBounded, low=[sp.lo, SIGMA_RANGE[0]], up=[sp.hi, SIGMA_RANGE[1]], eta=15, indpb=0.5)
    def clip(ind):
        ind[0] = float(np.clip(ind[0], sp.lo, sp.hi)); ind[1] = float(np.clip(ind[1], *SIGMA_RANGE)); return ind
    def evaluate_pop(pop):
        inv = [clip(i) for i in pop if not i.fitness.valid]
        for ind, f in zip(inv, prob.evaluate_many([sp.key(*i) for i in inv])):
            ind.fitness.values = (f,)
    pop = tb.population(n=MU); evaluate_pop(pop)
    best, stall = max(i.fitness.values[0] for i in pop), 0
    for gen in range(1, NGEN + 1):
        if len(prob.cache) >= MAX_EVALS:
            break
        off = algorithms.varOr(pop, tb, LAMBDA, CXPB, MUTPB); evaluate_pop(off)
        pop = tools.selBest(pop + off, MU); b = pop[0].fitness.values[0]
        log.append({**tag, "gen": gen, "best_inner_fit": b, "evaluated_configs": len(prob.cache)})
        stall = 0 if b > best + 1e-6 else stall + 1; best = max(best, b)
        if stall >= PATIENCE:
            break
    return prob.best()

def run_random(prob, sp, budget, seed):
    rng = np.random.default_rng(seed + 7); keys, tries = [], 0
    while len(set(keys)) < budget and tries < 50 * budget:
        keys.append(sp.key(rng.uniform(sp.lo, sp.hi), rng.uniform(*SIGMA_RANGE))); tries += 1
    prob.evaluate_many(list(dict.fromkeys(keys))[:budget]); return prob.best()

def run_grid(prob, sp, budget):
    g_n = max(2, budget // G_SIGMA)
    keys = [sp.key(a, s) for a in np.linspace(sp.lo, sp.hi, g_n) for s in np.linspace(*SIGMA_RANGE, G_SIGMA)]
    prob.evaluate_many(list(dict.fromkeys(keys))[:budget]); return prob.best()

# ------------------------------------------------ nested CV (resumable) ------------------------------------------------
_old = sorted(glob.glob(f"/kaggle/input/**/{os.path.basename(os.path.normpath(OUT_DIR))}/revision_results*.csv", recursive=True))
_old += [RES_CSV] if os.path.exists(RES_CSV) else []
rows = []
if _old:
    rows = pd.concat([pd.read_csv(p) for p in _old]).drop_duplicates(
        ["classifier", "method", "sel_r_target", "repeat", "fold"], keep="last").to_dict("records")
    print(f"RESUME: {len(rows)} rows loaded from {_old}", flush=True)
done = lambda c, m, rt, rep, o: any(r["classifier"] == c and r["method"] == m and float(r["sel_r_target"]) == rt and
                                    int(r["repeat"]) == rep and int(r["fold"]) == o for r in rows)
gen_log, t0, est_h, stopped = [], time.time(), {}, False

for clf_name in CLASSIFIERS:
    for rep in range(N_REPEATS):
        outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + rep)
        for o, (tr, te) in enumerate(outer.split(X_all, y_all)):
            if stopped:
                break
            todo = [m for m in METHODS if m != "Curve" and not done(clf_name, m, -1.0, rep, o)]
            curve_todo = [r for r in CURVE_R if "Curve" in METHODS and rep < CURVE_REPEATS and not done(clf_name, "Curve", r, rep, o)]
            if not todo and not curve_todo:
                continue
            el = (time.time() - t0) / 3600; est = 1.25 * est_h.get(clf_name, 0.3)
            if el + est > TIME_BUDGET_H:
                print(f"TIME BUDGET: {el:.2f} h used, next fold ~{est:.2f} h -> stopping; run again to resume."); stopped = True; break
            tf0 = time.time(); fseed = SEED + 1000 * rep + 10 * o
            tr = tr[~OUTLIER[tr]]
            rng = np.random.default_rng(fseed)                       # SAME search set as EPOS / baselines
            mi, ma = tr[y_all[tr] == 1], tr[y_all[tr] == 0]
            ma_s = rng.choice(ma, min(len(ma), SEARCH_MAJ_MAX), replace=False)
            s_idx = np.sort(np.concatenate([mi, ma_s])); w_s = np.where(y_all[s_idx] == 0, len(ma) / len(ma_s), 1.0)
            folds = []
            for a, b in StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=fseed).split(s_idx, y_all[s_idx]):
                Xa, Xb = scale(s_idx[a], s_idx[b], w_s[a])
                folds.append(dict(Xtr=Xa, ytr=y_all[s_idx[a]], wtr=w_s[a], Xva=Xb, yva=y_all[s_idx[b]], wva=w_s[b]))
            n_inner_ref = float(np.mean([f["ytr"].sum() for f in folds]))      # frauds in an inner-train part
            n_outer_ref = int(len(mi))                                         # frauds in the outer-train part
            Xtr, Xte = scale(tr, te, np.ones(len(tr)))
            prep_s = time.time() - tf0
            budget = {}                                    # EPOS budget of THIS fold (for RS / Grid)
            for r in rows:
                if r["classifier"] == clf_name and int(r["repeat"]) == rep and int(r["fold"]) == o and r["method"].startswith("EPOS"):
                    budget[r["method"].split("-")[1]] = int(r["evaluated_configs"])

            def finish(method, sp, kbest, info, n_cfg, search_s, r_target=-1.0):
                n_fin = sp.final_n(kbest[0], n_outer_ref) if sp is not None else kbest[0]
                ts = time.time()
                p = fit_predict(clf_name, Xtr, y_all[tr], np.ones(len(tr)), Xte, n_fin, kbest[1], fseed + 99, n_jobs=N_JOBS)
                m = metrics(y_all[te], p, info["thr"]); final_s = time.time() - ts
                lo_hit = sp is not None and kbest[0] <= (N_RANGE[0] if sp.kind == "orig" else 0) + 1e-9
                hi_hit = sp is not None and kbest[0] >= 0.97 * (N_RANGE[1] if sp.kind == "orig" else R_MAX * n_inner_ref)
                row = {"dataset": DATASET, "classifier": clf_name, "method": method, "sel_r_target": r_target, "repeat": rep,
                       "fold": o, "space": sp.kind if sp else "curve", "sel_n_inner": kbest[0], "sel_n": n_fin,
                       "sel_r": n_fin / n_outer_ref, "sel_sigma": kbest[1], "threshold": info["thr"], "inner_fit": info["fit"],
                       "evaluated_configs": n_cfg, "at_lower_bound": int(lo_hit), "at_upper_bound": int(hi_hit),
                       "n_fraud_train": n_outer_ref, "n_train_rows": len(tr), "n_test_rows": len(te),
                       "prep_s": prep_s, "search_s": search_s, "final_s": final_s, "total_s": search_s + final_s, **m}
                rows.append(row); pd.DataFrame(rows).to_csv(RES_CSV, index=False)
                print(f"{clf_name} r{rep} f{o + 1} {method:9s}{'' if r_target < 0 else f' r={r_target:<5}'} n={n_fin:<6} "
                      f"sigma={kbest[1]:.3f} cfg={n_cfg:<3} | TEST F1={m['F1']:.4f} PR-AUC={m['PR_AUC']:.4f} | "
                      f"search {search_s / 60:.1f} + final {final_s / 60:.1f} min", flush=True)

            for method in [m for m in ["EPOS-orig", "EPOS-wide", "RS-orig", "RS-wide", "Grid-orig", "Grid-wide"] if m in todo]:
                kind = method.split("-")[1]; sp = Space(kind, n_inner_ref)
                prob = InnerProblem(clf_name, folds, fseed); ts = time.time()
                if method.startswith("EPOS"):
                    kb, info = run_ea(prob, sp, fseed, gen_log, {"classifier": clf_name, "method": method, "repeat": rep, "fold": o})
                    budget[kind] = len(prob.cache)
                else:
                    B = budget.get(kind, EA_CFG.get(clf_name, EA_CFG["default"])[4])   # = EPOS budget of this fold (EA cap if EPOS not run)
                    kb, info = (run_random(prob, sp, B, fseed) if method.startswith("RS") else run_grid(prob, sp, B))
                finish(method, sp, kb, info, len(prob.cache), time.time() - ts)
            if curve_todo:
                prob = InnerProblem(clf_name, folds, fseed)
                for r_ in curve_todo:
                    ts = time.time(); k = (int(round(r_ * n_inner_ref)), CURVE_SIGMA); prob.evaluate_many([k])
                    n_fin = int(round(r_ * n_outer_ref))
                    finish("Curve", None, (n_fin, CURVE_SIGMA), prob.cache[k], 1, time.time() - ts, r_target=r_)
            del folds, Xtr, Xte
            est_h[clf_name] = (time.time() - tf0) / 3600
            print(f"--- {clf_name} repeat {rep} fold {o + 1} done in {60 * est_h[clf_name]:.1f} min "
                  f"(total {(time.time() - t0) / 60:.1f} min)", flush=True)

pd.DataFrame(gen_log).to_csv(f"{OUT_DIR}/generation_log.csv", index=False)
res = pd.DataFrame(rows)
if len(res):
    analyze(res)
print(f"\ntotal time {(time.time() - t0) / 60:.1f} min | results in {OUT_DIR}/" + (" | NOT finished -> run again" if stopped else ""))
