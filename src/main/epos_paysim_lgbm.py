# =====================================================================================================
# EPOS (original noise-augmented oversampling + evolutionary search) on PaySim (Financial_Fraud_Detection)
# LEAKAGE-FREE version with nested cross-validation and mean +- std of all metrics.
# LIGHTGBM VERSION (corrected): identical protocol, folds, seeds, search set, EA budget, features and metrics as
# epos_paysim_nested.py. Fix of the LightGBM instability (F1 64.6 +- 12.0 in the first run):
#   With ~0.13% frauds the hessians p(1-p) of the legitimate rows are tiny, so LightGBM's defaults
#   (min_child_weight=1e-3, reg_lambda=0) allow leaves with a few frauds and huge leaf values -G/H
#   -> raw scores explode on some folds and the inner-CV threshold no longer transfers to the test fold.
#   Fix: min_child_weight=1.0 and reg_lambda=1.0 (= the XGBoost defaults, which is why XGBoost was stable).
#   Use the SAME LGBM_PARAMS for the LightGBM baselines, otherwise the comparison is not fair.
#
#   Outer CV  : 5-fold StratifiedKFold x 2 repeats = 10 untouched test folds. SAME folds, seeds, search set and
#               preprocessing as the baseline scripts (baseline_*.py) -> every outer fold is paired with the baselines.
#   Inner CV  : 3-fold StratifiedKFold on a search set taken from the OUTER-TRAIN part only (all training frauds +
#               SEARCH_MAJ_MAX legitimate rows, re-weighted so the class balance equals the full outer-train)
#               -> fitness of the evolutionary search.
#   Synthetic : rows are generated ONLY from the frauds of the training part of a split and added ONLY to that
#               training part. Validation / test folds contain REAL rows only, at the real fraud rate.
#   Scaling   : StandardScaler fitted on the training part only (noise is in standardised units).
#   Threshold : chosen on the inner out-of-fold predictions (never on the test fold).
#   Final     : best (n, sigma) -> model on the FULL outer-train + n synthetic rows -> ONE evaluation on the test fold.
#
# Leakage / errors of the old code that are removed here:
#   (1) synthetic rows were added BEFORE the CV split -> copies of the same frauds in train AND test
#   (2) evaluation on the augmented (resampled) data instead of real data at the real fraud rate
#   (3) "best metric values observed during optimization" = the maximum of each metric over ALL individuals
#       (different models per metric, selected on the same folds they were reported on) -> removed
#   (4) one single noise vector for all synthetic rows -> now independent noise per row
#   (5) latent_dim had no effect -> replaced by sigma, the noise std (previously fixed 0.05)
#
# Run on Kaggle:   !pip install -q deap      then      !python epos_paysim_lgbm.py
#   CLASSIFIERS=lr,rf,lgbm,xgb (default) | FITNESS=F1 (default, as original EPOS) or MCC (as the baselines)
#   PAYSIM_CSV=/path/to/Financial_Fraud_Detection.csv   (otherwise searched in /kaggle/input or downloaded with kagglehub)
#   Long runs: the script stops cleanly before TIME_BUDGET_H hours (default 11) and saves everything; run it again
#   with the same OUT_DIR (in Kaggle: add the previous output as input) -> finished folds are skipped.
# Outputs (OUT_DIR): outer_test_results.csv, EPOS_paysim_summary_mean_std.txt (same format as the baseline summaries ->
#   summary_table.py), generation_log.csv
# =====================================================================================================
import os, glob, time, random, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from deap import base, creator, tools, algorithms
warnings.filterwarnings("ignore")

# ------------------------------------------------ settings ------------------------------------------------
SEED = 42
# NEW output folder: the old (unstable) run lives in epos_paysim_lgbm_results -> never resume from it
OUT_DIR = os.environ.get("OUT_DIR", "epos_paysim_lgbm_stable_results"); os.makedirs(OUT_DIR, exist_ok=True)
LGBM_CFG = "stable_mcw1.0_lambda1.0"               # written into every row; resume only accepts rows with this tag
CLASSIFIERS = os.environ.get("CLASSIFIERS", "lgbm").split(",")
OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 3, 2      # same as the baselines
SEARCH_MAJ_MAX = 100_000                     # legitimate rows in the search set (same as the baselines)
FITNESS = os.environ.get("FITNESS", "F1")         # "F1" (original EPOS) or "MCC" (fitness used by the baselines)
N_RANGE = (50, 300)                                # number of synthetic frauds (original EPOS)
SIGMA_RANGE = (0.01, 0.20)                         # noise std in standardised units (original: fixed 0.05)
MU, LAMBDA, NGEN, CXPB, MUTPB = 12, 24, 10, 0.7, 0.3
PATIENCE = 3                                       # stop if the best inner fitness does not improve
MAX_EVALS = 100                                    # max. distinct (n, sigma) configurations per outer fold
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))
N_JOBS = -1
# classifier settings = the baseline scripts (keep them identical, otherwise the comparison is not fair)
LR_PARAMS = {"max_iter": 2000, "C": 1.0}
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"}
RF_MAX_SAMPLES = 300_000                           # bootstrap rows per tree (as in the baselines)
LGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
               "subsample_freq": 1, "colsample_bytree": 0.8,
               "min_child_weight": 1.0, "reg_lambda": 1.0}   # stabilised (see header)
XGB_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "logloss"}

# ------------------------------------------------ data ------------------------------------------------
def find_csv():
    if os.environ.get("PAYSIM_CSV"):
        return os.environ["PAYSIM_CSV"]
    hits = sorted(glob.glob("/kaggle/input/**/Financial_Fraud_Detection.csv", recursive=True)) + \
        (["Financial_Fraud_Detection.csv"] if os.path.exists("Financial_Fraud_Detection.csv") else [])
    if hits:
        return hits[0]
    import kagglehub                                # fallback: download from Kaggle
    return os.path.join(kagglehub.dataset_download("nourddineelgamouz/financial-fraud-detection"), "Financial_Fraud_Detection.csv")

# features as in the baselines (FEATURE_SET = "paper"): amount, balanceDiffOrig, balanceDiffDest (signed log1p,
# row-wise) + one-hot transaction type; nameOrig / nameDest / isFlaggedFraud are not used
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
CSV = find_csv()
cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
df = pd.read_csv(CSV, usecols=lambda c: c in cols).reset_index(drop=True)
slog = lambda x: np.sign(x) * np.log1p(np.abs(x))
y_all = df["isFraud"].astype(np.int8).values
cont = np.column_stack([slog(df["amount"].values), slog((df["oldbalanceOrg"] - df["newbalanceOrig"]).values),
                        slog((df["oldbalanceDest"] - df["newbalanceDest"]).values)])
onehot = np.stack([(df["type"].values == t_) for t_ in TYPES], 1)
X_all = np.hstack([cont, onehot]).astype(np.float32)
N_CONT = 3                                                # noise only on the 3 continuous features
# original EPOS outlier rules - applied to TRAINING rows only (test folds stay complete), as in the baselines
OUTLIER = (((df["oldbalanceOrg"] == 0) & (df["amount"] > 0)) | (df["newbalanceOrig"] == df["oldbalanceOrg"]) |
           ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] > 1e7))).values
del df, cont, onehot
print(f"data: {CSV} | outlier rules flag {int(OUTLIER.sum())} rows (removed from training parts only)")

n_pos = int(y_all.sum())
print(f"PaySim (Financial_Fraud_Detection): n={len(y_all)}, d={X_all.shape[1]} ({N_CONT} continuous), frauds={n_pos} "
      f"({100 * n_pos / len(y_all):.3f}%) | classifiers {CLASSIFIERS} | fitness {FITNESS}")

# ------------------------------------------------ generator + classifier ------------------------------------------------
def make_clf(name, n_jobs=1):
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**LR_PARAMS)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=SEED, n_jobs=n_jobs, **RF_PARAMS)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, n_jobs=n_jobs, verbose=-1, **LGBM_PARAMS)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=SEED, n_jobs=n_jobs, **XGB_PARAMS)
    raise ValueError(f"unknown classifier {name}")

def scale(tr, va, w_tr):
    """StandardScaler fitted on the TRAINING rows only (continuous columns; one-hot columns unchanged)"""
    sc = StandardScaler().fit(X_all[tr, :N_CONT], sample_weight=w_tr)
    tf = lambda idx: np.hstack([sc.transform(X_all[idx, :N_CONT]), X_all[idx, N_CONT:]]).astype(np.float32)
    return tf(tr), tf(va)

def generate_synthetic(Xmin, n, sigma, rng):
    """original EPOS: random training frauds + independent Gaussian noise on the continuous (standardised) features;
    one-hot columns (if any) of the copied fraud are kept, so every synthetic row is a valid record"""
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

# ------------------------------------------------ metrics (sample-weighted) ------------------------------------------------
METRICS = ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC", "PR_AUC", "Specificity", "G_mean", "Balanced_Acc", "MCC"]
THR_GRID = np.unique(np.concatenate([np.linspace(0.001, 0.05, 50), np.linspace(0.05, 0.99, 95)]))

def counts(y, p, w, ts):
    """weighted TP, FP, FN, TN for every threshold in ts (prediction = p >= t)"""
    o = np.argsort(p); ps, w1, w0 = p[o], (w * (y == 1))[o], (w * (y == 0))[o]
    s1 = np.concatenate([np.cumsum(w1[::-1])[::-1], [0.0]]); s0 = np.concatenate([np.cumsum(w0[::-1])[::-1], [0.0]])
    k = np.searchsorted(ps, ts, side="left")
    tp, fp = s1[k], s0[k]; P, N = s1[0], s0[0]
    return tp, fp, P - tp, N - fp

def from_counts(tp, fp, fn, tn):
    e = 1e-12
    prec, rec, spec = tp / np.maximum(tp + fp, e), tp / np.maximum(tp + fn, e), tn / np.maximum(tn + fp, e)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, e)
    mcc = (tp * tn - fp * fn) / np.maximum(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), e)
    return {"Precision": prec, "Recall": rec, "F1": f1, "Accuracy": (tp + tn) / np.maximum(tp + fp + fn + tn, e),
            "Specificity": spec, "G_mean": np.sqrt(rec * spec), "Balanced_Acc": (rec + spec) / 2, "MCC": mcc}

def best_threshold(y, p, w):
    return float(THR_GRID[int(np.argmax(from_counts(*counts(y, p, w, THR_GRID))[FITNESS]))])

def metrics(y, p, t, w=None):
    w = np.ones(len(y)) if w is None else w
    c = counts(y, p, w, np.array([t])); m = {k: float(v[0]) for k, v in from_counts(*c).items()}
    m["ROC_AUC"] = roc_auc_score(y, p, sample_weight=w); m["PR_AUC"] = average_precision_score(y, p, sample_weight=w)
    m.update(TP=float(c[0][0]), FP=float(c[1][0]), FN=float(c[2][0]), TN=float(c[3][0]))
    return m

# ------------------------------------------------ EA (DEAP) ------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def make_toolbox(seed):
    rnd = random.Random(seed)
    tb = base.Toolbox()
    tb.register("individual", lambda: creator.Individual([rnd.uniform(*N_RANGE), rnd.uniform(*SIGMA_RANGE)]))
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxBlend, alpha=0.3)
    tb.register("mutate", tools.mutPolynomialBounded, low=[N_RANGE[0], SIGMA_RANGE[0]],
                up=[N_RANGE[1], SIGMA_RANGE[1]], eta=15, indpb=0.5)
    return tb

def clip(ind):
    ind[0] = float(np.clip(ind[0], *N_RANGE)); ind[1] = float(np.clip(ind[1], *SIGMA_RANGE)); return ind

def key(ind):                                       # integer n, sigma on a 0.005 grid -> cache hits
    return int(round(ind[0])), round(round(ind[1] / 0.005) * 0.005, 4)

class InnerProblem:
    """fitness = mean inner-fold FITNESS at ONE threshold chosen on the pooled inner out-of-fold predictions"""
    def __init__(self, clf_name, folds, seed):
        self.clf, self.folds, self.seed, self.cache = clf_name, folds, seed, {}

    def evaluate_many(self, keys):
        new = [k for k in dict.fromkeys(keys) if k not in self.cache]
        out = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(fit_predict)(self.clf, f["Xtr"], f["ytr"], f["wtr"], f["Xva"], k[0], k[1], self.seed + j)
            for k in new for j, f in enumerate(self.folds))
        J = len(self.folds)
        for i, k in enumerate(new):
            ps = out[i * J:(i + 1) * J]
            ys, ws = [f["yva"] for f in self.folds], [f["wva"] for f in self.folds]
            t = best_threshold(np.concatenate(ys), np.concatenate(ps), np.concatenate(ws))
            fv = [from_counts(*counts(a, b, c, np.array([t])))[FITNESS][0] for a, b, c in zip(ys, ps, ws)]
            self.cache[k] = {"fit": float(np.mean(fv)), "thr": t}
        return [self.cache[k]["fit"] for k in keys]

def run_epos(prob, seed, log, tag):
    random.seed(seed); np.random.seed(seed % (2 ** 32 - 1))
    tb = make_toolbox(seed)
    def evaluate_pop(pop):
        inv = [clip(ind) for ind in pop if not ind.fitness.valid]
        for ind, f in zip(inv, prob.evaluate_many([key(i) for i in inv])):
            ind.fitness.values = (f,)
    pop = tb.population(n=MU); evaluate_pop(pop)
    best, stall = max(i.fitness.values[0] for i in pop), 0
    for gen in range(1, NGEN + 1):
        if len(prob.cache) >= MAX_EVALS:
            break
        off = algorithms.varOr(pop, tb, LAMBDA, CXPB, MUTPB); evaluate_pop(off)
        pop = tools.selBest(pop + off, MU)                        # (mu + lambda) survival
        b = pop[0].fitness.values[0]
        log.append({**tag, "gen": gen, "best_inner_fit": b, "evaluated_configs": len(prob.cache)})
        stall = 0 if b > best + 1e-6 else stall + 1; best = max(best, b)
        if stall >= PATIENCE:
            break
    k_best = max(prob.cache, key=lambda k: (round(prob.cache[k]["fit"], 6), -k[0]))   # ties -> fewer rows
    return k_best, prob.cache[k_best]

# ------------------------------------------------ nested CV (resumable) ------------------------------------------------
PARTIAL = f"{OUT_DIR}/outer_test_results_partial.csv"
_base = os.path.basename(os.path.normpath(OUT_DIR))
_old = sorted(glob.glob(f"/kaggle/input/**/{_base}/outer_test_results*.csv", recursive=True))
_old += [PARTIAL] if os.path.exists(PARTIAL) else []
print("LightGBM params:", LGBM_PARAMS, "| OUT_DIR:", OUT_DIR, flush=True)
rows = []
if _old:
    _r = pd.concat([pd.read_csv(p) for p in _old])
    _n = len(_r)
    _r = _r[_r.get("lgbm_cfg", pd.Series("", index=_r.index)).astype(str) == LGBM_CFG] if "lgbm" in CLASSIFIERS else _r
    rows = _r.drop_duplicates(["classifier", "repeat", "fold"], keep="last").to_dict("records")
    print(f"RESUME: {len(rows)} finished outer folds loaded from {_old}"
          + (f" ({_n - len(_r)} rows of the OLD unstable LightGBM run ignored)" if _n > len(_r) else ""), flush=True)
DONE = {(r["classifier"], int(r["repeat"]), int(r["fold"])) for r in rows}
gen_log, t0, fold_h, stopped = [], time.time(), {}, False

for clf_name in CLASSIFIERS:
    for rep in range(N_REPEATS):
        outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + rep)
        for o, (tr, te) in enumerate(outer.split(X_all, y_all)):
            if (clf_name, rep, o) in DONE or stopped:
                continue
            el = (time.time() - t0) / 3600; est = 1.25 * fold_h.get(clf_name, {"rf": 1.0}.get(clf_name, 0.2))
            if el + est > TIME_BUDGET_H:
                print(f"TIME BUDGET: {el:.2f} h used, next fold ~{est:.2f} h -> stopping cleanly; run again to resume.")
                stopped = True; continue
            tf0 = time.time(); fseed = SEED + 1000 * rep + 10 * o
            tag = {"classifier": clf_name, "method": "EPOS", "repeat": rep, "fold": o}
            tr = tr[~OUTLIER[tr]]                  # outlier rules: training rows only
            # search set: all training frauds + SEARCH_MAJ_MAX legit rows (re-weighted), same seed as the baselines
            rng = np.random.default_rng(fseed)
            mi, ma = tr[y_all[tr] == 1], tr[y_all[tr] == 0]
            ma_s = rng.choice(ma, min(len(ma), SEARCH_MAJ_MAX), replace=False)
            s_idx = np.sort(np.concatenate([mi, ma_s]))
            w_s = np.where(y_all[s_idx] == 0, len(ma) / len(ma_s), 1.0)
            folds = []
            for a, b in StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=fseed).split(s_idx, y_all[s_idx]):
                Xa, Xb = scale(s_idx[a], s_idx[b], w_s[a])
                folds.append(dict(Xtr=Xa, ytr=y_all[s_idx[a]], wtr=w_s[a], Xva=Xb, yva=y_all[s_idx[b]], wva=w_s[b]))
            prob = InnerProblem(clf_name, folds, fseed)
            (n_sel, s_sel), info = run_epos(prob, fseed, gen_log, tag)
            del folds, prob.folds
            # final model: FULL outer-train + synthetic rows (from outer-train frauds only) -> real test fold, once
            Xtr, Xte = scale(tr, te, np.ones(len(tr)))
            p = fit_predict(clf_name, Xtr, y_all[tr], np.ones(len(tr)), Xte, n_sel, s_sel, fseed + 99, n_jobs=N_JOBS)
            m = metrics(y_all[te], p, info["thr"])
            fold_h[clf_name] = (time.time() - tf0) / 3600
            rows.append({**tag, "lgbm_cfg": LGBM_CFG, "sel_n": n_sel, "sel_sigma": s_sel, "threshold": info["thr"], "inner_fit": info["fit"],
                         "evaluated_configs": len(prob.cache), "train_rows": len(tr), "test_rows": len(te),
                         "fold_min": 60 * fold_h[clf_name], **m})
            print(f"{clf_name} | repeat {rep} fold {o + 1}: n={n_sel} sigma={s_sel:.3f} thr={info['thr']:.3f} "
                  f"inner {FITNESS}={info['fit']:.4f} | TEST F1={m['F1']:.4f} MCC={m['MCC']:.4f} PR-AUC={m['PR_AUC']:.4f}"
                  f" | fold {60 * fold_h[clf_name]:.1f} min, total {(time.time() - t0) / 60:.1f} min", flush=True)
            pd.DataFrame(rows).to_csv(PARTIAL, index=False)
            del Xtr, Xte

res = pd.DataFrame(rows)
res.to_csv(f"{OUT_DIR}/outer_test_results.csv", index=False)
pd.DataFrame(gen_log).to_csv(f"{OUT_DIR}/generation_log.csv", index=False)
summ = res.groupby(["classifier", "method"])[METRICS].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.reset_index().to_csv(f"{OUT_DIR}/EPOS_paysim_summary_mean_std.txt", index=False)
NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "lgbm": "LightGBM", "xgb": "XGBoost"}
for c_ in [c for c in CLASSIFIERS if c in set(res.classifier)]:
    r_ = res[res.classifier == c_]
    print(f"\n===== EPOS + {NAMES.get(c_, c_)} on PaySim (Financial_Fraud_Detection): mean +- std over {len(r_)} outer test folds (real data) =====")
    for mt in METRICS:
        print(f"{mt:13s}: {r_[mt].mean():.4f} +- {r_[mt].std():.4f}")
    print(f"selected n: median {r_.sel_n.median():.0f} | selected sigma: median {r_.sel_sigma.median():.3f}")
left = len(CLASSIFIERS) * OUTER_FOLDS * N_REPEATS - len(res[res.classifier.isin(CLASSIFIERS)])
print(f"\ntotal time {(time.time() - t0) / 60:.1f} min | results in {OUT_DIR}/"
      + (f" | {left} outer folds NOT finished yet -> run again to resume" if left else " | all folds finished"))
