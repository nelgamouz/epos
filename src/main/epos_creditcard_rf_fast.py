# =====================================================================================================
# EPOS (original noise-augmented oversampling + evolutionary search) on Credit Card Fraud (mlg-ulb/creditcardfraud)
# LEAKAGE-FREE version with nested cross-validation and mean +- std of all metrics.
# FAST RANDOM-FOREST VERSION (fits in one 12 h Kaggle session): identical protocol, folds, seeds, search set, RF
# settings and metrics as epos_creditcard_nested.py; ONLY the evolutionary search budget is smaller:
#   (mu+lambda) = (8+8), max 6 generations, patience 2, HARD CAP of 30 distinct (n, sigma) configurations per
#   outer fold (30 configs x 3 inner folds = 90 RF fits), search grid n step 10, sigma step 0.01.
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
# Run on Kaggle:   !pip install -q deap      then      !python epos_creditcard_rf_fast.py
#   CLASSIFIERS=lr,rf,lgbm,xgb (default) | FITNESS=F1 (default, as original EPOS) or MCC (as the baselines)
#   CREDITCARD_CSV=/path/to/creditcard.csv   (otherwise searched in /kaggle/input or downloaded with kagglehub)
#   Long runs: the script stops cleanly before TIME_BUDGET_H hours (default 11) and saves everything; run it again
#   with the same OUT_DIR (in Kaggle: add the previous output as input) -> finished folds are skipped.
# Outputs (OUT_DIR): outer_test_results.csv, EPOS_creditcard_summary_mean_std.txt (same format as the baseline summaries ->
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
OUT_DIR = os.environ.get("OUT_DIR", "epos_creditcard_rf_results"); os.makedirs(OUT_DIR, exist_ok=True)
CLASSIFIERS = os.environ.get("CLASSIFIERS", "rf").split(",")
OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 3, 2      # same as the baselines
SEARCH_MAJ_MAX = 60_000                     # legitimate rows in the search set (same as the baselines)
FITNESS = os.environ.get("FITNESS", "F1")         # "F1" (original EPOS) or "MCC" (fitness used by the baselines)
N_RANGE = (50, 300)                                # number of synthetic frauds (original EPOS)
SIGMA_RANGE = (0.01, 0.20)                         # noise std in standardised units (original: fixed 0.05)
MU, LAMBDA, NGEN, CXPB, MUTPB = 8, 8, 6, 0.7, 0.3
PATIENCE = 2                                       # stop if the best inner fitness does not improve
MAX_EVALS = int(os.environ.get("MAX_EVALS", 30))  # HARD cap: distinct (n, sigma) configurations per outer fold
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))
N_JOBS = -1
# classifier settings = the baseline scripts (keep them identical, otherwise the comparison is not fair)
LR_PARAMS = {"max_iter": 2000, "C": 1.0}
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"}
RF_MAX_SAMPLES = 300_000                           # bootstrap rows per tree (as in the baselines)
LGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
               "subsample_freq": 1, "colsample_bytree": 0.8}
XGB_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "logloss"}

# ------------------------------------------------ data ------------------------------------------------
def find_csv():
    if os.environ.get("CREDITCARD_CSV"):
        return os.environ["CREDITCARD_CSV"]
    hits = sorted(glob.glob("/kaggle/input/**/creditcard.csv", recursive=True)) + \
        (["creditcard.csv"] if os.path.exists("creditcard.csv") else [])
    if hits:
        return hits[0]
    import kagglehub                                # fallback: download from Kaggle
    return os.path.join(kagglehub.dataset_download("mlg-ulb/creditcardfraud"), "creditcard.csv")

CSV = find_csv()
df = pd.read_csv(CSV)
n_raw = len(df)
df = df.drop_duplicates().reset_index(drop=True)          # exact duplicates in train and test = leakage
y_all = df["Class"].astype(np.int8).values
X_df = df.drop(columns=["Class"])
X_df["Amount"] = np.log1p(X_df["Amount"].clip(lower=0))  # row-wise, nothing fitted -> leakage-free
X_all = X_df.values.astype(np.float64)
N_CONT = X_all.shape[1]                                   # all 30 features (Time, V1-V28, Amount) are continuous
del df, X_df
print(f"data: {CSV} | raw rows {n_raw}, duplicates removed {n_raw - len(y_all)}")

n_pos = int(y_all.sum())
print(f"Credit Card Fraud (mlg-ulb/creditcardfraud): n={len(y_all)}, d={X_all.shape[1]} ({N_CONT} continuous), frauds={n_pos} "
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

def key(ind):                                       # n on a step-10 grid, sigma on a 0.01 grid -> more cache hits
    return int(round(ind[0] / 10) * 10), round(round(ind[1] / 0.01) * 0.01, 4)

class InnerProblem:
    """fitness = mean inner-fold FITNESS at ONE threshold chosen on the pooled inner out-of-fold predictions"""
    def __init__(self, clf_name, folds, seed):
        self.clf, self.folds, self.seed, self.cache = clf_name, folds, seed, {}

    def evaluate_many(self, keys):
        new = [k for k in dict.fromkeys(keys) if k not in self.cache]
        new = new[:max(0, MAX_EVALS - len(self.cache))]          # hard budget cap
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
        return [self.cache[k]["fit"] if k in self.cache else -1.0 for k in keys]   # over budget -> discarded

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
rows = []
if _old:
    rows = (pd.concat([pd.read_csv(p) for p in _old]).drop_duplicates(["classifier", "repeat", "fold"], keep="last")
            .to_dict("records"))
    print(f"RESUME: {len(rows)} finished outer folds loaded from {_old}")
DONE = {(r["classifier"], int(r["repeat"]), int(r["fold"])) for r in rows}
gen_log, t0, fold_h, stopped = [], time.time(), {}, False

for clf_name in CLASSIFIERS:
    for rep in range(N_REPEATS):
        outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + rep)
        for o, (tr, te) in enumerate(outer.split(X_all, y_all)):
            if (clf_name, rep, o) in DONE or stopped:
                continue
            el = (time.time() - t0) / 3600; est = 1.15 * fold_h.get(clf_name, {"rf": 0.75}.get(clf_name, 0.2))
            if el + est > TIME_BUDGET_H:
                print(f"TIME BUDGET: {el:.2f} h used, next fold ~{est:.2f} h -> stopping cleanly; run again to resume.")
                stopped = True; continue
            tf0 = time.time(); fseed = SEED + 1000 * rep + 10 * o
            tag = {"classifier": clf_name, "method": "EPOS", "repeat": rep, "fold": o}
            tr = tr
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
            rows.append({**tag, "sel_n": n_sel, "sel_sigma": s_sel, "threshold": info["thr"], "inner_fit": info["fit"],
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
summ.reset_index().to_csv(f"{OUT_DIR}/EPOS_creditcard_summary_mean_std.txt", index=False)
NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "lgbm": "LightGBM", "xgb": "XGBoost"}
for c_ in [c for c in CLASSIFIERS if c in set(res.classifier)]:
    r_ = res[res.classifier == c_]
    print(f"\n===== EPOS + {NAMES.get(c_, c_)} on Credit Card Fraud (mlg-ulb/creditcardfraud): mean +- std over {len(r_)} outer test folds (real data) =====")
    for mt in METRICS:
        print(f"{mt:13s}: {r_[mt].mean():.4f} +- {r_[mt].std():.4f}")
    print(f"selected n: median {r_.sel_n.median():.0f} | selected sigma: median {r_.sel_sigma.median():.3f}")
left = len(CLASSIFIERS) * OUTER_FOLDS * N_REPEATS - len(res[res.classifier.isin(CLASSIFIERS)])
print(f"\ntotal time {(time.time() - t0) / 60:.1f} min | results in {OUT_DIR}/"
      + (f" | {left} outer folds NOT finished yet -> run again to resume" if left else " | all folds finished"))
