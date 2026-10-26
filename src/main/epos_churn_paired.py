# =====================================================================================================
# EPOS (original noise-augmented oversampling + evolutionary search) on Telco Customer Churn
# LEAKAGE-FREE nested CV, PAIRED with the churn baselines (baselines_churn.py): every outer test fold is identical.
# Classifiers: Logistic Regression, Random Forest, LightGBM, XGBoost (choose with CLASSIFIERS=...)
#
#   Data      : same cleaning as baselines_churn.py (drop customerID, TotalCharges -> numeric, EXACT DUPLICATES REMOVED)
#   Outer CV  : StratifiedKFold(5, shuffle, random_state=SEED+rep) x 3 repeats = 15 test folds (= baselines)
#   Inner CV  : StratifiedKFold(5, shuffle, random_state=SEED+1000*rep+10*fold) on the FULL outer-train (= baselines)
#   Preproc.  : same ColumnTransformer as the baselines (median-impute + StandardScaler for continuous, binary as is,
#               one-hot for categoricals), fitted on the training part of every split only
#   Synthetic : original EPOS - random training churners + independent Gaussian noise N(0, sigma) on the standardised
#               continuous features (binary / one-hot columns copied -> valid records); added to the training part only
#   Threshold : F1-optimal threshold on the pooled inner out-of-fold predictions (same rule as the baselines)
#   Fitness   : mean inner-fold F1 at that threshold (original EPOS; FITNESS=MCC -> same fitness as the baselines)
#   Classifier: identical settings to baselines_churn.py
#   Final     : best (n, sigma) -> model on the FULL outer-train + n synthetic rows -> ONE evaluation on the test fold.
#
# Run on Kaggle:   !pip install -q deap      then      !python epos_churn_paired.py
#   CHURN_CSV=/path/to/Telco-Customer-Churn.csv   (otherwise searched in /kaggle/input or downloaded with kagglehub)
#   Stops cleanly before TIME_BUDGET_H hours (default 11) and resumes from earlier output (same OUT_DIR as input).
# Outputs (OUT_DIR): outer_test_results.csv, EPOS_churn_summary_mean_std.txt (same format as the baseline summaries ->
#   summary_table.py), generation_log.csv
# =====================================================================================================
import os, glob, time, random, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from deap import base, creator, tools, algorithms
warnings.filterwarnings("ignore")

# ------------------------------------------------ settings ------------------------------------------------
SEED = 42
OUT_DIR = os.environ.get("OUT_DIR", "epos_churn_paired_results"); os.makedirs(OUT_DIR, exist_ok=True)
CLASSIFIERS = os.environ.get("CLASSIFIERS", "lr,rf,lgbm,xgb").split(",")
OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 5, 3      # same as baselines_churn.py (15 paired outer folds)
FITNESS = os.environ.get("FITNESS", "F1")         # "F1" (original EPOS) or "MCC" (fitness used by the baselines)
N_RANGE = (50, 300)                                # number of synthetic churners (original EPOS)
SIGMA_RANGE = (0.01, 0.20)                         # noise std in standardised units (original: fixed 0.05)
MU, LAMBDA, NGEN, CXPB, MUTPB = 12, 24, 10, 0.7, 0.3
PATIENCE = 3                                       # stop if the best inner fitness does not improve
MAX_EVALS = 100                                    # max. distinct (n, sigma) configurations per outer fold
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))
N_JOBS = -1
# classifier settings = the baseline scripts (keep them identical, otherwise the comparison is not fair)
LR_PARAMS = {"max_iter": 2000, "C": 1.0}
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2}
RF_MAX_SAMPLES = 10 ** 12                          # no row sub-sampling (as in baselines_churn.py)
LGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "max_bin": 63, "subsample": 0.8,
               "subsample_freq": 1, "colsample_bytree": 0.8}
XGB_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8, "eval_metric": "logloss"}

# ------------------------------------------------ data ------------------------------------------------
def find_csv():
    if os.environ.get("CHURN_CSV"):
        return os.environ["CHURN_CSV"]
    hits = sorted(glob.glob("/kaggle/input/**/Telco-Customer-Churn.csv", recursive=True)) + \
        sorted(glob.glob("/kaggle/input/**/*Telco*Churn*.csv", recursive=True))
    if hits:
        return hits[0]
    import kagglehub                                # fallback: same Kaggle source as baselines_churn.py
    return os.path.join(kagglehub.dataset_download("nourddineelgamouz/churn"), "Telco-Customer-Churn.csv")

CSV = find_csv()
df = pd.read_csv(CSV); df.columns = [c.strip() for c in df.columns]
df = df.drop(columns=[c for c in ["customerID", "CustomerID"] if c in df.columns])
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
                                       errors="coerce").fillna(0.0)
y_all = df["Churn"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(np.int8).values
X_df = df.drop(columns=["Churn"])
for c in X_df.columns:
    if not pd.api.types.is_numeric_dtype(X_df[c]):
        X_df[c] = X_df[c].astype(str).str.strip()
n_raw = len(y_all)
dup = X_df.assign(_y=y_all).duplicated().values         # exact duplicates (same rule as baselines_churn.py)
X_df, y_all = X_df[~dup].reset_index(drop=True), y_all[~dup]
cat = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
num = [c for c in X_df.columns if c not in cat]
binary = [c for c in num if X_df[c].nunique() <= 2]
cont = [c for c in num if c not in binary]
PREP = ColumnTransformer(
    [("cont", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cont),
     ("bin", SimpleImputer(strategy="most_frequent"), binary),
     ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)],
    sparse_threshold=0.0)
N_CONT = len(cont)                                        # continuous columns come first after PREP
del df
print(f"data: {CSV} | raw rows {n_raw}, exact duplicates removed {int(dup.sum())}")
n_pos = int(y_all.sum())
print(f"Telco Churn: n={len(y_all)}, raw features={X_df.shape[1]} ({N_CONT} continuous: {cont}), churners={n_pos} "
      f"({100 * n_pos / len(y_all):.1f}%) | classifiers {CLASSIFIERS} | fitness {FITNESS}")

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

def scale(tr, va, w_tr=None):
    """same preprocessing as baselines_churn.py, fitted on the TRAINING rows only"""
    pp = clone(PREP).fit(X_df.iloc[tr])
    return pp.transform(X_df.iloc[tr]).astype(np.float32), pp.transform(X_df.iloc[va]).astype(np.float32)

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

def best_threshold(y, p, w=None):
    """F1-optimal threshold on the pooled inner OOF predictions - the exact rule of baselines_churn.py"""
    P, R_, T = precision_recall_curve(y, p)
    if len(T) == 0:
        return 0.5
    return float(T[np.argmax(2 * P[:-1] * R_[:-1] / np.clip(P[:-1] + R_[:-1], 1e-12, None))])

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
        for o, (tr, te) in enumerate(outer.split(X_df, y_all)):
            if (clf_name, rep, o) in DONE or stopped:
                continue
            el = (time.time() - t0) / 3600; est = 1.25 * fold_h.get(clf_name, {"rf": 1.0}.get(clf_name, 0.2))
            if el + est > TIME_BUDGET_H:
                print(f"TIME BUDGET: {el:.2f} h used, next fold ~{est:.2f} h -> stopping cleanly; run again to resume.")
                stopped = True; continue
            tf0 = time.time(); fseed = SEED + 1000 * rep + 10 * o
            tag = {"classifier": clf_name, "method": "EPOS", "repeat": rep, "fold": o}
            tr = tr
            # inner CV on the FULL outer-train (same split and seed as baselines_churn.py)
            folds = []
            for a, b in StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=fseed).split(tr, y_all[tr]):
                Xa, Xb = scale(tr[a], tr[b])
                folds.append(dict(Xtr=Xa, ytr=y_all[tr[a]], wtr=np.ones(len(a)), Xva=Xb, yva=y_all[tr[b]],
                                  wva=np.ones(len(b))))
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
summ.reset_index().to_csv(f"{OUT_DIR}/EPOS_churn_summary_mean_std.txt", index=False)
NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "lgbm": "LightGBM", "xgb": "XGBoost"}
for c_ in [c for c in CLASSIFIERS if c in set(res.classifier)]:
    r_ = res[res.classifier == c_]
    print(f"\n===== EPOS + {NAMES.get(c_, c_)} on Telco Churn: mean +- std over {len(r_)} outer test folds (real data) =====")
    for mt in METRICS:
        print(f"{mt:13s}: {r_[mt].mean():.4f} +- {r_[mt].std():.4f}")
    print(f"selected n: median {r_.sel_n.median():.0f} | selected sigma: median {r_.sel_sigma.median():.3f}")
left = len(CLASSIFIERS) * OUTER_FOLDS * N_REPEATS - len(res[res.classifier.isin(CLASSIFIERS)])
print(f"\ntotal time {(time.time() - t0) / 60:.1f} min | results in {OUT_DIR}/"
      + (f" | {left} outer folds NOT finished yet -> run again to resume" if left else " | all folds finished"))
