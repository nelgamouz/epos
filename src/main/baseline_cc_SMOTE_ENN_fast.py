# ===== baseline_cc_SMOTE_ENN_fast.py : runs only "SMOTE-ENN" (same folds/seeds as all other Credit Card scripts) =====
# FAST VERSION - gives EXACTLY the same results as baseline_cc_SMOTE_ENN.py, but:
#   (1) fit cache: inner SMOTE-ENN configurations whose cleaned training set is identical (same rows kept, same
#       classifier, same seed) are fitted only once (the fit is deterministic -> identical predictions)
#   (2) ENN neighbour cache keeps all inner SMOTE sets of all folds (shared by all classifiers, int32)
#   (3) final outer-train ENN computed once per (data, enn_k, kind_sel) and reused by the other classifiers
#   (4) fast classifiers first (lr, lgbm, xgb, then rf) -> they are finished even if the session stops
#   (5) TIME_BUDGET_H (default 11.0 h): no new outer fold is started if it would not finish in time; the script then
#       saves everything and ends NORMALLY (Kaggle keeps the output).  Run again -> it resumes automatically:
#       RESUME_CSV may list several files / globs (comma separated); without it the script also looks for earlier
#       outputs of the same OUT_DIR name in /kaggle/input/** (add the previous session's output as input).
"""
TUNED BASELINES for the EPOS-G study  (No oversampling, Class weights, SMOTE, ADASYN, Borderline-SMOTE, SMOTE-ENN)
x classifiers (Logistic Regression, Random Forest, LightGBM, XGBoost)
=====================================================================================================
This file is the EPOS-G script with the baseline methods added; data preparation, outer/inner folds, seeds,
search set, threshold tuning, metrics, costs and statistics are IDENTICAL, so every outer fold here is paired
with the same outer fold of the EPOS-G run (same classifier, repeat, fold) -> use compare_results.py.
SMOTE-family tuning grid (fixed before running):  r = n_syn / n_minority in R_GRID,  k in {3,5,7}
  (Borderline-SMOTE: k in {3,5}, m = 10, kind in {borderline-1, borderline-2};
   SMOTE-ENN: k in {3,5}, ENN neighbours in {3,5}, ENN rule kind_sel in {all, mode}, ENN cleans both classes);
  selection: best inner FITNESS (ties -> fewer synthetic rows), threshold = inner-OOF F1-optimal threshold.
  Categorical / one-hot columns of interpolated rows are projected back to valid categories
  (binary -> nearest value, one-hot block -> argmax), i.e. every synthetic row is a valid record.
---------------------------------------------------------------------------------------------------------

EPOS-G (geometry-aware EPOS)  -  Credit Card Fraud (mlg-ulb/creditcardfraud)
============================================================================
All 30 features are continuous -> the metric is the Euclidean distance in the standardised space (scaler fitted
on the training part of each fold).  Geometry is measured against a FIXED-SIZE legitimate reference sample
(GEO_MAJ_REF rows of the fold's training legit rows, + all training frauds), so distances have the same scale in
the inner search folds (~40k legit) and in the final outer fold (~227k legit).
EPOS-G generator, theta = (r, sigma, c, gamma, eta):
  (1) kernel-density view + closed-form bandwidth.  x~ = x_b + sigma * h_S * a_b * L_c z,  z ~ N(0, I_d)
        Sigma_c = L_c L_c^T = (1 - c) diag(s_j^2) + c * S+_LedoitWolf          (as in EPOS-3D)
        h_S = (4 / ((d + 2) n_+))^(1/(d+4))       multivariate normal-reference (Silverman/Scott) bandwidth
        a_i = d_k(x_i) / median_l d_k(x_l)        variable-kernel factor (Breiman, Meisel & Purcell 1977),
                                                  d_k = Euclidean distance to the k-th nearest fraud
      sigma = 1, c = 1 is the AMISE-optimal Gaussian KDE for normal data; the EA only corrects it.
  (2) seed weighting by local safety: m_i = #legit among the k nearest neighbours of fraud i,
        P(seed = i) ~ exp(gamma * m_i / k) * 1[m_i < k]     (gamma<0 safe, 0 uniform, >0 border; outliers never seeds)
  (3) nearest-neighbour (Voronoi) safety filter: accept x~ iff d(x~, nearest fraud) <= eta * d(x~, nearest legit)
  EPOS-MO-G: NSGA-II over (r, sigma, c, gamma, eta, t). Ablations: -noAdapt (a_i = 1), -noSeedW (gamma = 0),
  -noFilter.  KDE-Silverman: no search (r = R_MAX, sigma = 1, c = 1, uniform seeds, no filter).
  Extra outputs: minority_typology.csv, synthetic_geometry.csv (syn_overlap = share of synthetic rows whose
  nearest reference neighbour is legit; syn_accept = filter acceptance rate), geometry_genes.png.
---------------------------------------------------------------------------------------------------------
(inherited framework description)
Noise-augmented oversampling tuned by a multi-objective evolutionary algorithm (NSGA-II), leakage-free.

GENERATORS (built from the TRAINING part of each fold only; all 30 features are continuous)
  original EPOS (EPOS-1D):      x_syn = x_b + eps,   eps ~ N(0, sigma^2 I),  b ~ U{training frauds},
                                n in [50, 300] synthetic rows, sigma = 0.05 (standardised units)
  EPOS-3D generator, theta = (r, sigma, c):
      n_syn = round(r * n_fraud),   x_syn = x_b + eps,   eps ~ N(0, sigma^2 * Sigma_c)
      Sigma_c = (1 - c) * diag(s_1^2, ..., s_d^2) + c * S+
          s_j^2 = variance of feature j among training frauds   (per-feature, class-conditional noise scale)
          S+    = Ledoit-Wolf shrunk covariance of the training frauds (noise follows the fraud correlation structure)
      c = 0 -> independent per-feature noise,  c = 1 -> full-covariance Gaussian kernel (smoothed bootstrap)

METHODS
  No oversampling, Class weights            : threshold tuned on inner OOF predictions
  EPOS-1D                                   : original EPOS, (mu+lambda) EA, single objective
  EPOS-3D                                   : (mu+lambda) EA in (r, sigma, c), single objective (FITNESS)
  EPOS-MO-3D                                : NSGA-II in (r, sigma, c, t), t = decision threshold (a gene, so all
                                              objectives describe the SAME model):
          max_{theta, t} ( F1(theta, t), Accuracy(theta, t), PR-AUC(theta) )   on inner-CV predictions
          one configuration is taken from the inner Pareto front with a rule fixed in advance (SELECTION):
              chebyshev : argmin_x max_k (ideal_k - f_k(x)) / (ideal_k - nadir_k)
              knee      : argmax_x sum_k (f_k(x) - nadir_k) / (ideal_k - nadir_k)
              weighted  : argmax_x sum_k w_k f_k(x)
              constraint: argmax F1 s.t. Accuracy >= Accuracy(No oversampling) - CONSTRAINT_DELTA
  Random-MO-3D                              : random (theta, t) search with the same budget as EPOS-MO-3D
  (optional) Grid-1D, Random-1D, Random-3D

PROTOCOL
  Outer CV (5-fold stratified x N_REPEATS): evaluation only. The test fold is never used by the scaler, the
  generator, the search or any threshold. Duplicated rows are removed before splitting.
  Inner CV (3-fold stratified) on a SEARCH SET = all outer-train frauds + a random subsample of SEARCH_MAJ_MAX
  legitimate rows (seed = SEED + 1000*repeat + 10*fold), legitimate rows re-weighted by the inverse sampling
  fraction, so every inner metric estimates the full-data value. After the search, the final model is retrained
  on the FULL outer-train fold and evaluated once on the FULL outer test fold.

CLAIMS TESTED ON THE OUTER TEST FOLDS
  (i)  Pareto dominance: in how many outer folds is EPOS-MO-3D dominated by a baseline in (F1, Accuracy, PR-AUC)?
  (ii) better-or-equal on most metrics: paired two-sided Wilcoxon per metric (18 metrics), Holm across metrics.
  All metrics are reported: F1, PR-AUC, ROC-AUC, MCC, balanced accuracy, precision, recall, specificity, G-mean,
  accuracy, Brier, log-loss, ECE, amount-based savings and normalised costs for several C_FN/C_FP ratios.

Run:  CREDITCARD_CSV=creditcard.csv python3 epos_mo3d_creditcard.py     (or let kagglehub download it)
"""
# ===================================== 0. CONFIGURATION =====================================
import os, sys, json, time, random, platform, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, log_loss
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")
SEED = 42
OUT_DIR = os.environ.get("OUT_DIR", "baseline_cc_SMOTE_ENN_results")
os.makedirs(OUT_DIR, exist_ok=True)
DATASET_NAME = "Credit Card Fraud"
DROP_TIME = False                  # True -> drop the "Time" column (seconds since the first transaction)

CLASSIFIERS = os.environ.get("CLASSIFIERS", "lr,lgbm,xgb,rf").split(",")   # e.g. CLASSIFIERS=lgbm,xgb (split long runs)
LGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
               "subsample_freq": 1, "colsample_bytree": 0.8}
XGB_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8, "tree_method": "hist"}

OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 3, 2    # 10 paired outer folds -> Wilcoxon can reach p < 0.05
SEARCH_MAJ_MAX = 60_000                          # legitimate rows in the search set (all frauds are kept)
FITNESS = "MCC"                                  # single-objective EPOS fitness: "MCC" | "F1" | "PR_AUC"

# --- search spaces (fixed before running) ---
N_RANGE_1D = (50, 300)             # original EPOS space
SIGMA_1D = 0.05
GRID_STEP = 5                      # only used if "Grid-1D" is in METHODS
R_MAX = 5.0                        # r = n_syn / n_fraud in [0, R_MAX]  (1:1 balance would need r ~ 580)
SIGMA_RANGE_3D = (0.0, 1.0)        # in units of the per-feature fraud std
COV_RANGE_3D = (0.0, 1.0)          # c: 0 = independent noise, 1 = full fraud covariance
# --- EPOS-G (geometry-aware generator) ---
K_GEO = 5                          # neighbours for safety degree, local scale and typology
GEO_MAJ_REF = 40_000               # legit reference rows for all distances (same scale in inner and outer folds)
SIGMA_RANGE_G = (0.0, 1.5)         # multiplier on the Silverman bandwidth h_S (1 = closed-form rule)
GAMMA_RANGE = (-3.0, 3.0)          # seed focus: < 0 safe, 0 uniform, > 0 border
ETA_RANGE = (0.5, 3.0)             # NN safety filter ratio (upper bound = filter off)
ADAPT_CLIP = (0.25, 4.0)           # bounds of the variable-kernel factor a_i
FILTER_ROUNDS = 6                  # max. resampling rounds to replace rejected synthetic rows
BOUND_TOL = 0.05

# --- single-objective EA ---
MU, LAMBDA, NGEN, PATIENCE = 20, 40, 15, 3
CXPB, MUTPB = 0.5, 0.5
MAX_EVALS_1D, MAX_EVALS_3D = 251, 150
CHUNK = 32
N_JOBS = -1
FINAL_N_JOBS = -1                  # threads for the final model on the full outer-train fold

# --- multi-objective EPOS ---
OBJECTIVES = ["F1", "Accuracy", "PR_AUC"]   # maximised jointly (any of F1, Accuracy, MCC, Balanced_Acc, Precision,
                                            # Recall, Specificity, G_mean, PR_AUC, ROC_AUC)
SELECTION = "chebyshev"                     # "chebyshev" | "knee" | "weighted" | "constraint"   (fix BEFORE running)
WEIGHTS = None                              # "weighted": {metric: weight}; None -> equal weights
CONSTRAINT_DELTA = 0.0005                   # "constraint": accuracy may drop at most this much vs No oversampling
THR_RANGE = (0.02, 0.98)
MU_MO, LAMBDA_MO, NGEN_MO = 40, 40, 15
MAX_THETA_MO = 150                          # generator configurations per MO run
THRESHOLD_REFINEMENT = True                 # also archive each configuration's inner F1-/FITNESS-optimal threshold
                                            # (memetic step, costs no training; applied to Random-MO-3D as well)

METHODS = ["SMOTE-ENN"]                 # this file runs ONE baseline method only
# also available: "EPOS-1D", "EPOS-3D", "Random-MO-3D", "Grid-1D", "Random-1D", "Random-3D"
# optional extra methods: "Grid-1D", "Random-1D", "Random-3D"  (a Random-X needs EPOS-X earlier in the list)
MO_REFERENCE = "Class weights"   # reference of the within-run tests (EPOS-G comparison: compare_results.py)
# ---- splitting a long run over several Kaggle sessions (all optional; results are identical to one long run,
#      because every outer fold has its own fixed seeds) ----
#   METHODS="SMOTE,ADASYN"   run only these methods      (No oversampling / Class weights for lgbm & xgb are already
#                                                          in the EPOS-G results -> they can be skipped here)
#   REPEATS="0"  FOLDS="0,1"  run only these repeats / outer folds
#   RESUME: finished (classifier, repeat, fold, method) rows are checkpointed to OUT_DIR/outer_test_results_partial.csv
#           after every method; a restarted run with the same OUT_DIR skips them (or point RESUME_CSV to an old file)
# (METHODS is fixed in this file; the METHODS environment variable is ignored)
REPEAT_IDS = [int(v) for v in os.environ["REPEATS"].split(",")] if os.environ.get("REPEATS") else None
FOLD_IDS = [int(v) for v in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None
if MO_REFERENCE not in METHODS:
    MO_REFERENCE = METHODS[0]
EXTRA_PAIRS = [("EPOS-1D", "Grid-1D"), ("EPOS-3D", "Random-3D"), ("EPOS-MO-3D", "Random-MO-3D"),
               ("EPOS-MO-G-noAdapt", "EPOS-MO-3D")]
ALPHA = 0.05

# --- cost analysis (fraud) ---
REVIEW_COST = 5.0                  # cost of investigating one alert; a missed fraud costs its Amount
COST_RATIOS = [10, 50, 100, 500]   # C_FN / C_FP for the normalised expected cost

random.seed(SEED); np.random.seed(SEED)

# ===================================== 1. DATA =====================================
def load_creditcard():
    if "CREDITCARD_CSV" in os.environ:
        return pd.read_csv(os.environ["CREDITCARD_CSV"])
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "mlg-ulb/creditcardfraud", "creditcard.csv")

df = load_creditcard()
n_raw = len(df)
df = df.drop_duplicates().reset_index(drop=True)          # duplicates in train and test = leakage
y_all = df["Class"].astype(np.int8).values
X_df = df.drop(columns=["Class"] + (["Time"] if DROP_TIME and "Time" in df.columns else []))
amount_all = (df["Amount"].clip(lower=0).values.astype(np.float64) if "Amount" in df.columns
              else np.ones(len(df)))
if "Amount" in X_df.columns:
    X_df["Amount"] = np.log1p(X_df["Amount"].clip(lower=0))   # row-wise, nothing fitted -> leakage-free
X_all = X_df.values.astype(np.float64)
D = X_all.shape[1]
n_pos = int(y_all.sum()); RHO = (len(y_all) - n_pos) / n_pos
nmin_outer = n_pos * (OUTER_FOLDS - 1) / OUTER_FOLDS
nmin_inner = nmin_outer * (INNER_FOLDS - 1) / INNER_FOLDS
print(f"{DATASET_NAME}: n={len(y_all)} (duplicates removed: {n_raw - len(y_all)}), d={D}, frauds={n_pos} "
      f"({100 * n_pos / len(y_all):.3f}%), imbalance ratio rho={RHO:.1f}:1")
print(f"   frauds: outer-train ~{nmin_outer:.0f}, inner-train ~{nmin_inner:.0f} | original space n in {N_RANGE_1D} = "
      f"ratio {N_RANGE_1D[0] / nmin_inner:.2f}-{N_RANGE_1D[1] / nmin_inner:.2f} of the inner-train frauds")
print(f"   3-D space: r in [0, {R_MAX}], sigma in {SIGMA_RANGE_3D}, c in {COV_RANGE_3D} | MO objectives {OBJECTIVES}, "
      f"selection = {SELECTION}")

def make_fold(tr, va, w_tr=None, w_va=None):
    """Scaler and fraud covariance are fitted on the TRAINING part only."""
    w_tr = np.ones(len(tr)) if w_tr is None else w_tr
    w_va = np.ones(len(va)) if w_va is None else w_va
    sc = StandardScaler().fit(X_all[tr], sample_weight=w_tr)
    f = dict(Xtr=sc.transform(X_all[tr]).astype(np.float32), ytr=y_all[tr], wtr=w_tr,
             Xva=sc.transform(X_all[va]).astype(np.float32), yva=y_all[va], wva=w_va, vva=amount_all[va])
    Xmin = f["Xtr"][f["ytr"] == 1].astype(np.float64)
    s = Xmin.std(0, ddof=1); s[s < 1e-6] = 1e-6
    S = LedoitWolf().fit(Xmin).covariance_
    f["gen"] = dict(Xmin=Xmin.astype(np.float32), s=s, S=S)
    f["geo"] = geometry(f)
    return f

def silverman(d, n):
    return float((4.0 / ((d + 2.0) * max(n, 2))) ** (1.0 / (d + 4.0)))

def geometry(f):
    """Euclidean k-NN geometry of the training frauds vs a fixed-size legit reference sample (training part only)."""
    from sklearn.neighbors import NearestNeighbors
    X, y = f["Xtr"], f["ytr"]; Xmin = f["gen"]["Xmin"]; nf = len(Xmin)
    leg = np.flatnonzero(y == 0)
    if len(leg) > GEO_MAJ_REF:
        leg = np.sort(np.random.default_rng(SEED + len(y)).choice(leg, GEO_MAJ_REF, replace=False))
    Xmaj = X[leg]; k = min(K_GEO, max(nf - 1, 1))
    Zref = np.vstack([Xmin, Xmaj]); yref = np.concatenate([np.ones(nf, np.int8), np.zeros(len(Xmaj), np.int8)])
    idx = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(Zref).kneighbors(Xmin, return_distance=False)[:, 1:]
    m = (yref[idx] == 0).sum(1)
    if nf > k:
        dk = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(Xmin).kneighbors(Xmin)[0][:, k]
        a = np.clip(dk / max(np.median(dk), 1e-6), *ADAPT_CLIP)
    else:
        a = np.ones(nf)
    s_ = m / k
    typ = {"safe": float(np.mean(s_ <= 0.2)), "borderline": float(np.mean((s_ > 0.2) & (s_ <= 0.6))),
           "rare": float(np.mean((s_ > 0.6) & (s_ < 1))), "outlier": float(np.mean(s_ >= 1))}
    return dict(k=k, m=m, a=a, h_S=silverman(D, nf), typ=typ, chol={},
                nn_min=NearestNeighbors(n_neighbors=1, algorithm="brute").fit(Xmin),
                nn_maj=NearestNeighbors(n_neighbors=1, algorithm="brute").fit(Xmaj))

def overlap_rate(fold, S):
    """share of synthetic rows whose nearest reference neighbour is a legit transaction"""
    if len(S) == 0:
        return np.nan
    G = fold["geo"]
    return float(np.mean(G["nn_maj"].kneighbors(S)[0][:, 0] < G["nn_min"].kneighbors(S)[0][:, 0]))

def geo_oversample(fold, spec, rng):
    """EPOS-G generator; returns (synthetic rows, acceptance rate of the safety filter)."""
    g, G = fold["gen"], fold["geo"]; Xmin = g["Xmin"]; nf = len(Xmin)
    n = int(round(spec.get("r", 0.0) * nf))
    if n <= 0 or nf == 0:
        return np.empty((0, D), np.float32), 1.0
    valid = G["m"] < G["k"]
    if not valid.any():
        valid = np.ones(nf, bool)
    lw = spec.get("gamma", 0.0) * G["m"] / G["k"]
    pw = np.exp(lw - lw.max()) * valid; pw /= pw.sum()
    sigma, c = spec.get("sigma", 0.0), spec.get("c", 0.0)
    if sigma > 0:
        L = np.linalg.cholesky((1 - c) * np.diag(g["s"] ** 2) + c * g["S"] + 1e-8 * np.eye(D))
    eta = spec.get("eta", ETA_RANGE[1]); filt = eta < ETA_RANGE[1] - 1e-9
    out, need, made, kept = [], n, 0, 0
    for _ in range(FILTER_ROUNDS if filt else 1):
        acc_est = kept / made if made else 0.67
        m_ = min(int(np.ceil(need / max(acc_est, 0.05) * 1.2)) + 8, 20 * n) if filt else need
        b = rng.choice(nf, m_, p=pw)
        S = Xmin[b].astype(np.float64)
        if sigma > 0:
            sc = sigma * G["h_S"] * (G["a"][b] if spec.get("adapt", True) else np.ones(m_))
            S += sc[:, None] * (rng.standard_normal((m_, D)) @ L.T)
        S = S.astype(np.float32); made += m_
        if filt:
            ok = G["nn_min"].kneighbors(S)[0][:, 0] <= eta * G["nn_maj"].kneighbors(S)[0][:, 0]
            S = S[ok]
        kept += len(S); out.append(S[:need]); need -= len(out[-1])
        if need <= 0:
            break
    return np.vstack(out), kept / max(made, 1)

R_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]        # same range as EPOS (r <= R_MAX); 1:1 balance would need r ~ 580

# ===================================== SMOTE-FAMILY BASELINES (tuned) =====================================
# Every sampler is tuned on the SAME inner folds, with the SAME fitness (FITNESS at the inner-OOF F1 threshold) and
# the SAME selection rule as the single-objective EPOS variants.  The amount of oversampling is parametrised like
# EPOS (r = n_syn / n_minority of the training part), so the selected r means the same thing in the search folds and
# in the final outer-train fit.  Only the synthetic rows are taken from imblearn (it appends them after the
# original rows); original rows keep their sample weights, synthetic rows get weight 1.
SAMPLER_GRID = {
    "SMOTE": [{"r": r, "k": k} for r in R_GRID for k in (3, 5, 7)],
    "ADASYN": [{"r": r, "k": k} for r in R_GRID for k in (3, 5, 7)],
    "Borderline-SMOTE": [{"r": r, "k": k, "m": 10, "kind": kd} for r in R_GRID for k in (3, 5)
                         for kd in ("borderline-1", "borderline-2")],
    "SMOTE-ENN": [{"r": r, "k": k, "enn_k": ek, "kind_sel": ks} for r in R_GRID for k in (3, 5)
                  for ek in (3, 5) for ks in ("all", "mode")]}
SAMPLER_GEN = {"SMOTE": "smote", "ADASYN": "adasyn", "Borderline-SMOTE": "bsmote", "SMOTE-ENN": "smoteenn"}

ENN_FAST_MIN = 200_000     # above this size enn_keep() uses the exact pruned ENN below (same result, much faster)
ENN_BOUND_REF = 50_000     # majority rows used for the upper bound of the k-NN radius

def enn_keep_fast(X, y, spec, n_jobs=1):
    """EXACT Edited Nearest Neighbours (same rule as imblearn: k+1 neighbours, first one dropped as the point itself,
    kind_sel all / mode) with a pruning step.  A majority row whose k nearest neighbours contain no minority row is
    always kept.  For every majority row x:  d_k(x | full data) <= u(x) = d_{k+1}(x | random majority subset), so if
    the nearest minority row is farther than u(x) it cannot be among x's k neighbours -> kept without a full query.
    Only minority rows and the remaining 'candidate' majority rows are queried against the full data."""
    from sklearn.neighbors import NearestNeighbors
    k = int(spec["enn_k"]); i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(12345)
    ref = i0[rng.choice(len(i0), min(len(i0), ENN_BOUND_REF), replace=False)]
    u = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(X[ref]).kneighbors(X[i0])[0][:, -1]
    dmin = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(X[i1]).kneighbors(X[i0])[0][:, 0]
    cand = i0[dmin <= u * (1 + 1e-6) + 1e-9]
    q = np.concatenate([i1, cand])
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(X)
    lab = y[nn.kneighbors(X[q], return_distance=False)[:, 1:]]
    if spec["kind_sel"] == "all":
        ok = np.all(lab == y[q][:, None], axis=1)
    else:                                                     # mode of the neighbour labels (ties -> class 0)
        ok = (np.where(lab.sum(1) * 2 > k, 1, 0) == y[q])
    keep = np.ones(len(y), bool); keep[q[~ok]] = False
    return np.flatnonzero(keep)

def enn_keep(X, y, spec, n_jobs=1):
    """SMOTE-ENN cleaning step: Edited Nearest Neighbours (Wilson 1972) on original + synthetic rows, both classes
    (sampling_strategy="all", as in imblearn's SMOTEENN).  Returns the indices of the rows that are kept."""
    if len(y) > ENN_FAST_MIN:
        fk = (_hash(X, y), int(spec["enn_k"]), spec["kind_sel"])
        if fk in _FINAL_ENN:
            return _FINAL_ENN[fk]
        keep = enn_keep_fast(X, y, spec, n_jobs)
        keep = keep if (y[keep] == 1).any() and (y[keep] == 0).any() else np.arange(len(y))
        if len(_FINAL_ENN) >= 8:
            _FINAL_ENN.pop(next(iter(_FINAL_ENN)))
        _FINAL_ENN[fk] = keep
        return keep if (y[keep] == 1).any() and (y[keep] == 0).any() else np.arange(len(y))
    try:
        # inner search: the 4 ENN settings (enn_k 3/5 x kind_sel all/mode) of one SMOTE set share ONE k-NN query
        # (neighbours for the largest enn_k, cached by the content of X); same rule as imblearn's ENN
        nb = _enn_neighbours(X)
        lab = y[nb[:, 1:int(spec["enn_k"]) + 1]]; k = int(spec["enn_k"])
        ok = (np.all(lab == y[:, None], axis=1) if spec["kind_sel"] == "all"
              else np.where(lab.sum(1) * 2 > k, 1, 0) == y)
        keep = np.flatnonzero(ok)
        return keep if (y[keep] == 1).any() and (y[keep] == 0).any() else np.arange(len(y))
    except ValueError as e:
        SAMPLER_FAILS.append(("enn", str(e)[:120])); return np.arange(len(y))

import hashlib, threading
_ENN_CACHE, _ENN_LOCK = {}, threading.Lock()
ENN_K_MAX = 5                          # largest enn_k of the SMOTE-ENN grid
ENN_CACHE_MAX = int(os.environ.get("ENN_CACHE_MAX", 400))   # ~1 MB per entry (all inner SMOTE sets of 10+ folds)
_FINAL_ENN = {}                        # final outer-train ENN result, reused across classifiers
_FIT_CACHE, _FIT_LOCK, FIT_CACHE_HITS = {}, threading.Lock(), [0, 0]   # [hits, misses]

def _hash(*arrs):
    h = hashlib.blake2b(digest_size=16)
    for a in arrs:
        a = np.ascontiguousarray(a); h.update(str(a.shape).encode()); h.update(a.view(np.uint8))
    return h.hexdigest()

def _enn_neighbours(X):
    from sklearn.neighbors import NearestNeighbors
    key = (X.shape, hashlib.blake2b(np.ascontiguousarray(X).view(np.uint8), digest_size=16).hexdigest())
    with _ENN_LOCK:
        if key in _ENN_CACHE:
            return _ENN_CACHE[key]
    nb = NearestNeighbors(n_neighbors=ENN_K_MAX + 1).fit(X).kneighbors(X, return_distance=False).astype(np.int32)
    with _ENN_LOCK:
        if len(_ENN_CACHE) >= ENN_CACHE_MAX:
            _ENN_CACHE.pop(next(iter(_ENN_CACHE)))
        _ENN_CACHE[key] = nb
    return nb
SAMPLER_FAILS = []                     # (generator, error) of configurations that could not generate rows

def smote_numpy(Xmin, n_new, k, seed):
    """Standard SMOTE (Chawla et al. 2002), same algorithm as imblearn's SMOTE but WITHOUT imblearn (only numpy +
    sklearn NearestNeighbors): k nearest MINORITY neighbours of every minority row (self excluded); each synthetic row
    picks a uniform random (seed, neighbour) pair and a uniform step in [0, 1):  s = x + step * (x_nn - x)."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(seed)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xmin).kneighbors(Xmin, return_distance=False)[:, 1:]
    idx = rng.randint(0, nn.size, size=int(n_new))
    rows, cols = idx // k, idx % k
    steps = rng.uniform(size=(int(n_new), 1))
    Xm = Xmin.astype(np.float32)
    return (Xm[rows] + steps * (Xm[nn[rows, cols]] - Xm[rows])).astype(np.float32)

def smote_family(X, y, n_new, spec, seed):
    """synthetic rows only (n_new requested) from SMOTE / ADASYN / Borderline-SMOTE fitted on (X, y)"""
    n_min = int((y == 1).sum()); empty = np.empty((0, X.shape[1]), np.float32)
    if n_new <= 0 or n_min < 3:
        return empty
    k = int(min(spec["k"], n_min - 1)); strat = {1: n_min + int(n_new)}; gen = spec["gen"]
    if gen in ("smote", "smoteenn"):                       # SMOTE / SMOTE-ENN: no imblearn needed
        return smote_numpy(X[y == 1], n_new, k, seed)
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    try:
        if gen in ("smote", "smoteenn"):                   # SMOTE-ENN: SMOTE step here, ENN step in fit_predict
            s = SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed)
        elif gen == "adasyn":
            # imblearn rounds ADASYN's per-seed counts (rint(ratio_i * n)); for n < n_min almost all round to 0
            # ("No samples will be generated") and for larger n the total overshoots.  We therefore request
            # max(n_new, 2 n_min) rows and keep a uniform random subset of exactly n_new: this preserves ADASYN's
            # density weighting in expectation and gives the same amount of oversampling as all other methods.
            strat = {1: n_min + max(int(n_new), 2 * n_min)}
            s = ADASYN(sampling_strategy=strat, n_neighbors=k, random_state=seed)
        else:
            s = BorderlineSMOTE(sampling_strategy=strat, k_neighbors=k, m_neighbors=int(spec.get("m", 10)),
                                kind=spec["kind"], random_state=seed)
        Xr, yr = s.fit_resample(X, y)
    except (ValueError, RuntimeError) as e:                # e.g. ADASYN: no majority neighbours -> nothing to do
        SAMPLER_FAILS.append((gen, str(e)[:120]))
        return empty
    S = np.asarray(Xr[len(X):], np.float32)
    assert np.all(np.asarray(yr[len(X):]) == 1)
    if len(S) > n_new:
        S = S[np.sort(np.random.default_rng(seed).choice(len(S), int(n_new), replace=False))]
    return S

def project_syn(fold, S):
    return S                                        # all features continuous

# ===================================== 2. GENERATOR + CLASSIFIER =====================================
def noise_oversample(fold, spec, rng):
    g = fold["gen"]; Xmin = g["Xmin"]; nf = len(Xmin)
    n = int(spec["n"]) if "n" in spec else int(round(spec.get("r", 0.0) * nf))
    if n <= 0 or nf == 0:
        return np.empty((0, D), np.float32)
    S = Xmin[rng.integers(0, nf, n)].astype(np.float64)
    sigma = spec.get("sigma", 0.0)
    if sigma > 0:
        if spec.get("gen") == "cov":
            c = spec.get("c", 0.0)
            cov = (1 - c) * np.diag(g["s"] ** 2) + c * g["S"]
            L = np.linalg.cholesky(cov + 1e-8 * np.eye(D))
            S += sigma * rng.standard_normal((n, D)) @ L.T
        else:                                                      # original EPOS: isotropic in standardised units
            S += sigma * rng.standard_normal((n, D))
    return S.astype(np.float32)

RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"}
RF_MAX_SAMPLES = 300_000           # bootstrap rows per tree (keeps RF feasible on ~5 M rows; no effect on small sets)
LR_PARAMS = {"max_iter": 2000, "C": 1.0}

def make_clf(name, n_jobs=1):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=n_jobs, random_state=SEED, **RF_PARAMS)
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**LR_PARAMS)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_jobs=n_jobs, random_state=SEED, eval_metric="logloss", **XGB_PARAMS)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_jobs=n_jobs, random_state=SEED, verbose=-1, **LGBM_PARAMS)

def balanced_weights(y, w):
    """'balanced' class weights computed on the WEIGHTED class masses: each class gets half of the total mass.
    The search set (re-weighted legit subsample) and the full outer-train therefore get the same balance."""
    W, W1 = w.sum(), w[y == 1].sum()
    return w * np.where(y == 1, W / (2 * max(W1, 1e-12)), W / (2 * max(W - W1, 1e-12)))

def fit_predict(clf_name, spec, fold, seed, n_jobs=1, return_syn=False):
    t0 = time.perf_counter(); acc = 1.0
    if spec.get("gen") == "geo":
        S, acc = geo_oversample(fold, spec, np.random.default_rng(seed))
    elif spec.get("gen") in ("smote", "adasyn", "bsmote", "smoteenn"):
        n_ = int(round(spec["r"] * int(fold["ytr"].sum())))
        S = project_syn(fold, smote_family(fold["Xtr"], fold["ytr"], n_, spec, int(seed) % (2 ** 31 - 1)))
    else:
        S = noise_oversample(fold, spec, np.random.default_rng(seed))
    X = np.vstack([fold["Xtr"], S]); y = np.concatenate([fold["ytr"], np.ones(len(S), np.int8)])
    w = np.concatenate([fold["wtr"], np.ones(len(S))])
    fck = None
    if spec.get("gen") == "smoteenn":
        keep = enn_keep(X, y, spec, n_jobs); X, y, w = X[keep], y[keep], w[keep]
        if not return_syn:                         # inner search: identical cleaned set -> identical fit
            fck = (clf_name, int(seed), bool(spec.get("cw", False)), _hash(X, y, w, fold["Xva"]))
            with _FIT_LOCK:
                if fck in _FIT_CACHE:
                    FIT_CACHE_HITS[0] += 1
                    return _FIT_CACHE[fck], len(S), time.perf_counter() - t0
                FIT_CACHE_HITS[1] += 1
    if spec.get("cw", False):
        w = balanced_weights(y, w)
    clf = make_clf(clf_name, n_jobs)
    if clf_name == "rf":
        clf.set_params(max_samples=min(1.0, RF_MAX_SAMPLES / len(y)))
    p = clf.fit(X, y, sample_weight=w).predict_proba(fold["Xva"])[:, 1]
    if fck is not None:
        with _FIT_LOCK:
            if len(_FIT_CACHE) >= 2000:
                _FIT_CACHE.pop(next(iter(_FIT_CACHE)))
            _FIT_CACHE[fck] = p.astype(np.float64)
    if return_syn:
        return p.astype(np.float64), len(S), time.perf_counter() - t0, S, acc
    return p.astype(np.float64), len(S), time.perf_counter() - t0

# ===================================== 3. METRICS + COST (sample-weighted) =====================================
MKEYS = ["F1", "PR_AUC", "ROC_AUC", "MCC", "Balanced_Acc", "Precision", "Recall", "Specificity",
         "G_mean", "Accuracy", "Brier", "LogLoss", "ECE"]
THR_METRICS = ["F1", "Accuracy", "MCC", "Balanced_Acc", "Precision", "Recall", "Specificity", "G_mean"]
assert all(o in THR_METRICS + ["PR_AUC", "ROC_AUC"] for o in OBJECTIVES), "OBJECTIVES must be higher-is-better"

def counts(y, p, t, w):
    yhat = p >= t
    return (w[yhat & (y == 1)].sum(), w[yhat & (y == 0)].sum(), w[~yhat & (y == 1)].sum(), w[~yhat & (y == 0)].sum())

def thr_metrics(y, p, t, w):
    tp, fp, fn, tn = counts(y, p, t, w)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "Accuracy": (tp + tn) / (tp + fp + fn + tn),
            "MCC": (tp * tn - fp * fn) / den if den else 0.0, "Balanced_Acc": (rec + spec) / 2,
            "Precision": prec, "Recall": rec, "Specificity": spec, "G_mean": float(np.sqrt(rec * spec))}

def all_metrics(y, p, t, w=None):
    w = np.ones(len(y)) if w is None else w
    m = thr_metrics(y, p, t, w)
    idx = np.minimum((p * 10).astype(int), 9)
    ece = sum(w[idx == b].sum() / w.sum() * abs(np.average(y[idx == b], weights=w[idx == b]) -
                                                np.average(p[idx == b], weights=w[idx == b]))
              for b in range(10) if (idx == b).any())
    tp, fp, fn, tn = counts(y, p, t, w)
    m.update({"PR_AUC": average_precision_score(y, p, sample_weight=w), "ROC_AUC": roc_auc_score(y, p, sample_weight=w),
              "Brier": float(np.average((p - y) ** 2, weights=w)),
              "LogLoss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=w, labels=[0, 1]), "ECE": float(ece),
              "TP": int(round(tp)), "FP": int(round(fp)), "FN": int(round(fn)), "TN": int(round(tn))})
    return m

def f1_opt_threshold(y, p, w):
    P, R, T = precision_recall_curve(y, p, sample_weight=w)
    if len(T) == 0:
        return 0.5
    f1 = 2 * P[:-1] * R[:-1] / np.clip(P[:-1] + R[:-1], 1e-12, None)
    return float(T[np.argmax(f1)])

def fit_opt_threshold(y, p, w):
    if FITNESS == "F1" or FITNESS == "PR_AUC":
        return f1_opt_threshold(y, p, w)
    qs = np.unique(np.quantile(p[y == 1], np.linspace(0, 1, 101)))      # candidate thresholds at fraud scores
    return float(max(qs, key=lambda t: thr_metrics(y, p, t, w)[FITNESS]))

THR_GRID = np.unique(np.concatenate([np.linspace(0.001, 0.05, 50), np.linspace(0.05, 0.99, 95)]))

def ratio_cost(y, p, t, c, w):
    tp, fp, fn, tn = counts(y, p, t, w)
    return float((c * fn + fp) / (c * max(tp + fn, 1e-12)))          # 1.0 = cost of flagging nothing

def savings(y, p, t, v, w):
    yhat = p >= t
    cost = (w * v)[(y == 1) & ~yhat].sum() + REVIEW_COST * w[yhat].sum()
    return float(1.0 - cost / max((w * v)[y == 1].sum(), 1e-12))

def best_thr(fun):
    return float(THR_GRID[int(np.argmin([fun(t) for t in THR_GRID]))])

def fmt(m, keys=MKEYS):
    return " ".join(f"{k}={m[k]:.4f}" for k in keys)

# ===================================== 4. FITNESS CACHE =====================================
class Problem:
    def __init__(self, clf_name, folds, seed):
        self.clf, self.folds, self.seed, self.cache = clf_name, folds, seed, {}

    def score(self, items):
        new = [(k, s) for k, s in dict(items).items() if k not in self.cache]
        J = len(self.folds)
        for c0 in range(0, len(new), CHUNK):
            batch = new[c0:c0 + CHUNK]
            t0 = time.perf_counter()
            out = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(fit_predict)(self.clf, s, self.folds[j], self.seed + j) for _, s in batch for j in range(J))
            wall = (time.perf_counter() - t0) / max(len(batch), 1)
            for i, (k, s) in enumerate(batch):
                o = out[i * J:(i + 1) * J]
                ys, ps = [f["yva"] for f in self.folds], [x[0] for x in o]
                ws, vs = [f["wva"] for f in self.folds], [f["vva"] for f in self.folds]
                yc, pc, wc = np.concatenate(ys), np.concatenate(ps), np.concatenate(ws)
                thr = fit_opt_threshold(yc, pc, wc)
                fv = ([average_precision_score(a, b, sample_weight=c) for a, b, c in zip(ys, ps, ws)]
                      if FITNESS == "PR_AUC" else [thr_metrics(a, b, thr, c)[FITNESS] for a, b, c in zip(ys, ps, ws)])
                self.cache[k] = {"spec": s, "fit": float(np.mean(fv)), "fit_sd": float(np.std(fv, ddof=1)), "thr": thr,
                                 "nsyn": float(np.mean([x[1] for x in o])), "fit_time": float(sum(x[2] for x in o)),
                                 "wall": wall, "fits": J, "ys": ys, "ps": ps, "ws": ws,
                                 "oof": (yc, pc, wc, np.concatenate(vs))}

class Tracker:
    """One search method; it is charged for every configuration IT requested (shared cache, separate accounting)."""
    def __init__(self, prob):
        self.prob, self.keys, self.curve = prob, [], []

    def request(self, items):
        self.prob.score(items)
        for k, _ in items:
            if k not in self.keys:
                self.keys.append(k)
        self.curve.append((len(self.keys), self.best_fit()))

    def best_key(self):
        c = self.prob.cache
        return max(self.keys, key=lambda k: (c[k]["fit"], -c[k]["nsyn"]))

    def best_fit(self):
        return self.prob.cache[self.best_key()]["fit"]

    def cost(self):
        c = self.prob.cache
        return {"evals": len(self.keys), "fits": sum(c[k]["fits"] for k in self.keys),
                "fit_time_s": sum(c[k]["fit_time"] for k in self.keys), "wall_s": sum(c[k]["wall"] for k in self.keys)}

def key1(v):
    n = int(round(min(max(v[0], N_RANGE_1D[0]), N_RANGE_1D[1])))
    return ("n", n), {"n": n, "sigma": SIGMA_1D}

def key3(v):
    r = round(min(max(v[0], 0.0), R_MAX), 2)
    s = round(min(max(v[1], SIGMA_RANGE_3D[0]), SIGMA_RANGE_3D[1]), 3)
    c = round(min(max(v[2], COV_RANGE_3D[0]), COV_RANGE_3D[1]), 2)
    if r <= 0:
        s, c = 0.0, 0.0
    if s <= 0:
        c = 0.0
    return ("c", r, s, c), {"r": r, "sigma": s, "c": c, "gen": "cov"}

SPACE = {"1D": ([N_RANGE_1D], key1, MAX_EVALS_1D),
         "3D": ([(0.0, R_MAX), SIGMA_RANGE_3D, COV_RANGE_3D], key3, MAX_EVALS_3D)}

def make_keyG(adapt=True, use_gamma=True, use_filter=True):
    bounds = [(0.0, R_MAX), SIGMA_RANGE_G, COV_RANGE_3D] + ([GAMMA_RANGE] if use_gamma else []) + \
             ([ETA_RANGE] if use_filter else [])
    cl = lambda x, b: min(max(x, b[0]), b[1])
    def keyf(v):
        r, s, c = round(cl(v[0], (0.0, R_MAX)), 2), round(cl(v[1], SIGMA_RANGE_G), 3), round(cl(v[2], COV_RANGE_3D), 2)
        i = 3; g = 0.0; e = ETA_RANGE[1]
        if use_gamma:
            g = round(cl(v[i], GAMMA_RANGE), 2); i += 1
        if use_filter:
            e = round(cl(v[i], ETA_RANGE), 2)
        if r <= 0:
            s, c, g, e = 0.0, 0.0, 0.0, ETA_RANGE[1]
        if s <= 0:
            c = 0.0
        return ("g", adapt, r, s, c, g, e), {"r": r, "sigma": s, "c": c, "gamma": g, "eta": e, "adapt": adapt,
                                             "gen": "geo"}
    start = [R_MAX, 1.0, 1.0] + ([0.0] if use_gamma else []) + ([ETA_RANGE[1]] if use_filter else [])
    return bounds, keyf, start

SPACE_START = {}
for _nm, _kw in {"G": {}, "G_noAdapt": {"adapt": False}, "G_noSeedW": {"use_gamma": False},
                 "G_noFilter": {"use_filter": False}}.items():
    _b, _k, _s = make_keyG(**_kw); SPACE[_nm] = (_b, _k, MAX_EVALS_3D); SPACE_START[_nm] = _s
MO_SPACE_OF = {"EPOS-MO-3D": "3D", "Random-MO-3D": "3D", "EPOS-MO-G": "G", "EPOS-MO-G-noAdapt": "G_noAdapt",
               "EPOS-MO-G-noSeedW": "G_noSeedW", "EPOS-MO-G-noFilter": "G_noFilter", "Random-MO-G": "G"}
RANDOM_REF = {"Random-MO-3D": "EPOS-MO-3D", "Random-MO-G": "EPOS-MO-G"}
KDE_SPEC = {"r": R_MAX, "sigma": 1.0, "c": 1.0, "gamma": 0.0, "eta": ETA_RANGE[1], "adapt": False, "gen": "geo"}

if not hasattr(creator, "FitE"):
    creator.create("FitE", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "IndE"):
    creator.create("IndE", list, fitness=creator.FitE)
if not hasattr(creator, "FitMO"):
    creator.create("FitMO", base.Fitness, weights=(1.0,) * len(OBJECTIVES))
if not hasattr(creator, "IndMO"):
    creator.create("IndMO", list, fitness=creator.FitMO)

def toolbox(lo, hi):
    tb = base.Toolbox()
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=[0.15 * (h - l) for l, h in zip(lo, hi)], indpb=0.7)
    return tb

def run_epos(trk, space, seed, label, gen_log, tag):
    bounds, keyf, budget = SPACE[space]
    random.seed(seed)
    lo, hi = [b[0] for b in bounds], [b[1] for b in bounds]
    tb, c = toolbox(lo, hi), trk.prob.cache

    def ev(inds):
        items = [keyf(i) for i in inds]; trk.request(items)
        for i, (k, _) in zip(inds, items):
            i.fitness.values = (c[k]["fit"], c[k]["nsyn"])

    def report(gen, pop):
        k = trk.best_key(); e = c[k]
        gen_log.append({**tag, "method": label, "generation": gen, "evals": len(trk.keys),
                        "pop_mean_fit": float(np.mean([i.fitness.values[0] for i in pop])), **e["spec"],
                        "threshold": e["thr"], "n_syn": e["nsyn"], f"inner_{FITNESS}": e["fit"]})
        print(f"      [{label}] gen {gen:02d} | evals {len(trk.keys):3d} | pop mean {FITNESS} "
              f"{gen_log[-1]['pop_mean_fit']:.4f} | best {e['spec']} n_syn={e['nsyn']:.0f} thr={e['thr']:.3f} "
              f"| inner {FITNESS}={e['fit']:.4f}")

    pop = [creator.IndE([random.uniform(l, h) for l, h in zip(lo, hi)]) for _ in range(MU)]
    ev(pop); report(0, pop)
    best, stall = trk.best_fit(), 0
    for gen in range(1, NGEN + 1):
        if len(trk.keys) >= budget or stall >= PATIENCE:
            break
        off = algorithms.varOr(pop, tb, LAMBDA, CXPB, MUTPB)
        for o in off:
            o[:] = [min(max(x, l), h) for x, l, h in zip(o, lo, hi)]
        ev(off)
        pop = tools.selBest(pop + off, MU)
        report(gen, pop)
        cur = trk.best_fit()
        best, stall = (cur, 0) if cur > best + 1e-9 else (best, stall + 1)

def run_random(trk, space, seed, budget):
    bounds, keyf, _ = SPACE[space]
    rng = random.Random(seed); tries = 0
    while len(trk.keys) < budget and tries < 50 * budget:
        items = []
        while len(items) < CHUNK and len(trk.keys) + len(items) < budget and tries < 50 * budget:
            tries += 1
            k, s = keyf([rng.uniform(l, h) for l, h in bounds])
            if k not in trk.keys and k not in dict(items):
                items.append((k, s))
        if not items:
            break
        trk.request(items)

def run_grid(trk):
    ns = list(range(N_RANGE_1D[0], N_RANGE_1D[1] + 1, GRID_STEP))
    for c0 in range(0, len(ns), CHUNK):
        trk.request([key1([n]) for n in ns[c0:c0 + CHUNK]])

# ===================================== 5. EPOS-MO (NSGA-II, threshold as a gene) =====================================
def mo_full(e, t):
    """All threshold metrics at t + threshold-free PR/ROC-AUC, averaged over the inner folds (sample-weighted)."""
    memo = e.setdefault("mo_memo", {}); t = round(float(t), 3)
    if t not in memo:
        if "tf" not in e:
            e["tf"] = {"PR_AUC": float(np.mean([average_precision_score(a, b, sample_weight=c)
                                                for a, b, c in zip(e["ys"], e["ps"], e["ws"])])),
                       "ROC_AUC": float(np.mean([roc_auc_score(a, b, sample_weight=c)
                                                 for a, b, c in zip(e["ys"], e["ps"], e["ws"])]))}
        per = [thr_metrics(a, b, t, c) for a, b, c in zip(e["ys"], e["ps"], e["ws"])]
        memo[t] = {**{m: float(np.mean([d[m] for d in per])) for m in THR_METRICS}, **e["tf"]}
    return memo[t]

def mo_vec(e, t):
    d = mo_full(e, t)
    return np.array([d[o] for o in OBJECTIVES])

def nondominated(F):
    F = np.asarray(F); keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if np.any(np.all(F >= F[i], 1) & np.any(F > F[i], 1)):
            keep[i] = False
    return np.flatnonzero(keep)

def hypervolume(F, ref=None):
    """Hypervolume (maximisation) w.r.t. ref = origin; metrics clipped to [0, 1]."""
    F = np.clip(np.asarray(F, float), 0, 1); ref = np.zeros(F.shape[1]) if ref is None else ref
    F = F[nondominated(F)]
    try:
        from deap.tools._hypervolume import hv as _hv
        return float(_hv.hypervolume((-F).tolist(), (-ref).tolist()))
    except Exception:
        U = np.random.default_rng(0).uniform(ref, 1, (50_000, F.shape[1])); cov = np.zeros(len(U), bool)
        for f in F:
            cov |= np.all(U <= f, 1)
        return float(cov.mean() * np.prod(1 - ref))

def run_mo(trk, space, seed, label, mo_log, tag, random_search=False, budget_points=None, budget_theta=None):
    """Archive {(theta_key, t): objective vector}. Only new theta values cost classifier trainings."""
    bounds, keyf, _ = SPACE[space]
    bounds = list(bounds) + [THR_RANGE]
    lo, hi = [b[0] for b in bounds], [b[1] for b in bounds]
    cache, archive = trk.prob.cache, {}

    def ev(vs):
        items = [keyf(v[:-1]) for v in vs]; trk.request(items)
        out = []
        for v, (k, _) in zip(vs, items):
            t = round(min(max(v[-1], THR_RANGE[0]), THR_RANGE[1]), 3)
            vec = mo_vec(cache[k], t); archive[(k, t)] = vec; out.append(vec)
            if THRESHOLD_REFINEMENT:                  # free local refinement: add the inner F1- and FITNESS-optimal
                e = cache[k]                          # thresholds of every new configuration (no extra training)
                if "thr_f1" not in e:
                    e["thr_f1"] = f1_opt_threshold(*e["oof"][:2], e["oof"][2])
                for tt in (e["thr_f1"], e["thr"]):
                    tt = round(float(tt), 3)
                    if (k, tt) not in archive:
                        archive[(k, tt)] = mo_vec(e, tt)
        return out

    def log(gen):
        F = np.array(list(archive.values())); fr = nondominated(F)
        mo_log.append({**tag, "method": label, "generation": gen, "configs": len(trk.keys), "points": len(archive),
                       "front_size": len(fr), "hypervolume": hypervolume(F),
                       **{f"best_{o}": float(F[:, i].max()) for i, o in enumerate(OBJECTIVES)}})
        print(f"      [{label}] gen {gen:02d} | configs {len(trk.keys):3d} | points {len(archive):4d} | front "
              f"{len(fr):3d} | HV {mo_log[-1]['hypervolume']:.5f} | best "
              + " ".join(f"{o}={mo_log[-1]['best_' + o]:.4f}" for o in OBJECTIVES))

    start = SPACE_START.get(space)                    # theory point (same for EA and random search)
    if random_search:
        rng = random.Random(seed)
        if start is not None:
            ev([start + [0.5]])
        while len(archive) < budget_points and len(trk.keys) < budget_theta:
            ev([[rng.uniform(l, h) for l, h in zip(lo, hi)] for _ in range(min(CHUNK, budget_points - len(archive), budget_theta - len(trk.keys)))])
        log(0)
        return archive

    random.seed(seed)
    tb = toolbox(lo, hi)
    pop = [creator.IndMO([random.uniform(l, h) for l, h in zip(lo, hi)]) for _ in range(MU_MO)]
    if start is not None:
        pop[0][:] = start + [0.5]
    for i, v in zip(pop, ev(pop)):
        i.fitness.values = tuple(v)
    pop = tools.selNSGA2(pop, MU_MO); log(0)
    best_hv, stall = mo_log[-1]["hypervolume"], 0
    for gen in range(1, NGEN_MO + 1):
        if len(trk.keys) >= MAX_THETA_MO or stall >= PATIENCE:
            break
        off = algorithms.varOr(pop, tb, LAMBDA_MO, CXPB, MUTPB)
        for o in off:
            o[:] = [min(max(x, l), h) for x, l, h in zip(o, lo, hi)]
        for i, v in zip(off, ev(off)):
            i.fitness.values = tuple(v)
        pop = tools.selNSGA2(pop + off, MU_MO)                         # (mu + lambda) NSGA-II survivor selection
        log(gen)
        hv = mo_log[-1]["hypervolume"]
        best_hv, stall = (hv, 0) if hv > best_hv + 1e-6 else (best_hv, stall + 1)
    return archive

def mo_select(archive, cache, acc_ref):
    """ONE (theta, t) from the inner archive, chosen by the rule fixed in SELECTION (never sees test data)."""
    keys = list(archive); F = np.array([archive[k] for k in keys])
    fr = nondominated(F); Ff = F[fr]
    ideal, nadir = Ff.max(0), Ff.min(0)
    N = (Ff - nadir) / np.clip(ideal - nadir, 1e-12, None)
    nsyn = np.array([cache[keys[i][0]]["nsyn"] for i in fr])
    if SELECTION == "knee":
        score = N.sum(1)
    elif SELECTION == "weighted":
        w = np.array([(WEIGHTS or {}).get(o, 1.0) for o in OBJECTIVES]); score = Ff @ (w / w.sum())
    elif SELECTION == "constraint":
        allm = [mo_full(cache[k[0]], k[1]) for k in keys]
        ok = [i for i, d in enumerate(allm) if d["Accuracy"] >= acc_ref - CONSTRAINT_DELTA]
        if ok:
            i = max(ok, key=lambda i: (allm[i]["F1"], -cache[keys[i][0]]["nsyn"]))
            return keys[i], len(fr), hypervolume(F)
        score = -np.max(1 - N, 1)
    else:
        score = -np.max(1 - N, 1)                                      # Chebyshev distance to the ideal point
    order = np.lexsort((nsyn, -score))
    return keys[fr[order[0]]], len(fr), hypervolume(F)

def boundary_hit(method, spec):
    if spec.get("gen") == "geo":
        if method not in MO_SPACE_OF:
            return ""
        hits = []
        if spec["r"] >= R_MAX * (1 - BOUND_TOL): hits.append("r_upper")
        if spec["sigma"] >= SIGMA_RANGE_G[1] * (1 - BOUND_TOL): hits.append("sigma_upper")
        w = BOUND_TOL * (GAMMA_RANGE[1] - GAMMA_RANGE[0])
        if spec["gamma"] >= GAMMA_RANGE[1] - w: hits.append("gamma_upper")
        if spec["gamma"] <= GAMMA_RANGE[0] + w: hits.append("gamma_lower")
        if spec["eta"] <= ETA_RANGE[0] + BOUND_TOL * (ETA_RANGE[1] - ETA_RANGE[0]): hits.append("eta_lower")
        return "+".join(hits)
    if "1D" in method:
        w = BOUND_TOL * (N_RANGE_1D[1] - N_RANGE_1D[0])
        return "upper" if spec["n"] >= N_RANGE_1D[1] - w else ("lower" if spec["n"] <= N_RANGE_1D[0] + w else "")
    if "3D" in method:
        return "upper" if spec["r"] >= R_MAX * (1 - BOUND_TOL) else ""
    return ""

# ===================================== 6. NESTED CV =====================================
rows, gen_log, mo_log, pareto_plot, search_info, typology = [], [], [], [], [], []
PARTIAL = f"{OUT_DIR}/outer_test_results_partial.csv"
import glob as _glob
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))
_base = os.path.basename(os.path.normpath(OUT_DIR))
_cands = ([p for g in os.environ["RESUME_CSV"].split(",") for p in sorted(_glob.glob(g.strip()))]
          if os.environ.get("RESUME_CSV") else
          sorted(_glob.glob(f"/kaggle/input/**/{_base}/outer_test_results*.csv", recursive=True)))
_cands += [PARTIAL] if os.path.exists(PARTIAL) else []
_src = ", ".join(_cands) if _cands else None
DONE = set()
if _src:
    _prev = (pd.concat([pd.read_csv(p) for p in _cands], ignore_index=True)
             .drop_duplicates(["classifier", "repeat", "fold", "method"], keep="last"))
    rows = [{k_: v_ for k_, v_ in r_.items() if not (isinstance(v_, float) and np.isnan(v_))}
            for r_ in _prev.to_dict("records")]
    DONE = set(zip(_prev.classifier, _prev["repeat"].astype(int), _prev.fold.astype(int), _prev.method))
    for nm_, lst_ in (("search_set_per_fold", search_info), ("minority_typology", typology)):
        _f = [os.path.join(os.path.dirname(p), f"{nm_}{sfx}.csv") for p in _cands for sfx in ("_partial", "")]
        _f = [p for p in dict.fromkeys(_f) if os.path.exists(p)]
        if _f:
            lst_ += (pd.concat([pd.read_csv(p) for p in _f], ignore_index=True)
                     .drop_duplicates(["classifier", "repeat", "fold"], keep="last").to_dict("records"))
    print(f"RESUME: {len(DONE)} finished (classifier, repeat, fold, method) rows loaded from {_src}")

def checkpoint():
    pd.DataFrame(rows).to_csv(PARTIAL, index=False)
    pd.DataFrame(search_info).to_csv(f"{OUT_DIR}/search_set_per_fold_partial.csv", index=False)
    pd.DataFrame(typology).to_csv(f"{OUT_DIR}/minority_typology_partial.csv", index=False)
t_all = time.time()
checkpoint() if rows else None
FOLD_TIMES, STOPPED = {}, False
for clf_name in CLASSIFIERS:
    if STOPPED:
        break
    for rep in range(N_REPEATS):
        if STOPPED:
            break
        outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + rep)
        for o, (tr, te) in enumerate(outer.split(X_all, y_all)):
            fseed = SEED + 1000 * rep + 10 * o
            if (REPEAT_IDS is not None and rep not in REPEAT_IDS) or (FOLD_IDS is not None and o not in FOLD_IDS):
                continue
            todo = [m for m in METHODS if (clf_name, rep, o, m) not in DONE]
            if not todo:
                print(f"   {clf_name} | repeat {rep} | fold {o + 1}: already done (resume) -> skipped"); continue
            _el = (time.time() - t_all) / 3600
            _est = 1.25 * FOLD_TIMES.get(clf_name, {"rf": 1.2}.get(clf_name, 0.3))       # hours, conservative
            if _el + _est > TIME_BUDGET_H:
                print(f"\nTIME BUDGET: {_el:.2f} h used, next fold needs ~{_est:.2f} h > {TIME_BUDGET_H} h -> "
                      f"stopping cleanly. Run again with the same OUT_DIR (add this output as input) to resume.")
                STOPPED = True; break
            _tf = time.time()
            tag = {"classifier": clf_name, "repeat": rep, "fold": o}
            print(f"\n======== {clf_name} | repeat {rep} | outer fold {o + 1}/{OUTER_FOLDS} ========")
            rng = np.random.default_rng(fseed)                          # search set (documented seed policy)
            mi, ma = tr[y_all[tr] == 1], tr[y_all[tr] == 0]
            ma_s = rng.choice(ma, min(len(ma), SEARCH_MAJ_MAX), replace=False)
            s_idx = np.sort(np.concatenate([mi, ma_s]))
            w_s = np.where(y_all[s_idx] == 0, len(ma) / len(ma_s), 1.0)
            inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=fseed)
            ifolds = [make_fold(s_idx[a], s_idx[b], w_s[a], w_s[b]) for a, b in inner.split(s_idx, y_all[s_idx])]
            search_info.append({**tag, "outer_train_rows": len(tr), "search_rows": len(s_idx), "frauds": len(mi),
                                "legit_kept": len(ma_s), "legit_total": len(ma), "legit_weight": len(ma) / len(ma_s),
                                "test_rows": len(te)})
            print(f"   search set: {len(s_idx)} rows ({len(mi)} frauds + {len(ma_s)} of {len(ma)} legit, weight "
                  f"{len(ma) / len(ma_s):.2f}) | final model: full outer-train ({len(tr)} rows), test {len(te)} rows")
            ofold = make_fold(tr, te)
            typology.append({**tag, **ofold["geo"]["typ"], "h_silverman": ofold["geo"]["h_S"]})
            print(f"   fraud typology (k={K_GEO}, Euclidean): " + ", ".join(f"{a} {b:.1%}" for a, b in ofold["geo"]["typ"].items())
                  + f" | Silverman h_S = {ofold['geo']['h_S']:.3f}")
            probs = {sp: Problem(clf_name, ifolds, fseed) for sp in list(SPACE) + ["base"]}
            selected, mo_arch, mo_sel = {}, {}, {}
            for method in todo:
                t0 = time.time()
                if method == "No oversampling":
                    trk = Tracker(probs["base"]); trk.request([(("none",), {"n": 0})])
                elif method == "Class weights":
                    trk = Tracker(probs["base"]); trk.request([(("cw",), {"n": 0, "cw": True})])
                elif method == "Grid-1D":
                    trk = Tracker(probs["1D"]); run_grid(trk)
                elif method in ("EPOS-1D", "EPOS-3D"):
                    sp = method[-2:]; trk = Tracker(probs[sp]); run_epos(trk, sp, fseed, method, gen_log, tag)
                elif method in ("Random-1D", "Random-3D"):
                    sp = method[-2:]; trk = Tracker(probs[sp])
                    run_random(trk, sp, fseed + 7, len(selected[f"EPOS-{sp}"][0].keys))
                elif method in SAMPLER_GRID:
                    trk = Tracker(probs["base"])
                    its = [(("sam", method) + tuple(sorted(cf.items())), {**cf, "gen": SAMPLER_GEN[method]})
                           for cf in SAMPLER_GRID[method]]
                    for c0 in range(0, len(its), CHUNK):
                        trk.request(its[c0:c0 + CHUNK])
                elif method == "KDE-Silverman":
                    trk = Tracker(probs["base"]); trk.request([(("kde",), dict(KDE_SPEC))])
                elif method in MO_SPACE_OF:
                    sp = MO_SPACE_OF[method]; trk = Tracker(probs[sp])
                    if method in RANDOM_REF:
                        ref = RANDOM_REF[method]
                        archive = run_mo(trk, sp, fseed + 7, method, mo_log, tag, random_search=True,
                                         budget_points=len(mo_arch[ref]), budget_theta=len(selected[ref][0].keys))
                    else:
                        archive = run_mo(trk, sp, fseed + 3, method, mo_log, tag)
                    if "No oversampling" in selected:
                        eb = probs["base"].cache[selected["No oversampling"][1]]
                        acc_ref = mo_full(eb, eb["thr"])["Accuracy"]
                    else:
                        acc_ref = 1 - y_all.mean()
                    (k_sel, t_sel), front_n, hv = mo_select(archive, probs[sp].cache, acc_ref)
                    mo_arch[method] = archive; mo_sel[method] = (k_sel, t_sel, front_n, hv, len(archive))
                else:
                    raise ValueError(f"unknown method {method}")
                search_wall = time.time() - t0
                if method in mo_sel:
                    k, thr_sel = mo_sel[method][0], mo_sel[method][1]
                else:
                    k = trk.best_key(); thr_sel = trk.prob.cache[k]["thr"]
                e = trk.prob.cache[k]; selected[method] = (trk, k)
                inner_sel = mo_full(e, thr_sel)
                # ---- final model on the FULL outer-train, evaluated once on the FULL untouched test fold ----
                t1 = time.perf_counter()
                p, nsyn, _, S_fin, acc_fin = fit_predict(clf_name, e["spec"], ofold, fseed + 99, n_jobs=FINAL_N_JOBS,
                                                         return_syn=True)
                syn_ov = overlap_rate(ofold, S_fin)
                final_s = time.perf_counter() - t1
                yte, vte, wte = ofold["yva"], ofold["vva"], ofold["wva"]
                m_t, m_05 = all_metrics(yte, p, thr_sel), all_metrics(yte, p, 0.5)
                yi, pi, wi, vi = e["oof"]
                costs = {}
                for cr in COST_RATIOS:                                    # thresholds chosen on inner OOF only
                    t_c = best_thr(lambda t: ratio_cost(yi, pi, t, cr, wi))
                    costs[f"Cost_r{cr}"] = ratio_cost(yte, p, t_c, cr, wte)
                t_s = best_thr(lambda t: -savings(yi, pi, t, vi, wi))
                costs["Savings"], costs["Savings_at_sel_thr"], costs["Savings_thr"] = \
                    savings(yte, p, t_s, vte, wte), savings(yte, p, thr_sel, vte, wte), t_s
                cst = trk.cost(); bh = boundary_hit(method, e["spec"])
                mo_info = ({"front_size": mo_sel[method][2], "inner_hypervolume": mo_sel[method][3],
                            "mo_points": mo_sel[method][4]} if method in mo_sel else {})
                rows.append({**tag, "method": method, **{f"sel_{a}": b for a, b in e["spec"].items()},
                             "n_syn_final": nsyn, "ratio_final": nsyn / max(ofold["ytr"].sum(), 1),
                             "syn_overlap": syn_ov, "syn_accept": acc_fin,
                             "threshold": thr_sel, "inner_fit": e["fit"], "boundary": bh,
                             **{f"inner_{a}": b for a, b in inner_sel.items()}, **mo_info,
                             **cst, "search_wall_s": search_wall, "final_fit_s": final_s,
                             **{m: m_t[m] for m in MKEYS + ["TN", "FP", "FN", "TP"]},
                             **{f"{m}@0.5": m_05[m] for m in ["F1", "MCC", "Precision", "Recall", "Accuracy"]}, **costs})
                print(f"   >> {method:16s} sel={e['spec']} n_syn={nsyn} thr={thr_sel:.4f} evals={cst['evals']} "
                      f"fits={cst['fits']} time={cst['fit_time_s']:.1f}s{' BOUND:' + bh if bh else ''}"
                      f"\n      TEST: {fmt(m_t)}\n      @0.5: F1={m_05['F1']:.4f} MCC={m_05['MCC']:.4f} | "
                      f"Savings={costs['Savings']:.4f} | CM [[TN {m_t['TN']} FP {m_t['FP']}] [FN {m_t['FN']} "
                      f"TP {m_t['TP']}]]")
                checkpoint()
            FOLD_TIMES[clf_name] = (time.time() - _tf) / 3600
            print(f"   fold time {60 * FOLD_TIMES[clf_name]:.1f} min | fit cache hits {FIT_CACHE_HITS[0]} / "
                  f"{sum(FIT_CACHE_HITS)} | elapsed {(time.time() - t_all) / 3600:.2f} h")
            if rep == 0 and o == 0 and MO_REFERENCE in mo_arch:
                arch = mo_arch[MO_REFERENCE]
                base_pts = {mth: mo_vec(tk.prob.cache[kk], tk.prob.cache[kk]["thr"])
                            for mth, (tk, kk) in selected.items() if mth not in mo_sel}
                pareto_plot.append({"F": np.array(list(arch.values())), "sel": arch[tuple(mo_sel[MO_REFERENCE][:2])],
                                    "base": base_pts})
total_min = (time.time() - t_all) / 60

# ===================================== 7. REPORT =====================================
res = pd.DataFrame(rows)
res.to_csv(f"{OUT_DIR}/outer_test_results.csv", index=False)
METHODS = [m for m in METHODS if m in set(res.method)] or METHODS
CLASSIFIERS = [c for c in CLASSIFIERS if c in set(res.classifier)] or CLASSIFIERS
if MO_REFERENCE not in METHODS:
    MO_REFERENCE = METHODS[0]
_full = res.groupby(["classifier", "method"]).size()
if _full.nunique() > 1:
    print("NOTE: methods have different numbers of finished folds (split / resumed run). The within-run tests below "
          "use only common folds where possible; use compare_results.py on the merged folders for the paper.")
_cnt = res.groupby(["classifier", "repeat", "fold"]).method.nunique()
_common = _cnt[_cnt == len(METHODS)].reset_index()[["classifier", "repeat", "fold"]]
res = res.merge(_common, on=["classifier", "repeat", "fold"])     # within-run report: folds with ALL methods only
if res.empty:
    print("No outer fold has all methods finished yet - partial results are saved; run again to resume."); sys.exit(0)
pd.DataFrame(gen_log).to_csv(f"{OUT_DIR}/epos_generation_log.csv", index=False)
pd.DataFrame(mo_log).to_csv(f"{OUT_DIR}/epos_mo_generation_log.csv", index=False)
pd.DataFrame(search_info).to_csv(f"{OUT_DIR}/search_set_per_fold.csv", index=False)
STAT_METRICS = MKEYS + ["Savings"] + [f"Cost_r{c}" for c in COST_RATIOS]
LOWER_BETTER = {"Brier", "LogLoss", "ECE"} | {f"Cost_r{c}" for c in COST_RATIOS}

for clf_name in CLASSIFIERS:
    R = res[res.classifier == clf_name]
    print(f"\n================ {DATASET_NAME} | {clf_name} | OUTER-FOLD TEST (mean +- std over {OUTER_FOLDS} folds x "
          f"{N_REPEATS} repeats) ================")
    for method in METHODS:
        d = R[R.method == method]
        print(f"\n   {method}")
        for m in STAT_METRICS + ["F1@0.5", "MCC@0.5"]:
            print(f"      {m:14s} {d[m].mean():.4f} +- {d[m].std(ddof=1):.4f}")
        print(f"      pooled CM: TN={d.TN.sum()} FP={d.FP.sum()} FN={d.FN.sum()} TP={d.TP.sum()} | mean synthetic "
              f"rows {d.n_syn_final.mean():.0f} (r={d.ratio_final.mean():.2f})")
summary = res.groupby(["classifier", "method"], sort=False)[STAT_METRICS + ["F1@0.5", "MCC@0.5"]].agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary.to_csv(f"{OUT_DIR}/summary_mean_std.csv")

# ---- computational cost ----
def hardware():
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    try:
        ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.1f} GB"
    except Exception:
        ram = "n/a"
    import sklearn, deap
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
         "scikit-learn": sklearn.__version__, "deap": getattr(deap, "__version__", "n/a")}
    for lib in ["lightgbm", "xgboost"]:
        try:
            v[lib] = __import__(lib).__version__
        except Exception:
            pass
    return {"os": platform.platform(), "cpu": cpu, "logical_cores": os.cpu_count(), "ram": ram, "n_jobs": N_JOBS, **v}

hw = hardware(); json.dump(hw, open(f"{OUT_DIR}/hardware.json", "w"), indent=2)
rt = res.groupby(["classifier", "method"], sort=False).agg(
    configs_per_fold=("evals", "mean"), fits_per_fold=("fits", "mean"), fit_time_s_per_fold=("fit_time_s", "mean"),
    search_wall_s_per_fold=("search_wall_s", "mean"), final_fit_s=("final_fit_s", "mean"),
    total_search_wall_s=("search_wall_s", "sum")).reset_index()
rt.insert(0, "dataset", DATASET_NAME)
rt["s_per_fit"] = rt.fit_time_s_per_fold / rt.fits_per_fold.clip(lower=1)
rt["search_set"] = f"all frauds + {SEARCH_MAJ_MAX} random legit rows (inverse-sampling weights)"
rt["final_model"] = "full outer-train; evaluated on full outer test fold"
rt.to_csv(f"{OUT_DIR}/runtime_table.csv", index=False)
print("\n================ Computational cost per outer fold ================")
print(rt.drop(columns=["search_set", "final_model"]).round(3).to_string(index=False))
print("   hardware:", json.dumps(hw), f"| total wall time {total_min:.1f} min")

# ---- search-space analysis ----
sa = res[res.method.str.contains("1D|3D")].groupby(["classifier", "method"], sort=False).agg(
    configs=("evals", "mean"), r_or_n=("ratio_final", "mean"),
    upper_bound_hits=("boundary", lambda s_: f"{(s_ == 'upper').sum()}/{len(s_)}"),
    threshold=("threshold", "mean"), test_F1=("F1", "mean"), test_Accuracy=("Accuracy", "mean"),
    test_PR_AUC=("PR_AUC", "mean"), test_MCC=("MCC", "mean"))
sa.to_csv(f"{OUT_DIR}/search_space_analysis.csv")
print("\n================ Search-space analysis (selected solutions) ================")
print(sa.round(4).to_string())

# ---- statistics ----
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return adj

MO_METHODS = [m for m in METHODS if m.startswith(("EPOS-MO-", "Random-MO-"))]
BASELINES = [m for m in METHODS if m not in MO_METHODS]
dom_rows = []
for clf_name in CLASSIFIERS:
    R = res[res.classifier == clf_name]
    V = {m: R[R.method == m].sort_values(["repeat", "fold"])[OBJECTIVES].values for m in METHODS}
    for a_ in MO_METHODS:
        any_dom = np.zeros(len(V[a_]), bool)
        for b_ in [m for m in METHODS if m != a_]:
            A, B = V[a_], V[b_]
            b_dom_a = np.all(B >= A, 1) & np.any(B > A, 1); a_dom_b = np.all(A >= B, 1) & np.any(A > B, 1)
            if b_ in BASELINES:
                any_dom |= b_dom_a
            ma_, mb_ = A.mean(0), B.mean(0)
            dom_rows.append({"classifier": clf_name, "A": a_, "B": b_, "folds": len(A),
                             "A_dominates_B": int(a_dom_b.sum()), "B_dominates_A": int(b_dom_a.sum()),
                             "incomparable_or_equal": int(len(A) - a_dom_b.sum() - b_dom_a.sum()),
                             "A_dominates_B_on_means": bool(np.all(ma_ >= mb_) and np.any(ma_ > mb_)),
                             "B_dominates_A_on_means": bool(np.all(mb_ >= ma_) and np.any(mb_ > ma_))})
        dom_rows.append({"classifier": clf_name, "A": a_, "B": "ANY BASELINE", "folds": len(V[a_]),
                         "B_dominates_A": int(any_dom.sum())})
dom = pd.DataFrame(dom_rows); dom.to_csv(f"{OUT_DIR}/dominance_analysis.csv", index=False)
print(f"\n================ (i) Pareto dominance on OUTER test folds, objectives {OBJECTIVES} ================")
print(dom.to_string(index=False))

pairs = [(MO_REFERENCE, m) for m in METHODS if m != MO_REFERENCE] + \
        [p_ for p_ in EXTRA_PAIRS if p_[0] in METHODS and p_[1] in METHODS]
st, verdicts = [], []
for clf_name in CLASSIFIERS:
    R = res[res.classifier == clf_name]
    for a_, b_ in pairs:
        block = []
        for metric in STAT_METRICS:
            xa = R[R.method == a_].sort_values(["repeat", "fold"])[metric].values
            xb = R[R.method == b_].sort_values(["repeat", "fold"])[metric].values
            d = (-1.0 if metric in LOWER_BETTER else 1.0) * (xa - xb)      # > 0  <=>  A better
            p = 1.0 if np.allclose(d, 0) else wilcoxon(xa, xb, zero_method="zsplit").pvalue
            block.append({"classifier": clf_name, "A": a_, "B": b_, "metric": metric,
                          "direction": "lower" if metric in LOWER_BETTER else "higher", "mean_A": xa.mean(),
                          "mean_B": xb.mean(), "A_better_by": d.mean(), "wins_A": int((d > 0).sum()),
                          "ties": int((d == 0).sum()), "losses_A": int((d < 0).sum()), "p": p})
        for r_, q in zip(block, holm([r_["p"] for r_ in block])):
            r_["p_holm"] = q
            r_["verdict"] = ("A better" if q < ALPHA and r_["A_better_by"] > 0 else
                             "A worse" if q < ALPHA and r_["A_better_by"] < 0 else "no sig. difference")
        st += block
        nb = sum(r_["verdict"] == "A better" for r_ in block); nw = sum(r_["verdict"] == "A worse" for r_ in block)
        verdicts.append({"classifier": clf_name, "A": a_, "B": b_, "metrics": len(block), "significantly_better": nb,
                         "no_sig_difference": len(block) - nb - nw, "significantly_worse": nw,
                         "better_or_equal": len(block) - nw, "better_mean": sum(r_["A_better_by"] > 0 for r_ in block)})
st, verdicts = pd.DataFrame(st), pd.DataFrame(verdicts)
st.to_csv(f"{OUT_DIR}/wilcoxon_holm_all_metrics.csv", index=False)
verdicts.to_csv(f"{OUT_DIR}/better_or_equal_summary.csv", index=False)
print("\n================ (ii) Wilcoxon per metric (paired outer folds), Holm across metrics ================")
print(st.round(4).to_string(index=False))
print(f"\n---- is A better-or-equal to B on most metrics? ({len(STAT_METRICS)} metrics, alpha={ALPHA}) ----")
print(verdicts.to_string(index=False))
if OUTER_FOLDS * N_REPEATS < 6:
    print("   WARNING: fewer than 6 paired outer folds -> Wilcoxon cannot reach p < 0.05.")

if MO_METHODS:
    mos = res[res.method.isin(MO_METHODS)].groupby(["classifier", "method"], sort=False).agg(
        front_size=("front_size", "mean"), inner_hypervolume=("inner_hypervolume", "mean"),
        configs=("evals", "mean"), points=("mo_points", "mean"), threshold=("threshold", "mean"),
        **{nm: (col, "mean") for nm, col in [("r", "sel_r"), ("sigma", "sel_sigma"), ("c", "sel_c"),
           ("gamma", "sel_gamma"), ("eta", "sel_eta"), ("overlap", "syn_overlap"), ("accept", "syn_accept")]
           if col in res.columns},
        **{f"inner_{o}": (f"inner_{o}", "mean") for o in OBJECTIVES}, **{f"test_{o}": (o, "mean") for o in OBJECTIVES})
    mos.to_csv(f"{OUT_DIR}/epos_mo_summary.csv")
    print(f"\n================ EPOS-MO summary (selection: {SELECTION}) ================")
    print(mos.round(4).to_string())

DOWN, UP = "$\\downarrow$", "$\\uparrow$"
la = res.groupby(["classifier", "method"], sort=False)[STAT_METRICS].agg(
    lambda s_: f"{s_.mean():.4f} $\\pm$ {s_.std(ddof=1):.4f}")
la.columns = [c + " " + (DOWN if c in LOWER_BETTER else UP) for c in la.columns]
open(f"{OUT_DIR}/table_all_metrics.tex", "w").write(
    "% ALL metrics, outer-fold TEST performance (mean +- std over outer folds x repeats); never used for selection\n"
    + la.T.to_latex(escape=False))

# ===================================== 8. PLOTS =====================================
if pareto_plot:
    P = pareto_plot[0]; F = P["F"]; fr = nondominated(F)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(F[:, 0], F[:, 1], c=F[:, 2] if F.shape[1] > 2 else "grey", s=8, alpha=0.4, cmap="viridis")
    ax.scatter(F[fr, 0], F[fr, 1], facecolors="none", edgecolors="k", s=30, label="inner Pareto front")
    ax.scatter(P["sel"][0], P["sel"][1], marker="*", s=300, c="red", label=f"selected ({SELECTION})")
    for nm, v in P["base"].items():
        ax.scatter(v[0], v[1], marker="s", s=40, label=nm)
    if F.shape[1] > 2:
        plt.colorbar(sc, ax=ax, label=f"inner {OBJECTIVES[2]}")
    ax.set(xlabel=f"inner {OBJECTIVES[0]}", ylabel=f"inner {OBJECTIVES[1]}", title=f"{MO_REFERENCE}: inner archive (fold 1)")
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/pareto_front.png", dpi=150); plt.close()

R = res[res.classifier == CLASSIFIERS[0]]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, m in zip(axes, ["F1", "MCC", "PR_AUC"]):
    ax.boxplot([R[R.method == mm][m] for mm in METHODS]); ax.set_xticks(range(1, len(METHODS) + 1), METHODS, rotation=25)
    ax.set(title=f"outer-fold test {m}")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/boxplots.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
for m in METHODS:
    d = R[R.method == m]; ax.plot(COST_RATIOS, [d[f"Cost_r{c}"].mean() for c in COST_RATIOS], marker="o", label=m)
ax.set(xscale="log", xlabel="C_FN / C_FP", ylabel="normalised expected cost (test)",
       title="cost curves (thresholds from inner OOF)")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/cost_curves.png", dpi=150); plt.close()

if mo_log:
    ml = pd.DataFrame(mo_log); ml = ml[ml.method == MO_REFERENCE]
    fig, ax = plt.subplots(figsize=(6, 4))
    for (rp, fo), d in ml.groupby(["repeat", "fold"]):
        ax.plot(d.generation, d.hypervolume, marker="o", ms=3, label=f"r{rp} f{fo + 1}")
    ax.set(xlabel="generation", ylabel="inner hypervolume", title=f"{MO_REFERENCE} convergence")
    ax.legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/mo_convergence.png", dpi=150); plt.close()

# ===================================== 9. GEOMETRY OUTPUTS =====================================
typ = pd.DataFrame(typology); typ.to_csv(f"{OUT_DIR}/minority_typology.csv", index=False)
print("\n================ fraud typology (outer-train folds, Euclidean k-NN) ================")
print(typ[["safe", "borderline", "rare", "outlier", "h_silverman"]].agg(["mean", "std"]).round(4).to_string())
ov = res.groupby(["classifier", "method"], sort=False)[["syn_overlap", "syn_accept", "ratio_final"]].mean()
ov.to_csv(f"{OUT_DIR}/synthetic_geometry.csv")
print("\n================ synthetic rows: overlap with legit (lower = safer) ================")
print(ov.round(4).to_string())
gm = [m for m in METHODS if MO_SPACE_OF.get(m, "").startswith("G")]
if gm:
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (col, rng_, lab) in zip(axes, [("sel_sigma", SIGMA_RANGE_G, "sigma (x Silverman bandwidth)"),
                                          ("sel_c", COV_RANGE_3D, "c (covariance mix)"),
                                          ("sel_gamma", GAMMA_RANGE, "gamma (seed focus)"),
                                          ("sel_eta", ETA_RANGE, "eta (safety filter)")]):
        for m in gm:
            v = res.loc[res.method == m, col].dropna()
            if len(v):
                ax.hist(v, bins=10, range=rng_, alpha=0.5, label=m)
        ax.set(xlabel=lab, ylabel="outer folds")
    axes[0].axvline(1.0, color="k", ls="--", lw=1); axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/geometry_genes.png", dpi=150); plt.close()

print(f"   sampler configurations that generated no rows: {len(SAMPLER_FAILS)}"
      + (f" e.g. {SAMPLER_FAILS[0]}" if SAMPLER_FAILS else ""))
print(f"\nSaved to ./{OUT_DIR}/  (total time {total_min:.1f} min)")
