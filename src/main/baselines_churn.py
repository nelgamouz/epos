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

EPOS-G : geometry-aware EPOS for Telco Customer Churn
=====================================================
New generator (EPOS-G).  All geometry is computed on the TRAINING part of each fold only.
  Metric   : HEOM (heterogeneous Euclidean-overlap metric, Wilson & Martinez 1997) on the preprocessed row
             d(x, z)^2 = sum_{j cont} ((x_j - z_j) / 4)^2  +  sum_{c categorical} 1[x_c != z_c]
             (continuous features are standardised, so 4 sd ~ the range; one-hot blocks are scaled by 1/sqrt(2)).
             HEOM is a plain Euclidean distance after the linear map x -> W x, so exact k-NN search can be used.
  (1) theory-based noise scale (kernel density view): noise oversampling = sampling from a Gaussian KDE of the
      minority class.  Bandwidth  H_i = diag( sigma * h_S * a_i * s_j+ )^2  with
          h_S = (4 / ((d + 2) n_+))^(1/(d+4))         Silverman / Scott normal-reference rule (d = #continuous)
          a_i = d_k(x_i) / median_l d_k(x_l)          variable-kernel factor (Breiman, Meisel & Purcell 1977);
                                                      d_k = distance to the k-th minority neighbour
      -> sigma = 1 is the AMISE-optimal bandwidth for normal data; the EA only corrects it (sigma in [0, 3]).
  (2) seed weighting by local safety: m_i = #majority rows among the k HEOM-neighbours of minority row i,
          P(seed = i)  proportional to  exp(gamma * m_i / k) * 1[m_i < k]
      gamma < 0 -> safe regions, gamma = 0 -> uniform, gamma > 0 -> class border (ADASYN-like);
      rows with m_i = k ("outliers", Napierala & Stefanowski 2016) are never used as seeds.
  (3) nearest-neighbour safety filter: a synthetic row x~ is accepted only if
          d(x~, nearest minority row) <= eta * d(x~, nearest majority row)
      eta = 1 -> x~ lies in the minority cell of the 1-NN (Voronoi) partition; larger eta = more permissive.
  Categorical features: re-drawn from P(x_c | y = 1) with probability p_cat (as in EPOS-3D).
  Genes of EPOS-MO-G:  theta = (r, sigma, p_cat, gamma, eta)  + decision threshold t   (NSGA-II, 3 objectives)
  Ablations (same search, one component removed): -noAdapt (a_i = 1), -noSeedW (gamma = 0), -noFilter (no filter).
  KDE-Silverman : no search at all: 1:1 balance, sigma = 1 (closed-form bandwidth), p_cat = 0.
  Extra outputs : minority_typology.csv (safe / borderline / rare / outlier shares),
                  syn_overlap = share of synthetic rows whose nearest training neighbour is a majority row,
                  syn_accept  = acceptance rate of the filter, geometry_genes.png.
---------------------------------------------------------------------------------------------------------
(Text below = description of the inherited EPOS / EPOS-MO framework.)
Generators (both "noise-augmented oversampling", built from the TRAINING part of each fold only):
  original EPOS (1-D and 2-D spaces):
    x_syn = x_b + eps,   b ~ U{minority rows},   eps ~ N(0, sigma^2 I) on continuous (standardised) features;
    binary / one-hot features are copied unchanged.
  EPOS-3D generator (mixed-type noise, theta = (r, sigma, p_cat)):
    continuous j :  x_syn_j = x_b_j + eps_j,   eps_j ~ N(0, (sigma * s_j+)^2),  s_j+ = std of feature j among the
                    training minority  (per-feature, class-conditional noise scale)
    categorical c:  with prob. 1 - p_cat keep x_b_c,  with prob. p_cat draw x_c from P(x_c | y = 1)
                    (empirical distribution of that feature among training minority rows; one-hot blocks and binary
                    columns are replaced as a whole, so every synthetic row stays a valid category)

This script answers three reviewer remarks:
  (R1) "EPOS searches only 50-300 samples; why an evolutionary algorithm for a 1-D space?"
       -> Grid-1D   : exhaustive search of n in [50, 300] (the true optimum of the 1-D space)
          EPOS-1D   : the original (mu+lambda) EA in the same space, sigma fixed
          Random-1D : random search with the SAME number of evaluations as EPOS-1D
          EPOS-2D   : EA in the extended, dataset-scaled space  r = n_syn / n_min in [0, r_max], sigma in [0, 1]
          Random-2D : random search with the same budget as EPOS-2D
          EPOS-3D   : EA in r x sigma x p_cat with the mixed-type generator above (proposed variant)
          Random-3D : random search with the same budget as EPOS-3D
  (MO) EPOS-MO-1D/2D/3D : multi-objective EPOS (NSGA-II, (mu+lambda)) that maximises OBJECTIVES = (F1, Accuracy,
          PR-AUC) JOINTLY. The decision threshold t is a gene, so F1 and Accuracy always describe the SAME model:
              max_{theta, t}  ( F1(theta, t), Acc(theta, t), PR-AUC(theta) )      on pooled inner-CV predictions
          One configuration is taken from the inner Pareto front with a rule fixed in advance (SELECTION):
              chebyshev : argmin_{x in front} max_k (ideal_k - f_k(x)) / (ideal_k - nadir_k)
              knee      : argmax_{x in front} sum_k (f_k(x) - nadir_k) / (ideal_k - nadir_k)
              weighted  : argmax sum_k w_k f_k(x)
              constraint: argmax F1 s.t. Acc >= Acc(No oversampling) - CONSTRAINT_DELTA
          Random-MO-3D : random (theta, t) search with the same budget (same #configurations, same #points).
       Claims tested on the OUTER test folds:
          (i)  Pareto dominance: in how many outer folds is the EPOS-MO model dominated by any baseline in
               (F1, Accuracy, PR-AUC)?  (dominance_analysis.csv)
          (ii) better-or-equal on most metrics: paired two-sided Wilcoxon per metric over all outer folds x repeats,
               Holm correction across the metrics of each comparison (wilcoxon_holm_all_metrics.csv)
       All 1-D methods share one cache per outer fold (identical fitness values), so the comparison is exact:
       we report evaluations needed, regret vs the grid optimum, anytime curves, the 1-D landscape and
       how often the selected n hits the bounds of [50, 300].
  (R2) "Computational cost is not reported"
       -> per method / outer fold: unique evaluations, classifier fits, measured fit time, wall time,
          plus hardware + library versions; saved as runtime_table.csv (dataset x classifier x method).
  (R3) "F1 alone is insufficient"
       -> F1, PR-AUC, ROC-AUC, MCC, balanced accuracy, precision, recall, specificity, G-mean, accuracy,
          Brier, log-loss, ECE, confusion matrices; metrics at the tuned threshold AND at 0.5;
          cost analysis: normalised expected cost for C_FN/C_FP in COST_RATIOS and value-based savings
          (customer value = MonthlyCharges x CLV_MONTHS; offer cost OFFER_COST; offer keeps RETENTION_SUCCESS of churners).
Protocol: nested CV (outer 5-fold x N_REPEATS for evaluation, inner 3-fold for selection). The outer test fold
is never used by the preprocessing, the generator, the search or any threshold.
Run:  CHURN_CSV=Telco-Customer-Churn.csv python3 epos_churn_only.py      (or let kagglehub download it)
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix,
                             precision_recall_curve, log_loss)
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")
SEED = 42
OUT_DIR = os.environ.get("OUT_DIR", "baselines_churn_results")
os.makedirs(OUT_DIR, exist_ok=True)

CHURN_SOURCE = (("csv", os.environ["CHURN_CSV"]) if "CHURN_CSV" in os.environ
                else ("kaggle", "nourddineelgamouz/churn", "Telco-Customer-Churn.csv"))
DATASET_NAME = "Telco Churn"

CLASSIFIERS = os.environ.get("CLASSIFIERS", "lr,rf,lgbm,xgb").split(",")   # e.g. CLASSIFIERS=lgbm,xgb (split long runs)
LGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "max_bin": 63, "subsample": 0.8,
               "subsample_freq": 1, "colsample_bytree": 0.8}
XGB_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8}
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2}
LR_PARAMS = {"max_iter": 2000, "C": 1.0}

OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 5, 3    # 15 paired outer folds -> Wilcoxon can reach p < 0.05
TUNE_THRESHOLD = True                            # threshold from pooled inner OOF scores (never the test fold)
FITNESS = "MCC"                                  # EA fitness: "MCC" (stable) | "PR_AUC" (threshold-free) | "F1" (paper)

# --- search spaces ---
N_RANGE_1D = (50, 300)          # the original EPOS space (absolute number of synthetic samples)
SIGMA_1D = 0.05                 # fixed noise std in the 1-D space (standardised units), as in the original EPOS
GRID_STEP = 1                   # Grid-1D step (1 -> all 251 values = exact 1-D optimum; 5 -> 51 values, ~5x faster,
                                #   but then EPOS-1D may land between grid points and show a small negative regret)
R_MAX = None                    # 2-D ratio upper bound; None -> rho - 1 (= full 1:1 balance), capped at 5
SIGMA_RANGE_2D = (0.0, 1.0)
SIGMA_RANGE_3D = (0.0, 1.0)     # in units of the per-feature minority std
PCAT_RANGE_3D = (0.0, 0.3)      # probability of re-drawing each categorical feature
BOUND_TOL = 0.05                # "boundary hit" = selected value within 5 % of the range from a bound

# --- EA (paper settings) ---
MU, LAMBDA, NGEN, PATIENCE = 20, 40, 20, 3
CXPB, MUTPB = 0.5, 0.5
MAX_EVALS_1D = 251              # the 1-D space has at most 251 distinct values
MAX_EVALS_2D = 200
MAX_EVALS_3D = 200
CHUNK = 64                      # candidates evaluated per parallel batch
N_JOBS = -1

METHODS = ["No oversampling", "Class weights", "SMOTE", "ADASYN", "Borderline-SMOTE", "SMOTE-ENN"]
# also available: "Grid-1D", "EPOS-1D", "Random-1D", "EPOS-2D", "Random-2D", "EPOS-3D", "Random-3D",
#                 "EPOS-MO-1D", "EPOS-MO-2D", "Random-MO-3D"

# --- EPOS-G (geometry-aware generator) ---
K_GEO = 5                       # neighbours for safety degree, local scale and typology
SIGMA_RANGE_G = (0.0, 3.0)      # multiplier on the Silverman bandwidth (1 = closed-form rule)
GAMMA_RANGE = (-3.0, 3.0)       # seed focus: < 0 safe, 0 uniform, > 0 border
ETA_RANGE = (0.5, 3.0)          # NN safety filter ratio
ADAPT_CLIP = (0.25, 4.0)        # bounds of the variable-kernel factor a_i
FILTER_ROUNDS = 6               # max. resampling rounds to replace rejected synthetic rows
# (a "Random-MO-xD" method needs "EPOS-MO-xD" earlier in the list; it uses the same budget)

# --- multi-objective EPOS (EPOS-MO) ---
OBJECTIVES = ["F1", "Accuracy", "PR_AUC"]   # maximised jointly; any of F1, Accuracy, MCC, Balanced_Acc, Precision,
                                            # Recall, Specificity, G_mean, PR_AUC, ROC_AUC (all "higher is better")
SELECTION = "chebyshev"                     # "chebyshev" | "knee" | "weighted" | "constraint"  (fix BEFORE running)
WEIGHTS = None                              # "weighted": {metric: weight}; None -> equal weights on OBJECTIVES
CONSTRAINT_DELTA = 0.01                     # "constraint": max F1 s.t. inner Acc >= Acc(No oversampling) - delta
THR_RANGE = (0.05, 0.95)                    # decision threshold gene
MU_MO, LAMBDA_MO, NGEN_MO = 40, 40, 20
MAX_THETA_MO = 200                          # generator configurations (= classifier trainings x inner folds) per run
MO_REFERENCE = "Class weights"   # reference of the within-run tests (EPOS-G comparison: compare_results.py)                  # main method of the statistical comparison
EXTRA_PAIRS = [("EPOS-1D", "Grid-1D"), ("EPOS-2D", "Random-2D"), ("EPOS-3D", "Random-3D"),
               ("EPOS-MO-3D", "Random-MO-3D"), ("EPOS-MO-G-noAdapt", "EPOS-MO-3D")]
ALPHA = 0.05

# --- cost analysis (churn) ---
COST_RATIOS = [1, 2, 5, 10]     # C_FN / C_FP
CLV_MONTHS = 12                 # value of a customer = MonthlyCharges * CLV_MONTHS (lost if he churns)
OFFER_COST = 50.0               # every flagged customer gets a retention offer that costs OFFER_COST
RETENTION_SUCCESS = 0.5         # share of flagged churners that the offer actually keeps

random.seed(SEED); np.random.seed(SEED)

# ===================================== 1. INPUT LAYER =====================================
def load(src):
    if src[0] == "csv":
        return pd.read_csv(src[1])
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, src[1], src[2])

def prepare_telco(df):
    """Row-wise cleaning only (nothing fitted -> no leakage): drop customerID, TotalCharges text -> numeric."""
    df = df.copy(); df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in ["customerID", "CustomerID"] if c in df.columns])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
                                           errors="coerce").fillna(0.0)
    y = df["Churn"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(np.int8).values
    X = df.drop(columns=["Churn"])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).str.strip()
    value = (X["MonthlyCharges"].values * CLV_MONTHS if "MonthlyCharges" in X.columns
             else np.full(len(y), 100.0 * CLV_MONTHS))
    return X, y, value.astype(np.float64)

raw = load(CHURN_SOURCE)
X_all, y_all, value_all = prepare_telco(raw)
dup = X_all.assign(_y=y_all).duplicated().values
if dup.any():                                   # exact duplicates could sit in train and test at once
    X_all, y_all, value_all = X_all[~dup].reset_index(drop=True), y_all[~dup], value_all[~dup]
cat = [c for c in X_all.columns if not pd.api.types.is_numeric_dtype(X_all[c])]
num = [c for c in X_all.columns if c not in cat]
binary = [c for c in num if X_all[c].nunique() <= 2]
cont = [c for c in num if c not in binary]
PREP = ColumnTransformer(
    [("cont", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cont),
     ("bin", SimpleImputer(strategy="most_frequent"), binary),
     ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)],
    sparse_threshold=0.0)
N_CONT = len(cont)
n_pos, n_neg = int(y_all.sum()), int((y_all == 0).sum())
RHO = n_neg / n_pos
R_MAX_EFF = float(min(5.0, RHO - 1.0)) if R_MAX is None else float(R_MAX)
nmin_outer_train = n_pos * (OUTER_FOLDS - 1) / OUTER_FOLDS
nmin_inner_train = nmin_outer_train * (INNER_FOLDS - 1) / INNER_FOLDS
print(f"{DATASET_NAME}: n={len(y_all)} (exact duplicates removed: {int(dup.sum())}), churners={n_pos} "
      f"({100 * n_pos / len(y_all):.1f}%), imbalance ratio rho={RHO:.2f}:1, continuous features={cont}")
print(f"   minority size: outer-train ~{nmin_outer_train:.0f}, inner-train ~{nmin_inner_train:.0f}")
print(f"   original space n in {N_RANGE_1D} = ratio {N_RANGE_1D[0] / nmin_outer_train:.3f}-"
      f"{N_RANGE_1D[1] / nmin_outer_train:.3f} of the outer-train minority (1:1 balance needs "
      f"~{nmin_outer_train * (RHO - 1):.0f} synthetic rows)")
print(f"   extended space r in [0, {R_MAX_EFF:.2f}] (r_max = rho - 1 -> full balance), sigma in {SIGMA_RANGE_2D}")
print(f"   3-D space: r in [0, {R_MAX_EFF:.2f}], sigma (x minority std) in {SIGMA_RANGE_3D}, p_cat in {PCAT_RANGE_3D}; "
      f"categorical/binary features perturbed: {binary + cat}")

def build_fold(X, y, v, tr, va):
    f = _build_fold(X, y, v, tr, va); f["geo"] = geometry(f); return f

def heom_weights(fold):
    w = np.ones(fold["Xtr"].shape[1], np.float32)
    w[:N_CONT] = 0.25                                   # standardised continuous: (x - z) / 4 sd
    for g in fold["groups"]:
        if len(g) > 1:
            w[g] = 1.0 / np.sqrt(2.0)                   # one-hot block: mismatch -> distance contribution 1
    return w

def silverman(d, n):
    return float((4.0 / ((d + 2.0) * max(n, 2))) ** (1.0 / (d + 4.0)))

def geometry(fold):
    """HEOM k-NN geometry of the training minority (training part of the fold only)."""
    from sklearn.neighbors import NearestNeighbors
    X, y = fold["Xtr"], fold["ytr"]; w = heom_weights(fold); Z = X * w
    Zmin, Zmaj = Z[y == 1], Z[y == 0]; nmin = len(Zmin); k = min(K_GEO, max(nmin - 1, 1))
    nn_all = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(Z)
    idx = nn_all.kneighbors(Zmin, return_distance=False)[:, 1:]
    m = (y[idx] == 0).sum(1)                            # majority neighbours of each minority row
    C = X[y == 1][:, :N_CONT]
    if N_CONT and nmin > k:
        dk = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(C).kneighbors(C)[0][:, k]
        a = np.clip(dk / max(np.median(dk), 1e-6), *ADAPT_CLIP)
    else:
        a = np.ones(nmin)
    s = m / k
    typ = {"safe": float(np.mean(s <= 0.2)), "borderline": float(np.mean((s > 0.2) & (s <= 0.6))),
           "rare": float(np.mean((s > 0.6) & (s < 1))), "outlier": float(np.mean(s >= 1))}
    return dict(w=w, k=k, m=m, a=a.astype(np.float32), s_plus=C.std(0) if N_CONT else np.ones(0),
                h_S=silverman(N_CONT, nmin), typ=typ,
                nn_min=NearestNeighbors(n_neighbors=1, algorithm="brute").fit(Zmin),
                nn_maj=NearestNeighbors(n_neighbors=1, algorithm="brute").fit(Zmaj))

def overlap_rate(fold, S):
    """share of synthetic rows whose nearest HEOM neighbour in the training set is a majority row"""
    if len(S) == 0:
        return np.nan
    g = fold["geo"]; Z = S * g["w"]
    return float(np.mean(g["nn_maj"].kneighbors(Z)[0][:, 0] < g["nn_min"].kneighbors(Z)[0][:, 0]))

def geo_oversample(fold, n, spec, rng):
    """EPOS-G generator; returns (synthetic rows, acceptance rate of the safety filter)."""
    Xmin = fold["Xtr"][fold["ytr"] == 1]; g = fold["geo"]
    if n <= 0 or len(Xmin) == 0:
        return np.empty((0, fold["Xtr"].shape[1]), np.float32), 1.0
    valid = g["m"] < g["k"]
    if not valid.any():
        valid[:] = True
    lw = spec.get("gamma", 0.0) * g["m"] / g["k"]
    pw = np.exp(lw - lw.max()) * valid; pw /= pw.sum()
    eta = spec.get("eta", ETA_RANGE[1]); filt = eta < ETA_RANGE[1] - 1e-9
    base_scale = spec.get("sigma", 0.0) * g["h_S"] * g["s_plus"]
    out, need, made, kept = [], n, 0, 0
    for _ in range(FILTER_ROUNDS if filt else 1):
        acc_est = kept / made if made else 0.67           # observed acceptance rate so far
        m_ = min(int(np.ceil(need / max(acc_est, 0.05) * 1.2)) + 8, 20 * n) if filt else need
        b = rng.choice(len(Xmin), m_, p=pw)
        S = Xmin[b].copy()
        if N_CONT and spec.get("sigma", 0.0) > 0:
            sc = base_scale[None, :] * (g["a"][b][:, None] if spec.get("adapt", True) else 1.0)
            S[:, :N_CONT] += (rng.normal(0.0, 1.0, (m_, N_CONT)) * sc).astype(np.float32)
        pc = spec.get("p_cat", 0.0)
        if pc > 0:
            for grp in fold["groups"]:
                hit = np.flatnonzero(rng.random(m_) < pc)
                if len(hit):
                    S[np.ix_(hit, grp)] = Xmin[np.ix_(rng.integers(0, len(Xmin), len(hit)), grp)]
        made += m_
        if filt:
            Z = S * g["w"]
            ok = g["nn_min"].kneighbors(Z)[0][:, 0] <= eta * g["nn_maj"].kneighbors(Z)[0][:, 0]
            S = S[ok]
        kept += len(S)
        out.append(S[:need]); need -= len(out[-1])
        if need <= 0:
            break
    return np.vstack(out).astype(np.float32), kept / max(made, 1)

def _build_fold(X, y, v, tr, va):
    pp = clone(PREP).fit(X.iloc[tr])            # fitted on the training part only
    # column blocks of the transformed matrix: [continuous | binary | one-hot(cat_1) | one-hot(cat_2) | ...]
    groups, pos = [], N_CONT
    for _ in binary:
        groups.append(np.array([pos])); pos += 1
    if cat:
        for cats in pp.named_transformers_["cat"].named_steps["oh"].categories_:
            groups.append(np.arange(pos, pos + len(cats))); pos += len(cats)
    return dict(Xtr=pp.transform(X.iloc[tr]).astype(np.float32), ytr=y[tr], vtr=v[tr],
                Xva=pp.transform(X.iloc[va]).astype(np.float32), yva=y[va], vva=v[va], groups=groups)

R_GRID = sorted({0.25, 0.5, 1.0, 1.5, round(R_MAX_EFF, 2)})   # R_MAX_EFF = full 1:1 balance

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

def enn_keep(X, y, spec, n_jobs=1):
    """SMOTE-ENN cleaning step: Edited Nearest Neighbours (Wilson 1972) on original + synthetic rows, both classes
    (sampling_strategy="all", as in imblearn's SMOTEENN).  Returns the indices of the rows that are kept."""
    from imblearn.under_sampling import EditedNearestNeighbours
    try:
        enn = EditedNearestNeighbours(sampling_strategy="all", n_neighbors=int(spec["enn_k"]),
                                      kind_sel=spec["kind_sel"], n_jobs=n_jobs)
        enn.fit_resample(X, y)
        keep = np.sort(enn.sample_indices_)
        return keep if (y[keep] == 1).any() and (y[keep] == 0).any() else np.arange(len(y))
    except ValueError as e:
        SAMPLER_FAILS.append(("enn", str(e)[:120])); return np.arange(len(y))
SAMPLER_FAILS = []                     # (generator, error) of configurations that could not generate rows

def smote_family(X, y, n_new, spec, seed):
    """synthetic rows only (n_new requested) from SMOTE / ADASYN / Borderline-SMOTE fitted on (X, y)"""
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    n_min = int((y == 1).sum()); empty = np.empty((0, X.shape[1]), np.float32)
    if n_new <= 0 or n_min < 3:
        return empty
    k = int(min(spec["k"], n_min - 1)); strat = {1: n_min + int(n_new)}; gen = spec["gen"]
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
    """map interpolated rows back to valid categories: binary -> nearest observed value, one-hot -> argmax"""
    if len(S) == 0:
        return S
    for g in fold["groups"]:
        if len(g) == 1:
            col = fold["Xtr"][:, g[0]]; lo, hi = col.min(), col.max()
            S[:, g[0]] = np.where(S[:, g[0]] >= (lo + hi) / 2, hi, lo)
        else:
            B = np.zeros((len(S), len(g)), np.float32); B[np.arange(len(S)), S[:, g].argmax(1)] = 1.0; S[:, g] = B
    return S

# ===================================== 2. GENERATOR + CLASSIFIERS =====================================
def noise_oversample(fold, n, sigma, rng, per_feature=False, p_cat=0.0):
    """per_feature=False, p_cat=0 -> original EPOS generator.  per_feature=True -> eps_j ~ N(0, (sigma*s_j+)^2).
    p_cat > 0 -> each categorical / binary feature is re-drawn from the minority distribution with prob. p_cat."""
    Xmin = fold["Xtr"][fold["ytr"] == 1]
    if n <= 0 or len(Xmin) == 0:
        return np.empty((0, fold["Xtr"].shape[1]), np.float32)
    S = Xmin[rng.integers(0, len(Xmin), n)].copy()
    if sigma > 0 and N_CONT:
        scale = sigma * (Xmin[:, :N_CONT].std(0) if per_feature else np.ones(N_CONT))
        S[:, :N_CONT] += (rng.normal(0.0, 1.0, (n, N_CONT)) * scale).astype(np.float32)
    if p_cat > 0:
        for g in fold["groups"]:
            hit = np.flatnonzero(rng.random(n) < p_cat)
            if len(hit):                              # draw from P(x_c | y=1) = value of a random minority row
                S[np.ix_(hit, g)] = Xmin[np.ix_(rng.integers(0, len(Xmin), len(hit)), g)]
    return S

def n_synthetic(spec, fold):
    """1-D: absolute n (paper definition). 2-D: n = round(r * n_min of THIS training set)."""
    if spec.get("n") is not None:
        return int(spec["n"])
    return int(round(spec.get("r", 0.0) * int(fold["ytr"].sum())))

def make_clf(name, cw, y):
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_jobs=1, random_state=SEED, verbose=-1,
                              class_weight="balanced" if cw else None, **LGBM_PARAMS)
    if name == "xgb":
        from xgboost import XGBClassifier
        spw = (y == 0).sum() / max((y == 1).sum(), 1) if cw else 1.0
        return XGBClassifier(n_jobs=1, random_state=SEED, eval_metric="logloss", scale_pos_weight=spw,
                             **XGB_PARAMS)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=1, random_state=SEED, class_weight="balanced" if cw else None,
                                      **RF_PARAMS)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(class_weight="balanced" if cw else None, **LR_PARAMS)

def fit_predict(clf_name, spec, fold, seed, return_syn=False):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = n_synthetic(spec, fold); acc = 1.0
    if spec.get("gen") == "geo":
        S, acc = geo_oversample(fold, n, spec, rng)
    elif spec.get("gen") in ("smote", "adasyn", "bsmote", "smoteenn"):
        S = project_syn(fold, smote_family(fold["Xtr"], fold["ytr"], n, spec, int(rng.integers(2 ** 31 - 1))))
    else:
        S = noise_oversample(fold, n, spec.get("sigma", 0.0), rng, per_feature=spec.get("gen") == "mixed",
                             p_cat=spec.get("p_cat", 0.0))
    n = len(S)
    X = np.vstack([fold["Xtr"], S]); y = np.concatenate([fold["ytr"], np.ones(len(S), np.int8)])
    if spec.get("gen") == "smoteenn":
        keep = enn_keep(X, y, spec); X, y = X[keep], y[keep]
    p = make_clf(clf_name, spec.get("cw", False), y).fit(X, y).predict_proba(fold["Xva"])[:, 1]
    if return_syn:
        return p.astype(np.float64), n, time.perf_counter() - t0, S, acc
    return p.astype(np.float64), n, time.perf_counter() - t0

# ===================================== 3. METRICS + COST =====================================
MKEYS = ["F1", "PR_AUC", "ROC_AUC", "MCC", "Balanced_Acc", "Precision", "Recall", "Specificity",
         "G_mean", "Accuracy", "Brier", "LogLoss", "ECE"]

def f1_opt_threshold(y, p):
    P, R, T = precision_recall_curve(y, p)
    if len(T) == 0:
        return 0.5
    f1 = 2 * P[:-1] * R[:-1] / np.clip(P[:-1] + R[:-1], 1e-12, None)
    return float(T[np.argmax(f1)])

def all_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    idx = np.minimum((p * 10).astype(int), 9)
    ece = sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(10) if (idx == b).any())
    return {"F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "PR_AUC": average_precision_score(y, p),
            "ROC_AUC": roc_auc_score(y, p), "MCC": matthews_corrcoef(y, yhat), "Balanced_Acc": (rec + spec) / 2,
            "Precision": prec, "Recall": rec, "Specificity": spec, "G_mean": float(np.sqrt(rec * spec)),
            "Accuracy": (tp + tn) / len(y), "Brier": float(np.mean((p - y) ** 2)),
            "LogLoss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]), "ECE": float(ece),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def fmt(m, keys=MKEYS):
    return " ".join(f"{k}={m[k]:.4f}" for k in keys)

THR_GRID = np.linspace(0.005, 0.995, 199)

def ratio_cost(y, p, thr, c):
    """Normalised expected cost: (c*FN + FP) / (c*P)  (1.0 = cost of flagging nobody)."""
    yhat = p >= thr
    return float((c * np.sum((y == 1) & ~yhat) + np.sum((y == 0) & yhat)) / (c * max(y.sum(), 1)))

def savings(y, p, thr, v):
    """1 - cost / cost(no model). Missed churner: v_i lost. Flagged customer: OFFER_COST, and a flagged
    churner is still lost with probability 1 - RETENTION_SUCCESS."""
    yhat = p >= thr
    cost = (v[(y == 1) & ~yhat].sum() + (1 - RETENTION_SUCCESS) * v[(y == 1) & yhat].sum()
            + OFFER_COST * yhat.sum())
    return float(1.0 - cost / max(v[y == 1].sum(), 1e-12))

def best_thr(fun):
    vals = [fun(t) for t in THR_GRID]
    return float(THR_GRID[int(np.argmin(vals))])

# ===================================== 4. FITNESS CACHE (shared per outer fold & space) =====================================
class Problem:
    def __init__(self, clf_name, folds, seed):
        self.clf, self.folds, self.seed, self.cache = clf_name, folds, seed, {}

    def score(self, items):
        """items = [(key, spec)]; evaluates the new ones in parallel over (candidate x inner fold)."""
        new = [(k, s) for k, s in dict(items).items() if k not in self.cache]
        J = len(self.folds)
        for c in range(0, len(new), CHUNK):
            batch = new[c:c + CHUNK]
            t0 = time.perf_counter()
            out = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(fit_predict)(self.clf, s, self.folds[j], self.seed + j) for _, s in batch for j in range(J))
            wall = (time.perf_counter() - t0) / max(len(batch), 1)
            for i, (k, s) in enumerate(batch):
                o = out[i * J:(i + 1) * J]
                ys = [f["yva"] for f in self.folds]; ps = [x[0] for x in o]
                yc, pc = np.concatenate(ys), np.concatenate(ps)
                thr = f1_opt_threshold(yc, pc) if TUNE_THRESHOLD else 0.5
                fit_vals = [all_metrics(a, b, thr)[FITNESS] for a, b in zip(ys, ps)]
                self.cache[k] = {"spec": s, "fit": float(np.mean(fit_vals)), "fit_sd": float(np.std(fit_vals, ddof=1)),
                                 "thr": thr, "nsyn": float(np.mean([x[1] for x in o])),
                                 "fit_time": float(sum(x[2] for x in o)), "wall": wall, "fits": J,
                                 "m": all_metrics(yc, pc, thr), "oof": (yc, pc), "ys": ys, "ps": ps,
                                 "vval": np.concatenate([f["vva"] for f in self.folds])}

class Tracker:
    """One search method: the set of candidates IT requested (cost is counted per method, not per cache)."""
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
        return max(self.keys, key=lambda k: (c[k]["fit"], -c[k]["nsyn"]))   # tie -> fewer synthetic rows

    def best_fit(self):
        return self.prob.cache[self.best_key()]["fit"]

    def cost(self):
        c = self.prob.cache
        return {"evals": len(self.keys), "fits": sum(c[k]["fits"] for k in self.keys),
                "fit_time_s": sum(c[k]["fit_time"] for k in self.keys), "wall_s": sum(c[k]["wall"] for k in self.keys)}

# --- the two spaces ---
def key1(v):
    n = int(round(min(max(v[0], N_RANGE_1D[0]), N_RANGE_1D[1])))
    return ("n", n), {"n": n, "sigma": SIGMA_1D}

def key2(v):
    r = round(min(max(v[0], 0.0), R_MAX_EFF), 2)
    s = round(min(max(v[1], SIGMA_RANGE_2D[0]), SIGMA_RANGE_2D[1]), 3)
    if r <= 0:
        s = 0.0
    return ("r", r, s), {"r": r, "sigma": s}

def key3(v):
    r = round(min(max(v[0], 0.0), R_MAX_EFF), 2)
    s = round(min(max(v[1], SIGMA_RANGE_3D[0]), SIGMA_RANGE_3D[1]), 3)
    pc = round(min(max(v[2], PCAT_RANGE_3D[0]), PCAT_RANGE_3D[1]), 3)
    if r <= 0:
        s, pc = 0.0, 0.0
    return ("m", r, s, pc), {"r": r, "sigma": s, "p_cat": pc, "gen": "mixed"}

SPACE = {"1D": ([N_RANGE_1D], key1, MAX_EVALS_1D), "2D": ([(0.0, R_MAX_EFF), SIGMA_RANGE_2D], key2, MAX_EVALS_2D),
         "3D": ([(0.0, R_MAX_EFF), SIGMA_RANGE_3D, PCAT_RANGE_3D], key3, MAX_EVALS_3D)}

def make_keyG(adapt=True, use_gamma=True, use_filter=True):
    bounds = [(0.0, R_MAX_EFF), SIGMA_RANGE_G, PCAT_RANGE_3D] + ([GAMMA_RANGE] if use_gamma else []) + \
             ([ETA_RANGE] if use_filter else [])
    cl = lambda x, lo_hi: min(max(x, lo_hi[0]), lo_hi[1])
    def keyf(v):
        r, s, pc = round(cl(v[0], (0.0, R_MAX_EFF)), 2), round(cl(v[1], SIGMA_RANGE_G), 3), round(cl(v[2], PCAT_RANGE_3D), 3)
        i = 3; g = 0.0; e = ETA_RANGE[1]
        if use_gamma:
            g = round(cl(v[i], GAMMA_RANGE), 2); i += 1
        if use_filter:
            e = round(cl(v[i], ETA_RANGE), 2)
        if r <= 0:
            s, pc, g, e = 0.0, 0.0, 0.0, ETA_RANGE[1]
        return ("g", adapt, r, s, pc, g, e), {"r": r, "sigma": s, "p_cat": pc, "gamma": g, "eta": e,
                                              "adapt": adapt, "gen": "geo"}
    # theory-informed starting point: 1:1 balance, Silverman bandwidth, uniform seeds, no filter
    start = [R_MAX_EFF, 1.0, 0.0] + ([0.0] if use_gamma else []) + ([ETA_RANGE[1]] if use_filter else [])
    return bounds, keyf, start

for _nm, _kw in {"G": {}, "G_noAdapt": {"adapt": False}, "G_noSeedW": {"use_gamma": False},
                 "G_noFilter": {"use_filter": False}}.items():
    _b, _k, _s = make_keyG(**_kw); SPACE[_nm] = (_b, _k, MAX_EVALS_3D); SPACE_START = globals().setdefault("SPACE_START", {}); SPACE_START[_nm] = _s
MO_SPACE_OF = {"EPOS-MO-1D": "1D", "EPOS-MO-2D": "2D", "EPOS-MO-3D": "3D", "Random-MO-3D": "3D",
               "EPOS-MO-G": "G", "EPOS-MO-G-noAdapt": "G_noAdapt", "EPOS-MO-G-noSeedW": "G_noSeedW",
               "EPOS-MO-G-noFilter": "G_noFilter", "Random-MO-G": "G"}
RANDOM_REF = {"Random-MO-3D": "EPOS-MO-3D", "Random-MO-G": "EPOS-MO-G"}
KDE_SPEC = {"r": round(R_MAX_EFF, 2), "sigma": 1.0, "p_cat": 0.0, "gamma": 0.0, "eta": ETA_RANGE[1], "adapt": False, "gen": "geo"}

if not hasattr(creator, "FitE"):
    creator.create("FitE", base.Fitness, weights=(1.0, -1.0))     # max fitness, then fewer synthetic rows
if not hasattr(creator, "IndE"):
    creator.create("IndE", list, fitness=creator.FitE)

def run_epos(trk, space, seed, label, gen_log, tag):
    bounds, keyf, budget = SPACE[space]
    random.seed(seed)
    lo, hi = [b[0] for b in bounds], [b[1] for b in bounds]
    tb = base.Toolbox()
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=[0.15 * (h - l) for l, h in zip(lo, hi)], indpb=0.7)
    c = trk.prob.cache

    def ev(inds):
        items = [keyf(i) for i in inds]; trk.request(items)
        for i, (k, _) in zip(inds, items):
            i.fitness.values = (c[k]["fit"], c[k]["nsyn"])

    def report(gen, pop):
        k = trk.best_key(); e = c[k]
        gen_log.append({**tag, "method": label, "generation": gen, "evals": len(trk.keys),
                        "pop_mean_fit": float(np.mean([i.fitness.values[0] for i in pop])), **e["spec"],
                        "threshold": e["thr"], "n_syn": e["nsyn"], **{f"inner_{m}": e["m"][m] for m in MKEYS}})
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
        pop = tools.selBest(pop + off, MU)                             # (mu + lambda) survivor selection
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
    for c in range(0, len(ns), CHUNK):
        trk.request([key1([n]) for n in ns[c:c + CHUNK]])

# ===================================== 4b. EPOS-MO (NSGA-II, threshold as a gene) =====================================
THR_METRICS = ["F1", "Accuracy", "MCC", "Balanced_Acc", "Precision", "Recall", "Specificity", "G_mean"]
assert all(o in THR_METRICS + ["PR_AUC", "ROC_AUC"] for o in OBJECTIVES), "OBJECTIVES must be higher-is-better"

def thr_metrics(y, p, t):
    yhat = p >= t
    tp = float(np.sum(yhat & (y == 1))); fp = float(np.sum(yhat & (y == 0)))
    fn = float(np.sum(~yhat & (y == 1))); tn = float(np.sum(~yhat & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "Accuracy": (tp + tn) / len(y),
            "MCC": (tp * tn - fp * fn) / den if den else 0.0, "Balanced_Acc": (rec + spec) / 2,
            "Precision": prec, "Recall": rec, "Specificity": spec, "G_mean": float(np.sqrt(rec * spec))}

def mo_full(e, t):
    """All threshold metrics at threshold t + threshold-free PR/ROC-AUC, averaged over the inner folds."""
    memo = e.setdefault("mo_memo", {})
    t = round(float(t), 3)
    if t not in memo:
        if "tf" not in e:
            e["tf"] = {"PR_AUC": float(np.mean([average_precision_score(a, b) for a, b in zip(e["ys"], e["ps"])])),
                       "ROC_AUC": float(np.mean([roc_auc_score(a, b) for a, b in zip(e["ys"], e["ps"])]))}
        per = [thr_metrics(a, b, t) for a, b in zip(e["ys"], e["ps"])]
        memo[t] = {**{m: float(np.mean([d[m] for d in per])) for m in THR_METRICS}, **e["tf"]}
    return memo[t]

def mo_vec(e, t):
    d = mo_full(e, t)
    return np.array([d[o] for o in OBJECTIVES])

def nondominated(F):
    """Indices of the non-dominated rows of F (maximisation)."""
    F = np.asarray(F); keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F >= F[i], 1) & np.any(F > F[i], 1)
            if dom.any():
                keep[i] = False
    return np.flatnonzero(keep)

def hypervolume(F, ref=None):
    """Hypervolume of a maximisation front w.r.t. ref (default: origin; metrics clipped to [0, 1])."""
    F = np.clip(np.asarray(F, float), 0, 1); ref = np.zeros(F.shape[1]) if ref is None else ref
    F = F[nondominated(F)]
    try:
        from deap.tools._hypervolume import hv as _hv
        return float(_hv.hypervolume((-F).tolist(), (-ref).tolist()))
    except Exception:
        rng = np.random.default_rng(0); U = rng.uniform(ref, 1, (50_000, F.shape[1]))
        cov = np.zeros(len(U), bool)
        for f in F:
            cov |= np.all(U <= f, 1)
        return float(cov.mean() * np.prod(1 - ref))

if not hasattr(creator, "FitMO"):
    creator.create("FitMO", base.Fitness, weights=(1.0,) * len(OBJECTIVES))
if not hasattr(creator, "IndMO"):
    creator.create("IndMO", list, fitness=creator.FitMO)

def run_mo(trk, space, seed, label, mo_log, tag, random_search=False, budget_points=None, budget_theta=None):
    """Returns the archive {(theta_key, t): objective vector} of every (configuration, threshold) evaluated.
    Only new theta values cost classifier trainings; thresholds are evaluated on the cached inner predictions."""
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
        return out

    def log(gen):
        F = np.array(list(archive.values())); fr = nondominated(F)
        mo_log.append({**tag, "method": label, "generation": gen, "configs": len(trk.keys), "points": len(archive),
                       "front_size": len(fr), "hypervolume": hypervolume(F),
                       **{f"best_{o}": float(F[:, i].max()) for i, o in enumerate(OBJECTIVES)}})
        print(f"      [{label}] gen {gen:02d} | configs {len(trk.keys):3d} | points {len(archive):4d} | "
              f"front {len(fr):3d} | HV {mo_log[-1]['hypervolume']:.4f} | best "
              + " ".join(f"{o}={mo_log[-1]['best_' + o]:.4f}" for o in OBJECTIVES))

    start = SPACE_START.get(space) if "SPACE_START" in globals() else None
    if random_search:
        rng = random.Random(seed)
        if start is not None:
            ev([start + [0.5]])
        while len(archive) < budget_points and len(trk.keys) < budget_theta:
            n_new = min(CHUNK, budget_points - len(archive), budget_theta - len(trk.keys))
            ev([[rng.uniform(l, h) for l, h in zip(lo, hi)] for _ in range(n_new)])
        log(0)
        return archive

    random.seed(seed)
    tb = base.Toolbox()
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=[0.15 * (h - l) for l, h in zip(lo, hi)], indpb=0.7)
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
        hv_now = mo_log[-1]["hypervolume"]
        best_hv, stall = (hv_now, 0) if hv_now > best_hv + 1e-5 else (best_hv, stall + 1)
    return archive

def mo_select(archive, cache, acc_ref):
    """Pick ONE (theta, t) from the inner archive with the rule fixed in SELECTION (never looks at test data)."""
    keys = list(archive); F = np.array([archive[k] for k in keys])
    fr = nondominated(F); Ff = F[fr]
    ideal, nadir = Ff.max(0), Ff.min(0); span = np.clip(ideal - nadir, 1e-12, None)
    N = (Ff - nadir) / span
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
        score = -np.max(1 - N, 1)                                      # nothing feasible -> Chebyshev
    else:
        score = -np.max(1 - N, 1)                                      # Chebyshev distance to the ideal point
    order = np.lexsort((nsyn, -score))                                 # best score, then fewer synthetic rows
    return keys[fr[order[0]]], len(fr), hypervolume(F)

def boundary_hit(method, spec):
    if spec.get("gen") == "geo" and method in MO_SPACE_OF:
        hits = []
        if spec["r"] >= R_MAX_EFF * (1 - BOUND_TOL): hits.append("r_upper")
        if spec["sigma"] >= SIGMA_RANGE_G[1] * (1 - BOUND_TOL): hits.append("sigma_upper")
        if spec["p_cat"] >= PCAT_RANGE_3D[1] * (1 - BOUND_TOL): hits.append("p_cat_upper")
        w = BOUND_TOL * (GAMMA_RANGE[1] - GAMMA_RANGE[0])
        if spec["gamma"] >= GAMMA_RANGE[1] - w: hits.append("gamma_upper")
        if spec["gamma"] <= GAMMA_RANGE[0] + w: hits.append("gamma_lower")
        if spec["eta"] <= ETA_RANGE[0] + BOUND_TOL * (ETA_RANGE[1] - ETA_RANGE[0]): hits.append("eta_lower")
        return "+".join(hits)
    if method not in MO_SPACE_OF and not method.startswith(("EPOS-", "Random-", "Grid-")):
        return ""
    if method.endswith("1D"):
        w = BOUND_TOL * (N_RANGE_1D[1] - N_RANGE_1D[0])
        return "upper" if spec["n"] >= N_RANGE_1D[1] - w else ("lower" if spec["n"] <= N_RANGE_1D[0] + w else "")
    if method.endswith("2D") or method.endswith("3D"):
        hits = []
        if spec["r"] >= R_MAX_EFF * (1 - BOUND_TOL):
            hits.append("upper")
        if method.endswith("3D") and spec["p_cat"] >= PCAT_RANGE_3D[1] * (1 - BOUND_TOL):
            hits.append("p_cat_upper")
        return "+".join(hits)
    return ""

# ===================================== 5. NESTED CV =====================================
rows, gen_log, curves, landscapes, mo_log, pareto_plot, typology = [], [], [], [], [], [], []
t_all = time.time()
for clf_name in CLASSIFIERS:
    for rep in range(N_REPEATS):
        outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + rep)
        for o, (tr, te) in enumerate(outer.split(X_all, y_all)):
            fseed = SEED + 1000 * rep + 10 * o
            tag = {"classifier": clf_name, "repeat": rep, "fold": o}
            print(f"\n======== {clf_name} | repeat {rep} | outer fold {o + 1}/{OUTER_FOLDS} ========")
            ofold = build_fold(X_all, y_all, value_all, tr, te)
            typology.append({**tag, **ofold["geo"]["typ"], "h_silverman": ofold["geo"]["h_S"]})
            print(f"   minority typology (k={K_GEO}, HEOM): " + ", ".join(f"{a} {b:.1%}" for a, b in ofold["geo"]["typ"].items())
                  + f" | Silverman h_S = {ofold['geo']['h_S']:.3f}")
            Xo, yo, vo = X_all.iloc[tr].reset_index(drop=True), y_all[tr], value_all[tr]
            inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=fseed)
            ifolds = [build_fold(Xo, yo, vo, a, b) for a, b in inner.split(Xo, yo)]
            probs = {sp: Problem(clf_name, ifolds, fseed) for sp in list(SPACE) + ["base"]}
            selected, mo_archives, mo_sel = {}, {}, {}
            for method in METHODS:
                t0 = time.time()
                if method == "No oversampling":
                    trk = Tracker(probs["base"]); trk.request([(("none",), {"n": 0})])
                elif method == "Class weights":
                    trk = Tracker(probs["base"]); trk.request([(("cw",), {"n": 0, "cw": True})])
                elif method in SAMPLER_GRID:
                    trk = Tracker(probs["base"])
                    its = [(("sam", method) + tuple(sorted(cf.items())), {**cf, "gen": SAMPLER_GEN[method]})
                           for cf in SAMPLER_GRID[method]]
                    for c0 in range(0, len(its), CHUNK):
                        trk.request(its[c0:c0 + CHUNK])
                elif method == "KDE-Silverman":
                    trk = Tracker(probs["base"]); trk.request([(("kde",), dict(KDE_SPEC))])
                elif method == "Grid-1D":
                    trk = Tracker(probs["1D"]); run_grid(trk)
                elif method == "EPOS-1D":
                    trk = Tracker(probs["1D"]); run_epos(trk, "1D", fseed, method, gen_log, tag)
                elif method == "Random-1D":
                    trk = Tracker(probs["1D"]); run_random(trk, "1D", fseed + 7, len(selected["EPOS-1D"][0].keys))
                elif method == "EPOS-2D":
                    trk = Tracker(probs["2D"]); run_epos(trk, "2D", fseed, method, gen_log, tag)
                elif method == "Random-2D":
                    trk = Tracker(probs["2D"]); run_random(trk, "2D", fseed + 7, len(selected["EPOS-2D"][0].keys))
                elif method == "EPOS-3D":
                    trk = Tracker(probs["3D"]); run_epos(trk, "3D", fseed, method, gen_log, tag)
                elif method == "Random-3D":
                    trk = Tracker(probs["3D"]); run_random(trk, "3D", fseed + 7, len(selected["EPOS-3D"][0].keys))
                elif method.startswith(("EPOS-MO-", "Random-MO-")):
                    sp = MO_SPACE_OF[method]; trk = Tracker(probs[sp])
                    if method.startswith("EPOS-MO-"):
                        archive = run_mo(trk, sp, fseed + 3, method, mo_log, tag)
                    else:
                        ref_trk, ref_arch = selected[RANDOM_REF[method]][0], mo_archives[RANDOM_REF[method]]
                        archive = run_mo(trk, sp, fseed + 7, method, mo_log, tag, random_search=True,
                                         budget_points=len(ref_arch), budget_theta=len(ref_trk.keys))
                    if "No oversampling" in selected:
                        eb = probs["base"].cache[selected["No oversampling"][1]]
                        acc_ref = mo_full(eb, eb["thr"])["Accuracy"]
                    else:
                        acc_ref = 1 - y_all.mean()
                    (k_sel, t_sel), front_n, hv = mo_select(archive, probs[sp].cache, acc_ref)
                    mo_archives[method] = archive
                    mo_sel[method] = (k_sel, t_sel, front_n, hv, len(archive))
                search_wall = time.time() - t0
                if method in mo_sel:
                    k, thr_sel = mo_sel[method][0], mo_sel[method][1]
                else:
                    k = trk.best_key(); thr_sel = trk.prob.cache[k]["thr"]
                e = trk.prob.cache[k]
                selected[method] = (trk, k)
                inner_sel = mo_full(e, thr_sel)              # inner-CV metrics of the selected (theta, t)
                # ---- final model: FULL outer-train, evaluated once on the FULL untouched outer test fold ----
                t1 = time.perf_counter()
                p, nsyn, _, S_fin, acc_fin = fit_predict(clf_name, e["spec"], ofold, fseed + 99, return_syn=True)
                syn_ov = overlap_rate(ofold, S_fin)
                final_s = time.perf_counter() - t1
                yte, vte = ofold["yva"], ofold["vva"]
                m_t, m_05 = all_metrics(yte, p, thr_sel), all_metrics(yte, p, 0.5)
                yi, pi = e["oof"]
                costs = {}
                for cr in COST_RATIOS:
                    t_c = best_thr(lambda t: ratio_cost(yi, pi, t, cr))          # chosen on inner OOF only
                    costs[f"Cost_r{cr}"] = ratio_cost(yte, p, t_c, cr)
                    costs[f"Cost_r{cr}_at0.5"] = ratio_cost(yte, p, 0.5, cr)
                t_s = best_thr(lambda t: -savings(yi, pi, t, e["vval"]))
                costs["Savings"], costs["Savings_at0.5"], costs["Savings_thr"] = \
                    savings(yte, p, t_s, vte), savings(yte, p, 0.5, vte), t_s
                cst = trk.cost()
                bh = boundary_hit(method, e["spec"])
                gap = ((selected["Grid-1D"][0].best_fit() - e["fit"])
                       if method in ("Grid-1D", "EPOS-1D", "Random-1D") and "Grid-1D" in selected else np.nan)
                mo_info = ({"front_size": mo_sel[method][2], "inner_hypervolume": mo_sel[method][3],
                            "mo_points": mo_sel[method][4]} if method in mo_sel else {})
                rows.append({**tag, "method": method, **{f"sel_{a}": b for a, b in e["spec"].items()},
                             "n_syn_final": nsyn, "ratio_final": nsyn / max(ofold["ytr"].sum(), 1),
                             "syn_overlap": syn_ov, "syn_accept": acc_fin,
                             "threshold": thr_sel, "inner_fit": e["fit"], "regret_vs_grid": gap, "boundary": bh,
                             **{f"inner_{a}": b for a, b in inner_sel.items()}, **mo_info,
                             **cst, "search_wall_s": search_wall, "final_fit_s": final_s,
                             **{m: m_t[m] for m in MKEYS + ["TN", "FP", "FN", "TP"]},
                             **{f"{m}@0.5": m_05[m] for m in ["F1", "MCC", "Precision", "Recall", "Balanced_Acc"]},
                             **costs})
                curves.append({**tag, "method": method, "curve": trk.curve})
                print(f"   >> {method:16s} sel={e['spec']} n_syn={nsyn} thr={thr_sel:.3f} evals={cst['evals']} "
                      f"fits={cst['fits']} time={cst['fit_time_s']:.1f}s{' BOUND:' + bh if bh else ''}"
                      f"{'' if np.isnan(gap) else f' regret={gap:.4f}'}\n      TEST: {fmt(m_t)}"
                      f"\n      @0.5: F1={m_05['F1']:.4f} MCC={m_05['MCC']:.4f} Recall={m_05['Recall']:.4f} | "
                      f"Savings={costs['Savings']:.4f} (0.5: {costs['Savings_at0.5']:.4f}) | CM [[{m_t['TN']} "
                      f"{m_t['FP']}] [{m_t['FN']} {m_t['TP']}]]")
            if rep == 0 and o == 0 and MO_REFERENCE in mo_archives:          # data for the Pareto-front plot
                arch = mo_archives[MO_REFERENCE]; ks = list(arch)
                base_pts = {}
                for mth, (tk, kk) in selected.items():
                    if mth in mo_sel:
                        continue
                    ee = tk.prob.cache[kk]; base_pts[mth] = mo_vec(ee, ee["thr"])
                ks_sel = mo_sel[MO_REFERENCE][:2]
                pareto_plot.append({"clf": clf_name, "F": np.array([arch[k] for k in ks]),
                                    "sel": arch[tuple(ks_sel)], "base": base_pts})
            if rep == 0 and o == 0 and all(m in selected for m in ["Grid-1D", "EPOS-2D", "EPOS-3D"]):
                c1 = probs["1D"].cache
                landscapes.append({"clf": clf_name, "n": [c1[k]["spec"]["n"] for k in selected["Grid-1D"][0].keys],
                                   "fit": [c1[k]["fit"] for k in selected["Grid-1D"][0].keys],
                                   "sd": [c1[k]["fit_sd"] for k in selected["Grid-1D"][0].keys],
                                   "pts2": [(probs["2D"].cache[k]["spec"]["r"], probs["2D"].cache[k]["spec"]["sigma"],
                                             probs["2D"].cache[k]["fit"]) for k in selected["EPOS-2D"][0].keys],
                                   "pts3": [(probs["3D"].cache[k]["spec"]["r"], probs["3D"].cache[k]["spec"]["p_cat"],
                                             probs["3D"].cache[k]["fit"]) for k in selected["EPOS-3D"][0].keys]})
total_min = (time.time() - t_all) / 60

# ===================================== 6. REPORT =====================================
res = pd.DataFrame(rows)
res.to_csv(f"{OUT_DIR}/outer_test_results.csv", index=False)
pd.DataFrame(gen_log).to_csv(f"{OUT_DIR}/epos_generation_log.csv", index=False)

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
    return {"os": platform.platform(), "cpu": cpu, "logical_cores": os.cpu_count(), "ram": ram,
            "n_jobs": N_JOBS, "parallelism": "threads over (candidate x inner fold)", **v}

hw = hardware()
json.dump(hw, open(f"{OUT_DIR}/hardware.json", "w"), indent=2)

for clf_name in CLASSIFIERS:
    R = res[res.classifier == clf_name]
    print(f"\n================ {DATASET_NAME} | {clf_name} | OUTER-FOLD TEST performance "
          f"(mean +- std over {OUTER_FOLDS} folds x {N_REPEATS} repeats) ================")
    for method in METHODS:
        d = R[R.method == method]
        print(f"\n   {method}")
        for m in MKEYS + ["F1@0.5", "MCC@0.5", "Recall@0.5", "Savings", "Savings_at0.5"] + \
                [f"Cost_r{c}" for c in COST_RATIOS]:
            print(f"      {m:14s} {d[m].mean():.4f} +- {d[m].std(ddof=1):.4f}")
        print(f"      pooled CM: TN={d.TN.sum()} FP={d.FP.sum()} FN={d.FN.sum()} TP={d.TP.sum()}")

agg = {m: ["mean", "std"] for m in MKEYS + ["Savings"] + [f"Cost_r{c}" for c in COST_RATIOS] + ["F1@0.5", "MCC@0.5"]}
summary = res.groupby(["classifier", "method"], sort=False).agg(agg)
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary.to_csv(f"{OUT_DIR}/summary_mean_std.csv")

# ---- (R1) search-space analysis ----
print("\n================ (R1) Search-space analysis ================")
sa = res[res.method.isin(["Grid-1D", "EPOS-1D", "Random-1D", "EPOS-2D", "Random-2D", "EPOS-3D", "Random-3D"])].groupby(
    ["classifier", "method"], sort=False).agg(
    evals=("evals", "mean"), inner_fit=("inner_fit", "mean"), regret_vs_grid=("regret_vs_grid", "mean"),
    regret_zero_share=("regret_vs_grid", lambda s: float((s.abs() < 1e-9).mean()) if s.notna().any() else np.nan),
    upper_bound_hits=("boundary", lambda s: f"{s.str.contains('upper').sum()}/{len(s)}"),
    ratio_final_mean=("ratio_final", "mean"), test_F1=("F1", "mean"), test_MCC=("MCC", "mean"),
    test_PR_AUC=("PR_AUC", "mean"))
print(sa.round(4).to_string())
sa.to_csv(f"{OUT_DIR}/search_space_analysis.csv")

# ---- (R2) runtime table ----
rt = res.groupby(["classifier", "method"], sort=False).agg(
    evals_per_fold=("evals", "mean"), fits_per_fold=("fits", "mean"), fit_time_s_per_fold=("fit_time_s", "mean"),
    search_wall_s_per_fold=("search_wall_s", "mean"), final_fit_s=("final_fit_s", "mean"),
    total_search_wall_s=("search_wall_s", "sum")).reset_index()
rt.insert(0, "dataset", DATASET_NAME)
rt["s_per_fit"] = rt.fit_time_s_per_fold / rt.fits_per_fold.clip(lower=1)
rt["rows_used_in_search"] = f"all outer-train rows (~{len(y_all) * (OUTER_FOLDS - 1) / OUTER_FOLDS:.0f}, no subsampling)"
rt.to_csv(f"{OUT_DIR}/runtime_table.csv", index=False)
print("\n================ (R2) Computational cost per outer fold ================")
print(rt.drop(columns=["rows_used_in_search"]).round(3).to_string(index=False))
print("   hardware:", json.dumps(hw))
print(f"   note: shared cache -> evaluations of Grid/EPOS/Random-1D overlap; each method is charged for "
      f"every candidate it requested (measured fit time), as if run alone.  total wall time {total_min:.1f} min")

# ---- statistics (paired over outer folds) ----
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return adj

STAT_METRICS = MKEYS + ["Savings"] + [f"Cost_r{c}" for c in COST_RATIOS]
LOWER_BETTER = {"Brier", "LogLoss", "ECE"} | {f"Cost_r{c}" for c in COST_RATIOS}
MO_METHODS = [m for m in METHODS if m.startswith(("EPOS-MO-", "Random-MO-"))]
BASELINES = [m for m in METHODS if m not in MO_METHODS]

# (i) Pareto dominance on the OUTER test folds in the objective space
dom_rows = []
for clf_name in CLASSIFIERS:
    R = res[res.classifier == clf_name]
    V = {m: R[R.method == m].sort_values(["repeat", "fold"])[OBJECTIVES].values for m in METHODS}
    for a_ in MO_METHODS:
        any_dom = np.zeros(len(V[a_]), bool)
        for b_ in [m for m in METHODS if m != a_]:
            A, B = V[a_], V[b_]
            b_dom_a = np.all(B >= A, 1) & np.any(B > A, 1)
            a_dom_b = np.all(A >= B, 1) & np.any(A > B, 1)
            if b_ in BASELINES:
                any_dom |= b_dom_a
            ma, mb = A.mean(0), B.mean(0)
            dom_rows.append({"classifier": clf_name, "A": a_, "B": b_, "folds": len(A),
                             "A_dominates_B": int(a_dom_b.sum()), "B_dominates_A": int(b_dom_a.sum()),
                             "incomparable_or_equal": int(len(A) - a_dom_b.sum() - b_dom_a.sum()),
                             "mean_vector_A": np.round(ma, 4).tolist(), "mean_vector_B": np.round(mb, 4).tolist(),
                             "B_dominates_A_on_means": bool(np.all(mb >= ma) and np.any(mb > ma)),
                             "A_dominates_B_on_means": bool(np.all(ma >= mb) and np.any(ma > mb))})
        dom_rows.append({"classifier": clf_name, "A": a_, "B": "ANY BASELINE", "folds": len(V[a_]),
                         "B_dominates_A": int(any_dom.sum())})
dom = pd.DataFrame(dom_rows)
dom.to_csv(f"{OUT_DIR}/dominance_analysis.csv", index=False)
print(f"\n================ (i) Pareto dominance on OUTER test folds, objectives {OBJECTIVES} ================")
print(dom.drop(columns=["mean_vector_A", "mean_vector_B"], errors="ignore").to_string(index=False))

# (ii) paired Wilcoxon per metric, Holm across the metrics of each comparison
pairs = [(MO_REFERENCE, m) for m in METHODS if m != MO_REFERENCE] + \
        [p_ for p_ in EXTRA_PAIRS if p_[0] in METHODS and p_[1] in METHODS and p_[0] != MO_REFERENCE]
st, verdicts = [], []
for clf_name in CLASSIFIERS:
    R = res[res.classifier == clf_name]
    for a_, b_ in pairs:
        block = []
        for metric in STAT_METRICS:
            xa = R[R.method == a_].sort_values(["repeat", "fold"])[metric].values
            xb = R[R.method == b_].sort_values(["repeat", "fold"])[metric].values
            sign = -1.0 if metric in LOWER_BETTER else 1.0
            d = sign * (xa - xb)                                       # > 0  <=>  A is better
            p = 1.0 if np.allclose(d, 0) else wilcoxon(xa, xb, zero_method="zsplit").pvalue
            block.append({"classifier": clf_name, "A": a_, "B": b_, "metric": metric,
                          "direction": "lower" if metric in LOWER_BETTER else "higher",
                          "mean_A": xa.mean(), "mean_B": xb.mean(), "A_better_by": d.mean(),
                          "wins_A": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses_A": int((d < 0).sum()),
                          "p": p})
        for r_, q in zip(block, holm([r_["p"] for r_ in block])):
            r_["p_holm"] = q
            r_["verdict"] = ("A better" if q < ALPHA and r_["A_better_by"] > 0 else
                             "A worse" if q < ALPHA and r_["A_better_by"] < 0 else "no sig. difference")
        st += block
        nb = sum(r_["verdict"] == "A better" for r_ in block); nw = sum(r_["verdict"] == "A worse" for r_ in block)
        verdicts.append({"classifier": clf_name, "A": a_, "B": b_, "metrics": len(block),
                         "significantly_better": nb, "no_sig_difference": len(block) - nb - nw,
                         "significantly_worse": nw, "better_or_equal": len(block) - nw,
                         "better_mean": sum(r_["A_better_by"] > 0 for r_ in block)})
st = pd.DataFrame(st); verdicts = pd.DataFrame(verdicts)
st.to_csv(f"{OUT_DIR}/wilcoxon_holm_all_metrics.csv", index=False)
verdicts.to_csv(f"{OUT_DIR}/better_or_equal_summary.csv", index=False)
print("\n================ (ii) Wilcoxon signed-rank per metric (paired outer folds), Holm across metrics ================")
print(st.round(4).to_string(index=False))
print(f"\n---- summary: is A better-or-equal to B on most metrics? ({len(STAT_METRICS)} metrics, alpha={ALPHA}) ----")
print(verdicts.to_string(index=False))
if len(res) // max(len(METHODS) * len(CLASSIFIERS), 1) < 6:
    print("   WARNING: fewer than 6 paired outer folds -> Wilcoxon cannot reach p < 0.05; use N_REPEATS >= 2.")

# ---- EPOS-MO summary (inner front size, hypervolume) ----
if MO_METHODS:
    mos = res[res.method.isin(MO_METHODS)].groupby(["classifier", "method"], sort=False).agg(
        front_size=("front_size", "mean"), inner_hypervolume=("inner_hypervolume", "mean"),
        hv_sd=("inner_hypervolume", "std"), configs=("evals", "mean"), points=("mo_points", "mean"),
        threshold=("threshold", "mean"), ratio_final=("ratio_final", "mean"),
        **{c: (c, "mean") for c in ["sel_sigma", "sel_p_cat", "sel_gamma", "sel_eta", "syn_overlap", "syn_accept"]
           if c in res.columns},
        **{f"inner_{o}": (f"inner_{o}", "mean") for o in OBJECTIVES}, **{f"test_{o}": (o, "mean") for o in OBJECTIVES})
    mos.to_csv(f"{OUT_DIR}/epos_mo_summary.csv")
    pd.DataFrame(mo_log).to_csv(f"{OUT_DIR}/epos_mo_generation_log.csv", index=False)
    print(f"\n================ EPOS-MO summary (selection rule: {SELECTION}) ================")
    print(mos.round(4).to_string())

# ---- LaTeX table ----
lt = res.groupby(["classifier", "method"], sort=False)[["F1", "PR_AUC", "MCC", "Balanced_Acc", "Recall", "Precision",
                                        "Savings"]].agg(lambda s: f"{s.mean():.3f} $\\pm$ {s.std(ddof=1):.3f}")
open(f"{OUT_DIR}/table_outer_test.tex", "w").write(
    "% Outer-fold TEST performance (mean +- std over outer folds x repeats); never used for model selection\n"
    + lt.to_latex(escape=False))
la = res.groupby(["classifier", "method"], sort=False)[STAT_METRICS].agg(
    lambda s_: f"{s_.mean():.3f} $\\pm$ {s_.std(ddof=1):.3f}")
DOWN, UP = "$\\downarrow$", "$\\uparrow$"
la.columns = [c + " " + (DOWN if c in LOWER_BETTER else UP) for c in la.columns]
open(f"{OUT_DIR}/table_all_metrics.tex", "w").write(
    "% ALL metrics, outer-fold TEST (mean +- std); arrows = direction of improvement\n" + la.T.to_latex(escape=False))

# ===================================== 7. PLOTS =====================================
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, ms in zip(axes, [["Grid-1D", "EPOS-1D", "Random-1D"], ["EPOS-2D", "Random-2D"], ["EPOS-3D", "Random-3D"]]):
    for m in ms:
        cs = [c["curve"] for c in curves if c["method"] == m and c["classifier"] == CLASSIFIERS[0]]
        if not cs:
            continue
        L = max(c[-1][0] for c in cs); xs = np.arange(1, L + 1)
        ys = [np.interp(xs, [p[0] for p in c], [p[1] for p in c], left=np.nan) for c in cs]
        ax.plot(xs, np.nanmean(ys, 0), label=m)
    ax.set(xlabel="unique evaluations", ylabel=f"best inner {FITNESS} (mean over outer folds)",
           title="anytime: " + {"Grid-1D": "1-D space 50-300", "EPOS-2D": "2-D (r, sigma)",
                                "EPOS-3D": "3-D (r, sigma, p_cat)"}[ms[0]])
    ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/anytime_curves.png", dpi=150); plt.close()

if landscapes:
    L = landscapes[0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    n, f, s = np.array(L["n"]), np.array(L["fit"]), np.array(L["sd"])
    axes[0].plot(n, f, lw=1); axes[0].fill_between(n, f - s, f + s, alpha=0.2)
    axes[0].set(xlabel="n synthetic samples", ylabel=f"inner {FITNESS}", title="1-D landscape (Grid-1D, fold 1)")
    P2 = np.array(L["pts2"])
    sc = axes[1].scatter(P2[:, 0], P2[:, 1], c=P2[:, 2], cmap="viridis", s=18)
    axes[1].axvline(N_RANGE_1D[1] / nmin_inner_train, ls="--", c="r", label="n=300 (inner train)")
    axes[1].set(xlabel="r = n_syn / n_min", ylabel="sigma", title="EPOS-2D evaluated points (fold 1)")
    axes[1].legend(); plt.colorbar(sc, ax=axes[1], label=f"inner {FITNESS}")
    P3 = np.array(L["pts3"])
    sc3 = axes[2].scatter(P3[:, 0], P3[:, 1], c=P3[:, 2], cmap="viridis", s=18)
    axes[2].set(xlabel="r = n_syn / n_min", ylabel="p_cat", title="EPOS-3D evaluated points (fold 1)")
    plt.colorbar(sc3, ax=axes[2], label=f"inner {FITNESS}")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/landscape.png", dpi=150); plt.close()

if pareto_plot:
    P = pareto_plot[0]; F = P["F"]; fr = nondominated(F)
    i0, i1 = 0, 1
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(F[:, i0], F[:, i1], c=F[:, 2] if F.shape[1] > 2 else "grey", s=8, alpha=0.4, cmap="viridis")
    ax.scatter(F[fr, i0], F[fr, i1], facecolors="none", edgecolors="k", s=30, label="inner Pareto front")
    ax.scatter(P["sel"][i0], P["sel"][i1], marker="*", s=300, c="red", label=f"selected ({SELECTION})")
    for nm, v in P["base"].items():
        ax.scatter(v[i0], v[i1], marker="s", s=40, label=nm)
    if F.shape[1] > 2:
        plt.colorbar(sc, ax=ax, label=OBJECTIVES[2])
    ax.set(xlabel=f"inner {OBJECTIVES[i0]}", ylabel=f"inner {OBJECTIVES[i1]}", title=f"{MO_REFERENCE}: inner archive (fold 1)")
    ax.legend(fontsize=6, loc="best"); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/pareto_front.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(14, 4.5))
R = res[res.classifier == CLASSIFIERS[0]]
ax.boxplot([R[R.method == m]["MCC"] for m in METHODS]); ax.set_xticks(range(1, len(METHODS) + 1), METHODS)
ax.set(ylabel="outer-fold test MCC", title=f"{DATASET_NAME}: outer-fold MCC"); plt.xticks(rotation=25)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/mcc_boxplot.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
for m in METHODS:
    d = R[R.method == m]
    ax.plot(COST_RATIOS, [d[f"Cost_r{c}"].mean() for c in COST_RATIOS], marker="o", label=m)
ax.set(xlabel="C_FN / C_FP", ylabel="normalised expected cost (test)", title="cost curves (threshold from inner OOF)")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/cost_curves.png", dpi=150); plt.close()

print(f"   sampler configurations that generated no rows: {len(SAMPLER_FAILS)}"
      + (f" e.g. {SAMPLER_FAILS[0]}" if SAMPLER_FAILS else ""))
print(f"\nSaved to ./{OUT_DIR}/  (total time {total_min:.1f} min)")

# ===================================== 8. GEOMETRY OUTPUTS =====================================
typ = pd.DataFrame(typology); typ.to_csv(f"{OUT_DIR}/minority_typology.csv", index=False)
print("\n================ minority typology (outer-train folds, HEOM k-NN) ================")
print(typ[["safe", "borderline", "rare", "outlier", "h_silverman"]].agg(["mean", "std"]).round(4).to_string())
ov = res.groupby(["classifier", "method"], sort=False)[["syn_overlap", "syn_accept", "ratio_final"]].mean()
ov.to_csv(f"{OUT_DIR}/synthetic_geometry.csv")
print("\n================ synthetic rows: overlap with majority (lower = safer) ================")
print(ov.round(4).to_string())
gm = [m for m in METHODS if m in MO_SPACE_OF and MO_SPACE_OF[m].startswith("G")]
if gm:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (col, rng_, lab) in zip(axes, [("sel_sigma", SIGMA_RANGE_G, "sigma (x Silverman bandwidth)"),
                                          ("sel_gamma", GAMMA_RANGE, "gamma (seed focus)"),
                                          ("sel_eta", ETA_RANGE, "eta (safety filter)")]):
        for m in gm:
            v = res.loc[res.method == m, col].dropna()
            if len(v):
                ax.hist(v, bins=10, range=rng_, alpha=0.5, label=m)
        ax.set(xlabel=lab, ylabel="outer folds")
    axes[0].axvline(1.0, color="k", ls="--", lw=1); axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/geometry_genes.png", dpi=150); plt.close()
print(f"geometry outputs saved to ./{OUT_DIR}/")
