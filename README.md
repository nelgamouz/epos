# EPOS – Evolutionary Parameter Optimization for OverSampling

Code and per-fold results for the paper

> N. El Gamouz, Y. Hajoui, A. Hessane, E. A. Abdellaoui Alaoui, S. Agoujil,
> *Evolutionary Optimization of Noise-Augmented Oversampling Parameters for Financial Imbalanced Classification*, IEEE Access (under review).

EPOS tunes the number of synthetic samples `n` and the noise level `sigma` of a bootstrap + Gaussian-noise
oversampler with a (mu+lambda) evolution strategy (DEAP), together with the decision threshold, inside nested
cross-validation. It is compared with six tuned baselines (no oversampling, class weights, SMOTE, ADASYN,
Borderline-SMOTE, SMOTE-ENN) on four classifiers (LR, RF, LightGBM, XGBoost).

## Datasets (not included – download from Kaggle)

| Dataset | Kaggle | File | Licence |
|---|---|---|---|
| Telco Customer Churn | https://www.kaggle.com/datasets/nourddineelgamouz/churn (copy of IBM Telco Customer Churn) | `Telco-Customer-Churn.csv` | IBM sample data |
| Credit Card Fraud (ULB) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `creditcard.csv` | DbCL v1.0 |
| PaySim | https://www.kaggle.com/datasets/nourddineelgamouz/financial-fraud-detection (copy of PaySim) | `Financial_Fraud_Detection.csv` | CC BY-SA 4.0 |

The scripts look for the file in `/kaggle/input/**`, then download it with `kagglehub`.
You can also give the path: `CHURN_CSV=...`, `CREDITCARD_CSV=...`, `PAYSIM_CSV=...`.

## Environment

Kaggle CPU notebook (Intel Xeon 2.20 GHz, 4 cores, 32.9 GB RAM, no GPU), Python 3.10.12.
`pip install -r requirements.txt`

## Repository layout

```
src/main/       nested-CV experiments (main results tables)
src/revision/   ablations: random/grid search at equal budget, wide ratio space, F1-vs-ratio curve
src/analysis/   merging of per-run files, statistical tests, LaTeX tables
results/        per-fold outer-test results, ablation, runtime and statistics outputs
```

## How to reproduce (order of execution)

Every script writes `<OUT_DIR>/outer_test_results.csv` (one row per classifier x method x repeat x fold).
Long runs stop before `TIME_BUDGET_H` (default 11 h, Kaggle limit 12 h) and **resume** when the previous output
folder is added as Kaggle input. Classifiers can be split over sessions with `CLASSIFIERS=lr,xgb` etc.

**1. Baselines** (6 tuned baselines, all classifiers)

| Dataset | Script(s) | Output folder |
|---|---|---|
| Churn | `baselines_churn.py` | `baselines_churn_results` |
| Credit Card | `baselines_creditcard.py`, then `baseline_cc_SMOTE_ENN_fast.py` (SMOTE-ENN), then `baselines_cc_lgbm_stable.py` (LightGBM re-run) | `baselines_cc_results`, `baseline_cc_SMOTE_ENN_results`, `baselines_cc_lgbm_stable_results` |
| PaySim | `baselines_paysim.py`, then `baselines_paysim_lgbm_stable.py` (LightGBM re-run) | `baselines_paysim_results`, `baselines_paysim_lgbm_stable_results` |

**2. EPOS**

| Dataset | Script(s) | Output folder |
|---|---|---|
| Churn | `epos_churn_paired.py` | `epos_churn_paired_results` |
| Credit Card | `epos_creditcard_nested.py` (LR, XGB), `epos_creditcard_rf_fast.py` (RF, reduced budget: max 30 configurations), `epos_creditcard_lgbm.py` (LightGBM) | `epos_creditcard_results`, `epos_creditcard_rf_results`, `epos_creditcard_lgbm_results` |
| PaySim | `epos_paysim_nested.py` (LR, RF, XGB), `epos_paysim_lgbm.py` (LightGBM) | `epos_paysim_results`, `epos_paysim_lgbm_stable_results` |

The `*_lgbm*` scripts use a stabilised LightGBM (`min_child_weight=1.0`, `reg_lambda=1.0`) for baselines and EPOS alike;
their rows replace the earlier LightGBM rows.

**3. Merge and statistics**

```
python src/analysis/merge_outer_results.py all --root /kaggle/input --out .   # later folders win for the same fold
python src/analysis/fix_misaligned_csv.py merged_results/CreditCard_baselines_outer_test_results.csv
python src/analysis/stats_churn.py        # Friedman / Nemenyi / Wilcoxon-Holm / Nadeau-Bengio
python src/analysis/stats_creditcard.py
python src/analysis/stats_paysim.py
python src/analysis/make_paper_tables.py
```

**4. Ablations (revision)**

```
python src/revision/epos_revision_churn.py
python src/revision/epos_revision_creditcard.py     # RF: CLASSIFIERS=rf OUT_DIR=rev_cc_rf (reduced EA budget)
python src/revision/epos_revision_paysim.py
python src/revision/make_ablation_tables.py         # reads all_rows.csv, wilcoxon_search.csv, curve_summary.csv
```
Methods: `EPOS-orig`, `RS-orig`, `Grid-orig` (n in [50,300]), `EPOS-wide`, `RS-wide`, `Grid-wide`
(r = n/|minority train| in [0,5]; Churn: r up to full balance), and `Curve` (fixed r grid, sigma = 0.05).
Random and grid search get exactly the number of configurations EPOS evaluated in the same fold.

## Protocol summary

* Outer CV: stratified 5-fold x 2 repeats (Churn: 5 x 3) -> 10 (15) untouched test folds; identical folds for all methods.
* Inner CV: stratified 3-fold (Churn: 5-fold) on the outer-training part.
* Seeds: `SEED = 42`; fold seed `42 + 1000*repeat + 10*fold` (same for every method -> paired folds).
* Inner search on Credit Card / PaySim: all training frauds + 60,000 / 100,000 random legitimate rows
  (fold seed), legitimate rows re-weighted to the true class ratio. Final model: refit on the full outer-training part.
* EPOS: mu = 12, lambda = 24, max 10 generations, patience 3, cxpb 0.7, mutpb 0.3, blend crossover (alpha 0.3),
  polynomial bounded mutation (eta 15, indpb 0.5), max 100 distinct configurations, `n in [50,300]`, `sigma in [0.01,0.20]`;
  noise only on standardised continuous features, one-hot features copied.
* Threshold: chosen on the pooled inner out-of-fold predictions only (never on the test fold).
* Tuning objective as run: EPOS uses inner-CV **F1** (`FITNESS=F1`, threshold = F1-optimal, 145-value grid; Churn: exact PR curve).
  The baseline scripts use inner-CV **MCC** (`FITNESS = "MCC"`; Credit Card/PaySim threshold = MCC-optimal among 101 quantiles of
  the minority scores, Churn threshold = F1-optimal). The EPOS scripts accept `FITNESS=MCC` for an identical objective.
* Preprocessing (fitted on training parts only):
  Churn – drop `customerID`, `TotalCharges` to numeric, remove exact duplicate rows, median imputation + standardisation
  (continuous), most-frequent imputation (binary/categorical), one-hot encoding;
  Credit Card – remove exact duplicate rows, `log1p(Amount)`, standardisation of all 30 features;
  PaySim – features `amount`, `oldbalanceOrg-newbalanceOrig`, `oldbalanceDest-newbalanceDest` (signed log1p) + one-hot `type`
  (`nameOrig`, `nameDest`, `isFlaggedFraud`, `step` not used); rule-based outlier filter applied to **training rows only**
  (`oldbalanceOrg==0 & amount>0`, `newbalanceOrig==oldbalanceOrg`, `oldbalanceDest==0 & newbalanceDest>1e7`); test folds complete.

## Results files

| File | Content |
|---|---|
| `results/per_fold/*_outer_test_results.csv` | one row per classifier x method x repeat x fold: selected parameters, threshold, inner fitness, all outer-test metrics, TP/FP/FN/TN, timings |
| `results/ablation/all_rows.csv` | per-fold ablation results (search strategy, wide space, curve) |
| `results/ablation/wilcoxon_search.csv`, `curve_summary.csv`, `all_summary.csv` | ablation tests and summaries |
| `results/runtime/runtime_summary_all.csv` | wall-clock time and number of configurations per method |
| `results/stats/<dataset>/` | Friedman, Nemenyi ranks, Wilcoxon-Holm and Nadeau-Bengio tests |

Telco Churn per-fold files: `results/per_fold/Churn_baselines_outer_test_results.csv` and
`Churn_EPOS_outer_test_results.csv` (output of `merge_outer_results.py churn`).

## Citation

```
@article{elgamouz2025epos,
  author  = {El Gamouz, Nourddine and Hajoui, Younes and Hessane, Abdelaaziz and Abdellaoui Alaoui, El Arbi and Agoujil, Said},
  title   = {Evolutionary Optimization of Noise-Augmented Oversampling Parameters for Financial Imbalanced Classification},
  journal = {IEEE Access},
  year    = {2025},
  note    = {under review}
}
```

