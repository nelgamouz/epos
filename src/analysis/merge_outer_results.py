# =====================================================================================================
# merge_outer_results.py - merge the many outer_test_results.csv files of ONE dataset into TWO files:
#     <prefix>_baselines_outer_test_results.csv   (No oversampling, Class weights, SMOTE, ADASYN, Borderline-SMOTE, SMOTE-ENN)
#     <prefix>_EPOS_outer_test_results.csv        (EPOS, all classifiers)
# Rules: files are read in the order of the lists below; if the same (classifier, method, repeat, fold) appears in
#        several files the LATER file wins (so the re-runs *_lgbm_stable, epos_*_rf, epos_*_lgbm replace old rows).
#        Partial checkpoint files (outer_test_results_partial.csv) are used only if a folder has no final file.
# Usage (Kaggle: add the result folders as input first):
#     !python merge_outer_results.py churn          (or creditcard / paysim / all)
#     or paste the code into a notebook cell and set DATASET below
#     !python merge_outer_results.py creditcard --root /kaggle/input --out /kaggle/working
# A coverage table (number of outer folds per classifier x method) is printed -> check that nothing is missing.
# =====================================================================================================
import argparse, glob, os, sys
import pandas as pd

DATASET = "all"        # used when no argument is given (e.g. code pasted into a notebook cell): churn/creditcard/paysim/all
BASELINES = ["No oversampling", "Class weights", "SMOTE", "ADASYN", "Borderline-SMOTE", "SMOTE-ENN"]
EXPECTED_FOLDS = {"churn": 15, "creditcard": 10, "paysim": 10}
# folder-name patterns (relative to --root), in priority order: later = newer = wins
SOURCES = {
    "churn": {
        "prefix": "Churn",
        "baselines": ["baselines_churn_results"],
        "epos": ["epos_churn_paired_results"],
    },
    "creditcard": {
        "prefix": "CreditCard",
        "baselines": ["baselines_cc_results", "baseline_cc_*_results", "baselines_cc_lgbm_stable_results"],
        "epos": ["epos_creditcard_results", "epos_creditcard_rf_results", "epos_creditcard_lgbm_results"],
    },
    "paysim": {
        "prefix": "PaySim",
        "baselines": ["baselines_paysim_results", "baseline_paysim_*_results", "baselines_paysim_lgbm_stable_results"],
        "epos": ["epos_paysim_results", "epos_paysim_lgbm_results", "epos_paysim_lgbm_stable_results"],
    },
}
KEY = ["classifier", "method", "repeat", "fold"]
CLF_ORDER = {"lr": 0, "rf": 1, "lgbm": 2, "xgb": 3}

def find_files(root, folder_pat):
    """outer_test_results.csv of every folder matching folder_pat (partial file only if no final file)"""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "**", folder_pat), recursive=True)):
        if not os.path.isdir(d):
            continue
        final, part = os.path.join(d, "outer_test_results.csv"), os.path.join(d, "outer_test_results_partial.csv")
        if os.path.exists(final):
            out.append(final)
        elif os.path.exists(part):
            out.append(part)
    return out

def merge(paths, keep_methods=None, rename=None):
    parts = []
    for p in paths:
        d = pd.read_csv(p)
        if not set(KEY) <= set(d.columns):
            print(f"   SKIP {p} (columns {set(KEY) - set(d.columns)} missing)"); continue
        if rename:
            d["method"] = d["method"].replace(rename)
        if keep_methods:
            d = d[d["method"].isin(keep_methods)]
        d["source_file"] = p
        parts.append(d)
        print(f"   read {len(d):4d} rows  {p}")
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts, ignore_index=True).drop_duplicates(KEY, keep="last")
    m["_c"] = m["classifier"].map(CLF_ORDER).fillna(9)
    m["_m"] = m["method"].map({b: i for i, b in enumerate(BASELINES + ["EPOS"])}).fillna(99)
    m = m.sort_values(["_c", "_m", "repeat", "fold"]).drop(columns=["_c", "_m"])
    first = KEY + [c for c in m.columns if c not in KEY]
    return m[first].reset_index(drop=True)

def coverage(m, expected, label):
    if m.empty:
        print(f"   {label}: NO ROWS"); return
    t = m.pivot_table(index="method", columns="classifier", values="fold", aggfunc="count").fillna(0).astype(int)
    t = t[[c for c in CLF_ORDER if c in t.columns]]
    print(f"\n   {label}: outer folds per method x classifier (expected {expected})")
    print("   " + t.to_string().replace("\n", "\n   "))
    bad = [(i, c) for i in t.index for c in t.columns if t.loc[i, c] != expected]
    if bad:
        print(f"   WARNING: incomplete / missing: {bad}")

def run(ds, root, out):
    cfg = SOURCES[ds]; exp = EXPECTED_FOLDS[ds]
    print(f"\n===================== {cfg['prefix']} =====================")
    for kind in ("baselines", "epos"):
        paths = [p for pat in cfg[kind] for p in find_files(root, pat)]
        print(f"\n-- {kind}: {len(paths)} file(s)")
        if kind == "baselines":
            m = merge(paths, keep_methods=BASELINES)
            name = f"{cfg['prefix']}_baselines_outer_test_results.csv"
        else:
            m = merge(paths, rename={"EPOS (ours)": "EPOS"}, keep_methods=["EPOS"])
            name = f"{cfg['prefix']}_EPOS_outer_test_results.csv"
        coverage(m, exp, kind)
        if not m.empty:
            os.makedirs(out, exist_ok=True)
            m.to_csv(os.path.join(out, name), index=False)
            print(f"   -> saved {os.path.join(out, name)}  ({len(m)} rows)")

if __name__ == "__main__":
    sys.argv = [a for i, a in enumerate(sys.argv) if not (a == "-f" or (i > 0 and sys.argv[i - 1] == "-f"))]
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=DATASET, choices=list(SOURCES) + ["all"])
    ap.add_argument("--root", default="/kaggle/input", help="folder that contains the result folders")
    ap.add_argument("--out", default="merged_results")
    a, _unknown = ap.parse_known_args()        # ignores Jupyter/Colab's own '-f kernel.json' argument
    for ds in (SOURCES if a.dataset == "all" else [a.dataset]):
        run(ds, a.root, a.out)
