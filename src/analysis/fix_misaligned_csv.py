import csv, numpy as np, pandas as pd, sys
# usage: python fix_misaligned_csv.py <merged_baselines.csv> [out.csv]   (notebook: set SRC below)
SRC = "merged_results/CreditCard_baselines_outer_test_results.csv"
args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
src = args[0] if args else SRC
dst = args[1] if len(args) > 1 else src.replace(".csv", "_FIXED.csv")
rows = list(csv.reader(open(src)))
h = rows[0]; KEY = h[:4]
C = h[h.index("n_syn_final"):h.index("Savings_thr") + 1]          # common block, same for every method
out = []
for r in rows[1:]:
    if len(r) == len(h):                         # row written with this header (e.g. LightGBM re-run) -> already aligned
        d = dict(zip(h, r)); d["sel_params"] = "|".join(f"{k}={v}" for k, v in d.items() if k.startswith("sel_") and v)
        out.append({k: v for k, v in d.items() if not k.startswith("sel_") or k == "sel_params"}); continue
    sel = r[4:len(r) - len(C)]                   # shorter row from an older file: method params first, common block last
    d = dict(zip(KEY, r[:4])); d.update(zip(C, r[len(r) - len(C):]))
    d["sel_params"] = "|".join(sel)
    out.append(d)
df = pd.DataFrame(out)
for c in df.columns:
    if c not in ("classifier", "method", "sel_params", "syn_overlap", "boundary", "ratio_final"):
        try: df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError): pass
tp, fp, fn = df.TP, df.FP, df.FN
f1 = 2 * tp / (2 * tp + fp + fn)
print("max |F1 - F1(TP,FP,FN)| =", float((f1 - df.F1).abs().max()))
for m in ["F1", "MCC", "PR_AUC", "ROC_AUC", "Precision", "Recall"]:
    print(m, "range", round(df[m].min(), 4), round(df[m].max(), 4))
df.to_csv(dst, index=False); print("saved", dst, len(df), "rows")
