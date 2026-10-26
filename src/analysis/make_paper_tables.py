import pandas as pd, numpy as np, sys
NAMES={"lr":"LR","rf":"RF","lgbm":"LightGBM","xgb":"XGBoost","ALL (pooled)":"Pooled"}
ABBR={"No oversampling":"NoOv","Class weights":"CW","SMOTE":"SMOTE","ADASYN":"ADASYN","Borderline-SMOTE":"B-SMOTE","SMOTE-ENN":"SMOTE-ENN"}
def fp(p): return "$<$0.001" if p<0.001 else f"{p:.3f}"
def tables(d, name, lab):
    f=pd.read_csv(f"{d}/friedman.csv"); w=pd.read_csv(f"{d}/wilcoxon_vs_epos.csv")
    out=[]; md=[]
    out+= [r"\begin{table}[!t]",r"\centering",r"\scriptsize",
      rf"\caption{{Friedman test on the {name} dataset (paired outer test folds, $\alpha=0.05$). $p$: Friedman $p$-value; CD: Nemenyi critical difference; EPOS: average rank (position among 7 methods, 1 = best); Best: best-ranked method on F1.}}",
      rf"\label{{tab:friedman_{lab}}}",r"\begin{tabular}{lrrrrlll}",r"\toprule",
      r" & \multicolumn{3}{c}{Friedman $p$} & & \multicolumn{2}{c}{EPOS rank} & \\ \cmidrule(lr){2-4}\cmidrule(lr){6-7}",
      r"Classifier & F1 & MCC & PR-AUC & CD & F1 & PR-AUC & Best (F1) \\",r"\midrule"]
    md.append(f"| Classifier | Friedman p F1 | MCC | PR-AUC | CD | EPOS rank F1 | EPOS rank PR-AUC | Best (F1) |\n|---|---|---|---|---|---|---|---|")
    for c in ["lr","rf","lgbm","xgb","ALL (pooled)"]:
        g=f[f.classifier==c].set_index("metric")
        if not len(g): continue
        if c.startswith("ALL"): out.append(r"\midrule")
        rk=lambda m: f"{g.loc[m,'ref_avg_rank']:.2f} ({int(g.loc[m,'ref_position'])})"
        best=g.loc["F1","best_by_rank"].replace("EPOS (ours)","EPOS")
        out.append(f"{NAMES[c]} & {fp(g.loc['F1','p_friedman'])} & {fp(g.loc['MCC','p_friedman'])} & {fp(g.loc['PR_AUC','p_friedman'])} & {g.loc['F1','CD_nemenyi']:.2f} & {rk('F1')} & {rk('PR_AUC')} & {best} \\\\")
        md.append(f"| {NAMES[c]} | {fp(g.loc['F1','p_friedman'])} | {fp(g.loc['MCC','p_friedman'])} | {fp(g.loc['PR_AUC','p_friedman'])} | {g.loc['F1','CD_nemenyi']:.2f} | {rk('F1')} | {rk('PR_AUC')} | {best} |".replace("$<$","<"))
    out+=[r"\bottomrule",r"\end{tabular}",r"\end{table}",""]
    w=w[w.metric=="F1"]; bases=list(ABBR)
    out+=[r"\begin{table*}[!t]",r"\centering",r"\scriptsize",
      rf"\caption{{Wilcoxon signed-rank test, EPOS vs.\ each baseline on the {name} dataset (F1). Cell: mean paired difference EPOS $-$ baseline in percentage points, Holm-adjusted $p$-value and wins/ties/losses of EPOS over the outer folds. $^{{+}}$/$^{{-}}$: EPOS significantly better/worse ($\alpha=0.05$); $^{{\dagger}}$: also significant with the Nadeau--Bengio corrected $t$-test.}}",
      rf"\label{{tab:wilcoxon_{lab}}}",r"\begin{tabular}{l"+"c"*6+"}",r"\toprule",
      "Classifier & "+" & ".join(ABBR[b] for b in bases)+r" \\",r"\midrule"]
    md.append("\n| Classifier | "+" | ".join(ABBR[b] for b in bases)+" |\n|"+"---|"*7)
    for c in ["lr","rf","lgbm","xgb"]:
        cells=[];mcells=[]
        for b in bases:
            r=w[(w.classifier==c)&(w.baseline==b)]
            if not len(r): cells.append("--"); mcells.append("–"); continue
            r=r.iloc[0]; sig=r.p_wilcoxon_holm<0.05
            mk=("$^{+}$" if r.mean_diff>0 else "$^{-}$") if sig else ""; mk+=r"$^{\dagger}$" if sig and r.p_nb_corrected<0.05 else ""
            mm=("**" if sig else "")
            cells.append(f"{100*r.mean_diff:+.2f}{mk} ({fp(r.p_wilcoxon_holm)}; {r.wins}/{r.ties}/{r.losses})")
            mcells.append(f"{mm}{100*r.mean_diff:+.2f}{mm} ({fp(r.p_wilcoxon_holm)}; {r.wins}/{r.ties}/{r.losses})".replace("$<$","<"))
        if not any(x!="--" for x in cells): continue
        out.append(f"{NAMES[c]} & "+" & ".join(cells)+r" \\"); md.append(f"| {NAMES[c]} | "+" | ".join(mcells)+" |")
    out+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}",""]
    return "\n".join(out), "\n".join(md)
for d,n,l in [("stats_creditcard","Credit Card","creditcard"),("stats_paysim","PaySim","paysim")]:
    t,m=tables(d,n,l); open(f"paper_tables_{l}.tex","w").write(t); print(f"### {n}\n"+m+"\n")
