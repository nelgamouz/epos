import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=pd.read_csv('all_rows.csv'); w=pd.read_csv('wilcoxon_search.csv'); c=pd.read_csv('curve_summary.csv')
DS={'churn':'Telco Churn','creditcard':'Credit Card','paysim':'PaySim'}; CL={'lr':'LR','rf':'RF','lgbm':'LGBM','xgb':'XGB'}
mean=r[r.method!='Curve'].groupby(['dataset','classifier','method']).F1.mean()
pairs=[('EPOS-orig','RS-orig'),('EPOS-orig','Grid-orig'),('EPOS-wide','RS-wide'),('EPOS-wide','Grid-wide'),('EPOS-wide','EPOS-orig')]
L=[r'\begin{table}[t]\centering\small',r'\caption{Search-strategy and search-space ablation (outer-test F1, \%). $\Delta$ = mean paired difference A$-$B over outer folds, with wins/losses of A in parentheses. Random search (RS) and grid search use the same number of evaluated configurations as EPOS in each fold. No difference is significant after Holm correction (Wilcoxon signed-rank, $\alpha=0.05$, five comparisons per dataset--classifier pair).}\label{tab:search_ablation}',
r'\setlength{\tabcolsep}{3pt}\begin{tabular}{llcc rrrrr}\toprule',
r'& & & EPOS-orig & \multicolumn{2}{c}{EPOS-orig vs.} & \multicolumn{2}{c}{EPOS-wide vs.} & EPOS-wide vs.\\',
r'Dataset & Clf. & Folds & F1 & RS-orig & Grid-orig & RS-wide & Grid-wide & EPOS-orig\\\midrule']
for d in DS:
  for k in CL:
    s=w[(w.dataset==d)&(w.clf==k)]
    if s.empty: continue
    f=int(s.folds.iloc[0]); cells=[]
    for a,b in pairs:
      x=s[(s.A==a)&(s.B==b)].iloc[0]; cells.append(f"{x.dF1:+.2f} ({int(x.W)}/{int(x.L)})")
    fl=f"{f}" + (r"$^\ddagger$" if (d!='churn' and f<10) or (d=='churn' and f<15) else "")
    L.append(f"{DS[d]} & {CL[k]} & {fl} & {100*mean[(d,k,'EPOS-orig')]:.2f} & "+" & ".join(cells)+r"\\")
  L.append(r'\midrule' if d!='paysim' else r'\bottomrule')
L+= [r'\end{tabular}',r'\par\footnotesize $^\ddagger$Incomplete run (fewer outer folds than planned); interpret with caution.\end{table}']
open('search_ablation_table.tex','w').write("\n".join(L))
# curve table
T=[r'\begin{table}[t]\centering\small',r'\caption{Effect of the oversampling amount $r=n/|\mathcal{D}^{+}_{\text{train}}|$ ($\sigma=0.05$, threshold tuned on inner folds, five outer folds). $r=0$: no synthetic samples (threshold tuning only).}\label{tab:curve}',
r'\begin{tabular}{llrrrrc}\toprule',r'Dataset & Clf. & F1 ($r{=}0$) & best $r$ & F1 (best) & Range & EPOS-orig $-$ ($r{=}0$) (W/L)\\\midrule']
for d in DS:
  for k in CL:
    x=c[(c.dataset==d)&(c.clf==k)]
    if x.empty: continue
    x=x.iloc[0]
    if x.folds<5: continue
    T.append(f"{DS[d]} & {CL[k]} & {x.F1_r0:.2f} & {x.best_r:g} & {x.F1_best:.2f} & {x.curve_range:.2f} & {x.EPOSorig_minus_r0:+.2f} ({int(x.W)}/{int(x.L)})\\\\")
  T.append(r'\midrule' if d!='paysim' else r'\bottomrule')
T+=[r'\end{tabular}\end{table}']
open('curve_table.tex','w').write("\n".join(T))
# figure
cv=r[r.method=='Curve']; fig,ax=plt.subplots(1,3,figsize=(12,3.4))
for i,d in enumerate(DS):
  for k in CL:
    g=cv[(cv.dataset==d)&(cv.classifier==k)]
    if g.empty or g.fold.nunique()*g['repeat'].nunique()<5: continue
    m=g.groupby('sel_r_target').F1.agg(['mean','sem'])*100
    ax[i].errorbar(m.index,m['mean'],yerr=m['sem'],color={'lr':'C0','rf':'C1','lgbm':'C2','xgb':'C3'}[k],marker='o',ms=3,capsize=2,label=CL[k])
  ax[i].set_title(DS[d]); ax[i].set_xlabel(r'oversampling ratio $r$'); ax[i].grid(alpha=.3)
ax[0].set_ylabel('outer-test F1 (%)'); [a.legend(fontsize=8) for a in ax]
plt.tight_layout(); plt.savefig('oversampling_curve.pdf'); plt.savefig('oversampling_curve.png',dpi=150)
print(cv.groupby(['dataset','classifier']).apply(lambda g: g.groupby(['repeat','fold']).ngroups))
