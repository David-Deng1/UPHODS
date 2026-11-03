import json, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
def load(p): return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])
R = "results/final/"
df = pd.concat([load(R+"main_uphods.jsonl"), load(R+"main_base.jsonl"), load(R+"main_swucb.jsonl")])
df["best"] = np.where(df.best < 1e-8, 0.0, df.best)
med = df.pivot_table(index=["function","dim"], columns="method", values="best", aggfunc="median")
print("per function: UPHODS-v2 better median on k/6 dims (sign-rank p one-sided)")
for b in ["SW-UCB","FSS-1","FSS-3","FSS-4","GP-PO-v2"]:
    out = {}
    for fn, g in med.groupby(level=0):
        k = int((g["UPHODS-v2"] < g[b]).sum())
        try: p = wilcoxon(g["UPHODS-v2"], g[b], alternative="less").pvalue
        except ValueError: p = 1.0
        out[fn] = f"{k}/6 p={p:.3f}"
    print(b, out)
print("\nvs SW-UCB by dimension (MWU two-sided W/T/L over 6 functions; P(superiority))")
for d, g in df[df.method.isin(["UPHODS-v2","SW-UCB"])].groupby("dim"):
    w=t=l=0; ps=[]
    for fn, h in g.groupby("function"):
        x=h[h.method=="UPHODS-v2"].best.values; y=h[h.method=="SW-UCB"].best.values
        p = mannwhitneyu(x,y,alternative="two-sided").pvalue if not np.allclose(x,y) else 1
        A = (np.sum(x[:,None]<y[None,:]) + 0.5*np.sum(x[:,None]==y[None,:]))/ (len(x)*len(y)); ps.append(A)
        if p>=0.05: t+=1
        elif np.median(x)<np.median(y): w+=1
        else: l+=1
    print(d, f"{w}/{t}/{l}", "A12 mean", round(np.mean(ps),2))
u = df[df.method=="UPHODS-v2"]
print("\nGP cost per run by dim (UPHODS-v2): mean gp_seconds, max single fit, n_pairs, n_train, wall")
print(u.groupby("dim")[["gp_seconds","gp_max_seconds","n_pairs","n_train","wall_seconds","ns"]].mean().round(3).to_string())
print("switches: v2", u.ns.mean().round(1), "SW-UCB", df[df.method=="SW-UCB"].ns.mean().round(1))
