"""Summary for the misaligned-preference scenario: DM utility of the picked
design (higher is better).  Usage: summarize_misaligned.py file1 file2 ..."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

files = sys.argv[1:]
target = "UPHODS-v2"
df = pd.concat([pd.DataFrame([json.loads(l) for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]) for f in files])
for col in ("dm_pick_u", "dm_best_u"):
    med = df.pivot_table(index=["function", "dim"], columns="method", values=col, aggfunc="median")
    rank = (-med).rank(axis=1).mean().sort_values()
    print(f"\n### {col}: mean rank (higher utility = better)")
    print(rank.round(2).to_string())
    print(f"\n{target} vs others (Mann-Whitney p<0.05, W/T/L) and Wilcoxon on medians (one-sided)")
    for m in [c for c in med if c != target]:
        w = t = l = 0
        for (fn, d), g in df.groupby(["function", "dim"]):
            x = g[g.method == target][col].values; y = g[g.method == m][col].values
            p = mannwhitneyu(x, y).pvalue
            if p < 0.05:
                w, l = (w + 1, l) if np.median(x) > np.median(y) else (w, l + 1)
            else:
                t += 1
        pw = wilcoxon(med[target], med[m], alternative="greater").pvalue
        print(f"  {m:18s} {w}/{t}/{l}   p={pw:.1e}")
    if col == "dm_pick_u":
        pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:.3g}")
        print(med.to_string())
