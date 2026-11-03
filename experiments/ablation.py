"""Two-sided comparison of UPHODS-v2 with its ablation variants.

usage: ablation.py <jsonl> [--target UPHODS-v2]

Per instance: Mann-Whitney U (two-sided, p<0.05) win/tie/loss of the target.
Overall: Wilcoxon signed-rank on instance medians (two-sided), Holm-adjusted,
mean rank difference, mean number of switches and the share of time spent on
the GP.  Errors below 1e-8 count as 0 (CEC convention), as in criteria.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--target", default="UPHODS-v2")
    a = ap.parse_args()
    df = pd.DataFrame(
        [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    )
    df = df[df.method.str.startswith(a.target)].copy()
    df["best"] = np.where(df.best < 1e-8, 0.0, df.best)
    variants = sorted(m for m in df.method.unique() if m != a.target)
    med = df.pivot_table(
        index=["function", "dim"], columns="method", values="best", aggfunc="median"
    )
    rows, ps = [], {}
    for v in variants:
        w = t = l = 0
        for (fn, d), g in df.groupby(["function", "dim"]):
            x = g[g.method == a.target].best.values
            y = g[g.method == v].best.values
            if np.allclose(x, y):
                t += 1
                continue
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            if p >= 0.05:
                t += 1
            elif np.median(x) < np.median(y):
                w += 1
            else:
                l += 1
        try:
            ps[v] = wilcoxon(med[a.target], med[v]).pvalue
        except ValueError:
            ps[v] = 1.0
        better = float(np.mean(med[a.target] < med[v]))
        rows.append(dict(variant=v, W=w, T=t, L=l, share_target_better=better))
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for i, v in enumerate(order):
        run = max(run, (len(order) - i) * ps[v])
        adj[v] = min(run, 1.0)
    ns = df.groupby("method").ns.mean()
    gp = df.groupby("method").apply(
        lambda g: (g.gp_seconds / g.wall_seconds).mean(), include_groups=False
    )
    print(f"target {a.target}: mean switches {ns[a.target]:.1f}")
    print(
        "| variant | W | T | L | target better on (instances) | Wilcoxon p (two-sided) | Holm | mean NS |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        v = r["variant"]
        print(
            f"| {v} | {r['W']} | {r['T']} | {r['L']} | {r['share_target_better']:.0%} | "
            f"{ps[v]:.2e} | {adj[v]:.2e} | {ns[v]:.1f} |"
        )
    print("GP time share of wall time (mean over runs):", gp.round(3).to_dict())


if __name__ == "__main__":
    main()
