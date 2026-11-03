"""Summary of the mechanism scenarios (PROTOCOL.md section 10).

    uv run python experiments/scenario_report.py results/final/scen > results/final/scen/summary.md

For each setting: feasibility rates (penalty), UPHODS-v2 vs its ablations
(two-sided tests, Holm within the setting), mean ranks of the non-ablation
methods and UPHODS-v2 vs every baseline (one-sided Wilcoxon, Holm).
Errors below 1e-8 count as 0 (CEC convention).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
D = Path(sys.argv[1])
T = "UPHODS-v2"
SETTINGS = [
    ("penalty", "1000"),
    ("penalty", "1000000"),
    ("discrete", "10"),
    ("discrete", "30"),
]


def load(paths):
    rows = []
    for p in paths:
        if p.exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df["best"] = np.where(df.best < 1e-8, 0.0, df.best)
    return df


def wil(x, y, alt):
    if np.allclose(x, y):
        return 1.0
    try:
        return wilcoxon(x, y, alternative=alt).pvalue
    except ValueError:
        return 1.0


def holm(ps):
    adj, run = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.get)):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(run, 1.0)
    return adj


def wtl(df, a, b):
    w = t = l = 0
    for _, g in df.groupby(["function", "dim"]):
        x, y = g[g.method == a].best.values, g[g.method == b].best.values
        if (
            np.allclose(x, y)
            or mannwhitneyu(x, y, alternative="two-sided").pvalue >= 0.05
        ):
            t += 1
        elif np.median(x) < np.median(y):
            w += 1
        else:
            l += 1
    return w, t, l


for sc, lv in SETTINGS:
    df = load([D / f"{sc}_{lv}_{k}.jsonl" for k in ("uphods", "base", "swucb")])
    if df.empty:
        continue
    print(
        f"\n## {sc} level {lv}  ({df.seed.nunique()} seeds, {df.groupby(['function', 'dim']).ngroups} instances)\n"
    )
    if "feasible" in df:
        fr = df.groupby("method").feasible.mean().sort_values()
        print(
            "Share of runs whose best design is feasible:",
            ", ".join(f"{k} {v:.0%}" for k, v in fr.items()),
        )
        print(f"All methods together: {df.feasible.mean():.0%}\n")
    med = df.pivot_table(
        index=["function", "dim"], columns="method", values="best", aggfunc="median"
    )
    abl = [m for m in med.columns if m.startswith(T + "-")]
    print("UPHODS-v2 vs ablations (two-sided; W/T/L from UPHODS-v2's side)\n")
    print(
        "| variant | W/T/L | UPHODS-v2 better on | Wilcoxon p | Holm | mean NS (variant / v2) |"
    )
    print("|---|---|---|---|---|---|")
    ps = {v: wil(med[T], med[v], "two-sided") for v in abl}
    adj = holm(ps)
    ns = df.groupby("method").ns.mean()
    for v in abl:
        better = np.mean(med[T] < med[v])
        print(
            f"| {v} | {'%d/%d/%d' % wtl(df, T, v)} | {better:.0%} | {ps[v]:.2e} | {adj[v]:.2e} | "
            f"{ns[v]:.1f} / {ns[T]:.1f} |"
        )
    core = [m for m in med.columns if m not in abl]
    rk = med[core].rank(axis=1).mean().sort_values()
    print(
        "\nMean ranks (non-ablation methods):",
        ", ".join(f"{k} {v:.2f}" for k, v in rk.items()),
    )
    print(f"Friedman p = {friedmanchisquare(*[med[c] for c in core]).pvalue:.1e}\n")
    ps = {m: wil(med[T], med[m], "less") for m in core if m != T}
    adj = holm(ps)
    print("| vs | W/T/L | Wilcoxon p (one-sided) | Holm |")
    print("|---|---|---|---|")
    for m in sorted(ps, key=ps.get):
        print(f"| {m} | {'%d/%d/%d' % wtl(df, T, m)} | {ps[m]:.2e} | {adj[m]:.2e} |")
    print("\nPer-instance medians (UPHODS-v2, -noRN, -fixedW, SW-UCB):\n")
    cols = [c for c in [T, T + "-noRN", T + "-fixedW", "SW-UCB"] if c in med]
    print(med[cols].to_string(float_format=lambda v: f"{v:.4g}"))
