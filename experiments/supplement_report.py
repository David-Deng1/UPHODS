"""Summaries of the supplementary experiments (PROTOCOL.md section 9).

    uv run python experiments/supplement_report.py results/final > results/final/supplement.summary.md

Errors below 1e-8 count as 0 (CEC convention).  Per-instance tests are
Mann-Whitney U (two-sided, p<0.05); overall tests are Wilcoxon signed-rank on
instance medians with Holm adjustment within each table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

R = Path(sys.argv[1])
sys.stdout.reconfigure(encoding="utf-8")


def load(*paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df["best"] = np.where(df.best < 1e-8, 0.0, df.best)
    return df


def wtl(a, b, col="best", higher=False):
    """Win/tie/loss of sample groups a vs b over instances (dicts inst -> array)."""
    w = t = l = 0
    for k in a:
        x, y = a[k], b[k]
        if np.allclose(x, y):
            t += 1
            continue
        if mannwhitneyu(x, y, alternative="two-sided").pvalue >= 0.05:
            t += 1
        elif (np.median(x) > np.median(y)) == higher:
            w += 1
        else:
            l += 1
    return w, t, l


def groups(df, method, col="best"):
    d = df[df.method == method]
    return {k: g[col].values for k, g in d.groupby(["function", "dim"])}


def wil(x, y, alternative="two-sided"):
    if np.allclose(x, y):  # identical medians on every instance
        return 1.0
    try:
        return wilcoxon(x, y, alternative=alternative).pvalue
    except ValueError:
        return 1.0


def holm(ps):
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, (len(order) - i) * ps[k])
        adj[k] = min(run, 1.0)
    return adj


def gmean_ratio(a, b):
    """Geometric mean over instances of median(a)/median(b) (floor 1e-8)."""
    r = [np.log(max(np.median(a[k]), 1e-8) / max(np.median(b[k]), 1e-8)) for k in a]
    return float(np.exp(np.mean(r)))


main = load(R / "main_uphods.jsonl", R / "main_base.jsonl", R / "main_swucb.jsonl")
mis = load(R / "mis_uphods.jsonl", R / "mis_base.jsonl", R / "mis_swucb.jsonl")

# ---------------------------------------------------------------- 9.1 noise
print("# Supplementary experiments (PROTOCOL 9), final seeds 0-29\n")
print("## 9.1 Preference noise (flip probability p)\n")
print(
    "Benchmark (36 instances, final error; ratio = geometric mean of median error vs p = 0, <1 is better).\n"
)
print(
    "| method | p | ratio vs p=0 | Wilcoxon p vs p=0 | W/T/L vs SW-UCB | one-sided p vs SW-UCB (x10 Bonferroni bound) | W/T/L vs FSS-4 |"
)
print("|---|---|---|---|---|---|---|")
sw, f4 = groups(main, "SW-UCB"), groups(main, "FSS-4")
for m in ["UPHODS-v2", "GP-PO-v2"]:
    base = groups(main, m)
    for p in [0.0, 0.1, 0.2, 0.3]:
        g = base if p == 0 else groups(load(R / f"noise/main_p{p}.jsonl"), m)
        keys = sorted(g)
        x = np.array([np.median(g[k]) for k in keys])
        x0 = np.array([np.median(base[k]) for k in keys])
        xs = np.array([np.median(sw[k]) for k in keys])
        pw = "—" if p == 0 else f"{wil(x, x0):.2e}"
        # Holm over the comparisons of the main table (10 baselines) is not
        # recomputed here; report the raw one-sided p and a x10 Bonferroni bound
        ps = wil(x, xs, "less")
        print(
            f"| {m} | {p} | {gmean_ratio(g, base):.3f} | {pw} | {'%d/%d/%d' % wtl(g, sw)} | "
            f"{ps:.2e} (x10: {min(1, 10 * ps):.2e}) | {'%d/%d/%d' % wtl(g, f4)} |"
        )
print(
    "\nMisaligned scenario (12 instances, utility of the DM's pick; higher is better).\n"
)
print(
    "| method | p | mean dm_pick_u | Wilcoxon p vs p=0 | W/T/L vs SW-UCB | one-sided p vs SW-UCB |"
)
print("|---|---|---|---|---|---|")
swm = groups(mis, "SW-UCB", "dm_pick_u")
for m in ["UPHODS-v2", "GP-PO-v2"]:
    base = groups(mis, m, "dm_pick_u")
    for p in [0.0, 0.1, 0.2, 0.3]:
        g = (
            base
            if p == 0
            else groups(load(R / f"noise/mis_p{p}.jsonl"), m, "dm_pick_u")
        )
        keys = sorted(g)
        x = np.array([np.median(g[k]) for k in keys])
        x0 = np.array([np.median(base[k]) for k in keys])
        xs = np.array([np.median(swm[k]) for k in keys])
        pw = "—" if p == 0 else f"{wil(x, x0):.2e}"
        print(
            f"| {m} | {p} | {np.mean([g[k].mean() for k in keys]):.3f} | {pw} | "
            f"{'%d/%d/%d' % wtl(g, swm, higher=True)} | {wil(x, xs, 'greater'):.2e} |"
        )

# ---------------------------------------------------------- 9.2 sensitivity
print(
    "\n## 9.2 One-at-a-time sensitivity of UPHODS-v2 (18 instances: 6 functions x D in {10, 30, 100})\n"
)
sub = main[main.dim.isin([10, 30, 100])]
dflt = groups(sub, "UPHODS-v2")
sws = groups(sub, "SW-UCB")
print(f"Default: W/T/L vs SW-UCB = {'%d/%d/%d' % wtl(dflt, sws)}\n")
print(
    "| parameter | value | ratio vs default | W/T/L vs default | Wilcoxon p | Holm | W/T/L vs SW-UCB | mean NS |"
)
print("|---|---|---|---|---|---|---|---|")
rows, ps = [], {}
for f in sorted((R / "sens").glob("*.jsonl")):
    key, val = f.stem.rsplit("_", 1)
    d = load(f)
    g = groups(d, "UPHODS-v2")
    keys = sorted(g)
    x = np.array([np.median(g[k]) for k in keys])
    x0 = np.array([np.median(dflt[k]) for k in keys])
    ps[f.stem] = wil(x, x0)
    rows.append(
        (
            key,
            float(val),
            f.stem,
            gmean_ratio(g, dflt),
            wtl(g, dflt),
            wtl(g, sws),
            d.ns.mean(),
        )
    )
adj = holm(ps)
ns0 = sub[sub.method == "UPHODS-v2"].ns.mean()
for key, val, stem, ratio, a, b, ns in sorted(rows):
    print(
        f"| {key} | {val:g} | {ratio:.3f} | {'%d/%d/%d' % a} | {ps[stem]:.2e} | {adj[stem]:.2e} | "
        f"{'%d/%d/%d' % b} | {ns:.1f} |"
    )
print(
    f"\n(default mean NS on these instances: {ns0:.1f}; W/T/L vs default counts the variant's wins)"
)

# ------------------------------------------------------------ 9.4 budget
print("\n## 9.4 Budget 5000 (12 instances: 6 functions x D in {30, 100})\n")
bud = load(R / "budget/uphods.jsonl", R / "budget/base.jsonl", R / "budget/swucb.jsonl")
med = bud.pivot_table(
    index=["function", "dim"], columns="method", values="best", aggfunc="median"
)
rk = med.rank(axis=1).mean().sort_values()
print("Mean ranks:", ", ".join(f"{k} {v:.2f}" for k, v in rk.items()))
print(
    f"\nFriedman p = {friedmanchisquare(*[med[c] for c in med.columns]).pvalue:.1e}\n"
)
tg = groups(bud, "UPHODS-v2")
ps = {m: wil(med["UPHODS-v2"], med[m], "less") for m in med.columns if m != "UPHODS-v2"}
adj = holm(ps)
print("| vs | W/T/L | Wilcoxon p (one-sided) | Holm |")
print("|---|---|---|---|")
for m in sorted(ps, key=ps.get):
    print(
        f"| {m} | {'%d/%d/%d' % wtl(tg, groups(bud, m))} | {ps[m]:.2e} | {adj[m]:.2e} |"
    )
print("\nUPHODS-v2 vs SW-UCB by dimension (B = 5000):")
for dim in [30, 100]:
    a = {k: v for k, v in tg.items() if k[1] == dim}
    b = {k: v for k, v in groups(bud, "SW-UCB").items() if k[1] == dim}
    print(f"- {dim}D: W/T/L {'%d/%d/%d' % wtl(a, b)}")
u = bud[bud.method == "UPHODS-v2"]
print("\nGP cost at B = 5000 (UPHODS-v2, mean per run):")
print(
    u.groupby("dim")[
        ["n_pairs", "n_train", "gp_seconds", "gp_max_seconds", "wall_seconds", "ns"]
    ]
    .mean()
    .round(3)
    .to_string()
)

# ------------------------------------------------------------ 9.3 timing
print("\n## 9.3 GP fit time vs number of pairs (median of 5, single-threaded BLAS)\n")
t = json.loads((R / "gp_timing.json").read_text(encoding="utf-8"))
print("Hardware:", t["info"])
print("\n| D | pairs | designs | cold fit (s) | warm-started fit (s) |")
print("|---|---|---|---|---|")
for r in t["rows"]:
    print(
        f"| {r['dim']} | {r['n_pairs']} | {r['n_designs']} | {r['cold_median_s']:.3f} | {r['warm_median_s']:.3f} |"
    )
