"""Tables for Section 4 of the revised manuscript, from the final runs.

    uv run python experiments/paper_tables.py results/final > results/final/paper_tables.md

Conventions (PROTOCOL.md section 6):
* errors below 1e-8 count as 0 (CEC convention);
* per instance: two-sided Mann-Whitney U, alpha = 0.05 -> win/tie/loss of UPHODS;
* overall: Friedman test and mean ranks on instance medians; Wilcoxon signed-rank
  on instance medians, one-sided (UPHODS better) against baselines, two-sided
  against ablations; Holm adjustment within each table (family stated in the
  table note);
* A12 = Vargha-Delaney probability that a UPHODS run ends better than a
  baseline run (ties count one half), averaged over instances;
* FSS* / Single*: per-instance best member of the family by median, the
  conservative comparison of the original submission; reported with
  unadjusted p because they are not part of the pre-registered family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
R = Path(sys.argv[1])
T = "UPHODS-v2"
NAME = {
    "UPHODS-v2": "UPHODS",
    "GP-PO-v2": "GP-PO",
    "Single-GA": "GA",
    "Single-PSO": "PSO",
    "Single-SA": "SA",
    "Single-GD": "GD",
    "UPHODS-v2-noHys": "UPHODS-noHys",
    "UPHODS-v2-noET": "UPHODS-noET",
    "UPHODS-v2-fixedW": "UPHODS-fixedW",
    "UPHODS-v2-noRN": "UPHODS-noRN",
}
CORE = [
    "UPHODS-v2",
    "SW-UCB",
    "FSS-1",
    "FSS-2",
    "FSS-3",
    "FSS-4",
    "GP-PO-v2",
    "Single-GA",
    "Single-PSO",
    "Single-SA",
    "Single-GD",
]
ABL = ["UPHODS-v2-noHys", "UPHODS-v2-noET", "UPHODS-v2-fixedW", "UPHODS-v2-noRN"]


def nm(m):
    return NAME.get(m, m)


def load(*paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    df = pd.DataFrame(rows)
    df["best"] = np.where(df.best < 1e-8, 0.0, df.best)
    return df


def add_families(df, col="best", higher=False):
    extra = []
    for fam, prefix in (("FSS*", "FSS-"), ("Single*", "Single-")):
        sub = df[df.method.str.startswith(prefix)]
        for _, g in sub.groupby(["function", "dim"]):
            med = g.groupby("method")[col].median()
            best = med.idxmax() if higher else med.idxmin()
            e = g[g.method == best].copy()
            e["method"] = fam
            extra.append(e)
    return pd.concat([df] + extra, ignore_index=True)


def wtl(df, a, b, col="best", higher=False):
    w = t = l = 0
    for _, g in df.groupby(["function", "dim"]):
        x, y = g[g.method == a][col].values, g[g.method == b][col].values
        if len(x) == len(y) and np.allclose(x, y):
            t += 1
        elif mannwhitneyu(x, y, alternative="two-sided").pvalue >= 0.05:
            t += 1
        elif (np.median(x) > np.median(y)) == higher:
            w += 1
        else:
            l += 1
    return w, t, l


def a12(df, a, b, col="best", higher=False):
    vals = []
    for _, g in df.groupby(["function", "dim"]):
        x = g[g.method == a][col].values[:, None]
        y = g[g.method == b][col].values[None, :]
        better = (x > y) if higher else (x < y)
        vals.append(np.mean(better + 0.5 * (x == y)))
    return float(np.mean(vals))


def relimp(df, a, b, col="best"):
    """Median over instances of (median_b - median_a) / |median_b|: the
    relative reduction of the median error of b achieved by a (same
    definition as experiments/summarize.py; instances on which the two
    methods produced identical runs are skipped)."""
    r = []
    for _, g in df.groupby(["function", "dim"]):
        x, y = g[g.method == a][col].values, g[g.method == b][col].values
        if len(x) == len(y) and np.allclose(x, y):
            continue
        r.append((np.median(y) - np.median(x)) / (abs(np.median(y)) + 1e-12))
    return float(np.median(r)) if r else float("nan")


def wil(x, y, alt):
    if np.allclose(x, y):
        return 1.0
    try:
        return float(wilcoxon(x, y, alternative=alt).pvalue)
    except ValueError:
        return 1.0


def holm(ps):
    adj, run = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.get)):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(run, 1.0)
    return adj


def fmt_p(p):
    return "1" if p >= 0.9995 else (f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}")


def comparison_table(df, methods, title, col="best", higher=False, note=""):
    df = add_families(df[df.method.isin(methods)], col, higher)
    med = df.pivot_table(
        index=["function", "dim"], columns="method", values=col, aggfunc="median"
    )
    sign = -1 if higher else 1
    rk = (sign * med[methods]).rank(axis=1).mean()
    fried = friedmanchisquare(*[med[m] for m in methods]).pvalue
    alt = "greater" if higher else "less"
    base = [m for m in methods if m != T]
    ps = {m: wil(med[T], med[m], alt) for m in base}
    adj = holm(ps)
    print(f"\n### {title}\n")
    print(
        f"{len(med)} instances, 30 runs each. Friedman test over the {len(methods)} methods: p = {fried:.1e}. "
        f"Mean rank of UPHODS: {rk[T]:.2f}.\n"
    )

    def ri(b):
        return "" if higher else f" {100 * relimp(df, T, b, col):.1f}% |"

    print(
        "| Baseline | Mean rank | W/T/L | Mean A12 |" + ("" if higher else " Median rel. error reduction |") + " Wilcoxon p | Holm p |"
    )
    print("|---|---|---|---|" + ("" if higher else "---|") + "---|---|")
    for m in sorted(base, key=lambda k: rk[k]):
        print(
            f"| {nm(m)} | {rk[m]:.2f} | {'%d/%d/%d' % wtl(df, T, m, col, higher)} | "
            f"{a12(df, T, m, col, higher):.2f} |{ri(m)} {fmt_p(ps[m])} | {fmt_p(adj[m])} |"
        )
    for fam in ("FSS*", "Single*"):
        if fam in med:
            p = wil(med[T], med[fam], alt)
            print(
                f"| {fam} | — | {'%d/%d/%d' % wtl(df, T, fam, col, higher)} | "
                f"{a12(df, T, fam, col, higher):.2f} |{ri(fam)} {fmt_p(p)} | (unadjusted) |"
            )
    print(
        f"\nW/T/L: instances on which UPHODS is significantly better / not different / worse "
        f"(two-sided Mann-Whitney U, 0.05). Wilcoxon: one-sided, instance medians. "
        f"Holm family: the {len(base)} individual baselines. {note}"
    )
    return df, med, rk


main = load(R / "main_uphods.jsonl", R / "main_base.jsonl", R / "main_swucb.jsonl")
print("# Tables for Section 4 (generated by experiments/paper_tables.py; seeds 0-29)")

# ------------------------------------------------------------ main table
dfm, medm, rkm = comparison_table(
    main, CORE, "Table 5. Main comparison (methods as originally defined)"
)

# ------------------------------------------------------------ by dimension
print("\n### Table 6. UPHODS against SW-UCB and Single* by dimension\n")
print(
    "| D | vs SW-UCB W/T/L | A12 | vs Single* W/T/L | A12 | mean rank of UPHODS (11 methods) |"
)
print("|---|---|---|---|---|---|")
rank_inst = medm[CORE].rank(axis=1)
for d in sorted(dfm.dim.unique()):
    sub = dfm[dfm.dim == d]
    print(
        f"| {d} | {'%d/%d/%d' % wtl(sub, T, 'SW-UCB')} | {a12(sub, T, 'SW-UCB'):.2f} | "
        f"{'%d/%d/%d' % wtl(sub, T, 'Single*')} | {a12(sub, T, 'Single*'):.2f} | "
        f"{rank_inst.xs(d, level='dim')[T].mean():.2f} |"
    )

# ------------------------------------------------------------ controlled
# FSS and SW-UCB with the archive transfer; the single solvers (no switching) from the main runs
base = load(R / "main_base.jsonl")
ctrl = pd.concat(
    [
        load(R / "main_uphods.jsonl"),
        load(R / "ctrl_fss.jsonl"),
        load(R / "ctrl_swucb.jsonl"),
        base[base.method.str.startswith("Single-")],
    ],
    ignore_index=True,
)
CTRL = [m for m in CORE if m != "GP-PO-v2"]
comparison_table(
    ctrl,
    CTRL,
    "Table 7. Controlled comparison: every hybrid uses the elite-archive transfer",
    note="Single solvers are unchanged from Table 5.",
)

# ------------------------------------------------------------ ablations
print(
    "\n### Table 8. Ablations (two-sided tests; W/T/L from the side of full UPHODS)\n"
)
mis = load(R / "mis_uphods.jsonl", R / "mis_base.jsonl", R / "mis_swucb.jsonl")
scen = []
for sc, lv in (
    ("penalty", "1000"),
    ("penalty", "1000000"),
    ("discrete", "10"),
    ("discrete", "30"),
):
    d = load(
        *[R / "scen" / f"{sc}_{lv}_{k}.jsonl" for k in ("uphods", "base", "swucb")]
    )
    d["setting"] = f"{sc}-{lv}"
    scen.append(d)
scen = pd.concat(scen, ignore_index=True)
medA = main.pivot_table(
    index=["function", "dim"], columns="method", values="best", aggfunc="median"
)
psA = {v: wil(medA[T], medA[v], "two-sided") for v in ABL}
adjA = holm(psA)
medM = mis.pivot_table(
    index=["function", "dim"], columns="method", values="dm_pick_u", aggfunc="median"
)
psM = {v: wil(medM[T], medM[v], "two-sided") for v in ABL}
adjM = holm(psM)
ns = main.groupby("method").ns.mean()
print(
    "| Variant | Benchmark W/T/L (36) | Holm p | Mean switches | Misaligned W/T/L (12) | Holm p | "
    "Mechanism scenarios W/T/L (48) | smallest Holm p |"
)
print("|---|---|---|---|---|---|---|---|")
print(f"| UPHODS (full) | — | — | {ns[T]:.1f} | — | — | — | — |")
for v in ABL:
    w = t = l = 0
    ph = []
    for st, d in scen.groupby("setting"):
        a, b, c = wtl(d, T, v)
        w, t, l = w + a, t + b, l + c
        med = d.pivot_table(
            index=["function", "dim"], columns="method", values="best", aggfunc="median"
        )
        ps = {u: wil(med[T], med[u], "two-sided") for u in ABL}
        ph.append(holm(ps)[v])
    print(
        f"| {nm(v)} | {'%d/%d/%d' % wtl(main, T, v)} | {fmt_p(adjA[v])} | {ns[v]:.1f} | "
        f"{'%d/%d/%d' % wtl(mis, T, v, 'dm_pick_u', True)} | {fmt_p(adjM[v])} | {w}/{t}/{l} | {fmt_p(min(ph))} |"
    )
print(
    "\nHolm family: the four ablations within each setting. Misaligned scenario: utility of the "
    "design the simulated designer picks (higher is better). Mechanism scenarios: static-penalty "
    "constraint (P = 1e3, 1e6) and discrete variables (K = 10, 30), 6 functions x D in {10, 30} each; "
    "these scenarios were designed after the benchmark results were known (Section 4.4)."
)
nsw = scen.groupby(["setting", "method"]).ns.mean().unstack()
print(
    "\nMean switches in the mechanism scenarios: "
    + "; ".join(
        f"{s}: UPHODS {nsw.loc[s, T]:.1f}, noHys {nsw.loc[s, 'UPHODS-v2-noHys']:.1f}"
        for s in nsw.index
    )
)

# ------------------------------------------------------------ misaligned
comparison_table(
    mis,
    CORE,
    "Table 9. Preference-misaligned scenario (utility of the designer's pick, higher is better)",
    col="dm_pick_u",
    higher=True,
)

# ------------------------------------------------------------ switches and GP cost
print(
    "\n### Table 10. Switching frequency and preference-model cost of UPHODS by dimension (B = 1000)\n"
)
u = main[main.method == T]
g = u.groupby("dim").agg(
    ns=("ns", "mean"),
    pairs=("n_pairs", "mean"),
    designs=("n_train", "mean"),
    gp=("gp_seconds", "mean"),
    gpmax=("gp_max_seconds", "max"),
    gpmaxm=("gp_max_seconds", "mean"),
    wall=("wall_seconds", "mean"),
)
sw = main[main.method == "SW-UCB"].groupby("dim").ns.mean()
print(
    "| D | switches (UPHODS) | switches (SW-UCB) | comparisons | compared designs | GP time per run (s) | longest fit per run, mean (s) | longest fit overall (s) | run time (s) |"
)
print("|---|---|---|---|---|---|---|---|---|")
for d, r in g.iterrows():
    print(
        f"| {d} | {r.ns:.1f} | {sw[d]:.1f} | {r.pairs:.0f} | {r.designs:.0f} | {r.gp:.2f} | {r.gpmaxm:.3f} | {r.gpmax:.3f} | {r.wall:.2f} |"
    )
print(
    f"\nAll dimensions: UPHODS {u.ns.mean():.1f} switches per run, SW-UCB {main[main.method == 'SW-UCB'].ns.mean():.1f}. "
    "Times are wall-clock on one CPU thread; the benchmark objective itself costs microseconds."
)

# ------------------------------------------------------------ supplementary per-instance table
rows = []
for (fn, d), gi in main[main.method.isin(CORE)].groupby(["function", "dim"]):
    for m, gm in gi.groupby("method"):
        q1, q2, q3 = np.percentile(gm.best, [25, 50, 75])
        rows.append(
            dict(
                function=fn,
                dim=d,
                method=nm(m),
                median=q2,
                q1=q1,
                q3=q3,
                mean=gm.best.mean(),
                sd=gm.best.std(ddof=1),
                ns=gm.ns.mean(),
            )
        )
S = pd.DataFrame(rows)
S.to_csv(R / "tableS1_per_instance.csv", index=False, float_format="%.6e")
print(
    "\nSupplementary Table S1 (per-instance median, IQR, mean, SD, switches for all 11 methods) "
    "written to results/final/tableS1_per_instance.csv"
)
