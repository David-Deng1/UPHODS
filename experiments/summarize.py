"""Summarise a results file: per-instance medians, mean ranks, win/tie/loss
of a target method against every other method (Mann-Whitney U, two-sided,
alpha = 0.05), Friedman test and Wilcoxon-Holm tests on instance medians.

Example:
    uv run python experiments/summarize.py results/dev/v1.jsonl --target UPHODS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon


def load(path):
    rows = [
        json.loads(l)
        for l in Path(path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return pd.DataFrame(rows)


def add_families(df):
    """Per-instance best-of-family pseudo-methods (chosen by median), as in
    the 0201 version: FSS* = best of FSS-1..4, Single* = best single solver."""
    extra = []
    for fam, prefix in (("FSS*", "FSS-"), ("Single*", "Single-")):
        sub = df[df.method.str.startswith(prefix)]
        for (fn, d), g in sub.groupby(["function", "dim"]):
            best = g.groupby("method").best.median().idxmin()
            e = g[g.method == best].copy()
            e["method"] = fam
            e["picked"] = best
            extra.append(e)
    return pd.concat([df] + extra, ignore_index=True) if extra else df


def holm(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--target", default="UPHODS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = add_families(load(a.path))
    T = a.target
    lines = []
    P = lines.append

    med = df.pivot_table(
        index=["function", "dim"], columns="method", values="best", aggfunc="median"
    )
    methods = [m for m in med.columns]
    ranks = med.rank(axis=1)
    core = [m for m in methods if m not in ("FSS*", "Single*")]
    rank_core = med[core].rank(axis=1)
    P(
        f"# Summary of {a.path}  (target {T}, {df.seed.nunique()} seeds, {len(med)} instances)\n"
    )
    P("## Mean rank over instances (lower is better; families excluded from ranking)\n")
    mr = rank_core.mean().sort_values()
    for m, v in mr.items():
        P(f"- {m}: {v:.2f}")
    if len(core) > 2:
        st = friedmanchisquare(*[med[m].values for m in core])
        P(f"\nFriedman chi2 = {st.statistic:.2f}, p = {st.pvalue:.2e}\n")

    P(
        f"## {T} vs others: per-instance Mann-Whitney (p<0.05) W/T/L, and Wilcoxon on medians\n"
    )
    P("| vs | W | T | L | median rel. diff | Wilcoxon p (one-sided) | Holm |")
    P("|---|---|---|---|---|---|---|")
    others = [m for m in methods if m != T]
    pw, rows = [], []
    for m in others:
        w = t = l = 0
        rel = []
        for (fn, d), g in df.groupby(["function", "dim"]):
            x = g[g.method == T].best.values
            y = g[g.method == m].best.values
            if len(x) == 0 or len(y) == 0:
                continue
            if len(x) == len(y) and np.allclose(x, y):
                t += 1
                continue
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            if p < 0.05:
                if np.median(x) < np.median(y):
                    w += 1
                else:
                    l += 1
            else:
                t += 1
            rel.append((np.median(y) - np.median(x)) / (abs(np.median(y)) + 1e-12))
        a_, b_ = med[T].values, med[m].values
        try:
            p1 = wilcoxon(a_, b_, alternative="less").pvalue
        except ValueError:
            p1 = 1.0
        pw.append(p1)
        rows.append((m, w, t, l, np.median(rel) if rel else np.nan, p1))
    adj = holm(pw)
    for (m, w, t, l, r, p1), ph in zip(rows, adj):
        P(f"| {m} | {w} | {t} | {l} | {100 * r:+.1f}% | {p1:.2e} | {ph:.2e} |")

    P("\n## Per-instance medians\n")
    show = [T] + [
        m for m in ["GP-PO", "SW-UCB", "FSS*", "Single*"] if m in methods and m != T
    ]
    show += [m for m in methods if m not in show]
    P("| function | D | " + " | ".join(show) + " |")
    P("|---|---|" + "---|" * len(show))
    for (fn, d), row in med.iterrows():
        best = row[[m for m in core]].min()
        cells = []
        for m in show:
            v = row[m]
            s = f"{v:.3g}"
            cells.append(f"**{s}**" if np.isclose(v, best) else s)
        P(f"| {fn} | {d} | " + " | ".join(cells) + " |")
    if "picked" in df:
        P("\n## Best-of-family picks\n")
        pk = (
            df.dropna(subset=["picked"])
            .groupby(["method", "function", "dim"])
            .picked.first()
        )
        P(
            pk.reset_index()
            .pivot_table(
                index=["function", "dim"],
                columns="method",
                values="picked",
                aggfunc="first",
            )
            .to_string()
        )
    if "ns" in df:
        P("\n## Mean number of switches\n")
        P(
            df.pivot_table(
                index="method", values=["ns", "aip", "wall_seconds"], aggfunc="mean"
            ).to_string()
        )
    txt = "\n".join(lines)
    out = Path(a.out) if a.out else Path(a.path).with_suffix(".summary.md")
    out.write_text(txt, encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
