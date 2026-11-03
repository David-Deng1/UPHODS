"""Figures for the revised manuscript, drawn from the final runs (seeds 0-29).

    uv run --with matplotlib python experiments/make_figures.py results/final

Writes PDF (vector, for submission) and PNG (600 dpi, preview) files to
results/final/figures/.  Every plotted value is computed here from the raw
JSON-lines records; nothing is typed in by hand.  Errors below 1e-8 count as
0 (CEC convention) and are drawn at the 1e-8 floor on log axes.

Instance choice for the 12-panel figures (box plots, convergence): all six
functions at D = 10 and D = 100, i.e. the low-dimensional regime where UPHODS
ties SW-UCB and the high-dimensional regime where it leads; the box plots of
all 36 instances are in the supplementary figure S1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

R = Path(sys.argv[1])
OUT = R / "figures"
OUT.mkdir(exist_ok=True)
FLOOR = 1e-8

# ---------------------------------------------------------------- style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated order
GRAY_FILL = "#d9d8d2"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    }
)

NAME = {  # code name -> name used in the manuscript
    "UPHODS-v2": "UPHODS",
    "SW-UCB": "SW-UCB",
    "FSS-1": "FSS-1",
    "FSS-2": "FSS-2",
    "FSS-3": "FSS-3",
    "FSS-4": "FSS-4",
    "GP-PO-v2": "GP-PO",
    "Single-GA": "GA",
    "Single-PSO": "PSO",
    "Single-SA": "SA",
    "Single-GD": "GD",
}
CORE = list(NAME)
FUNCS = [
    "Ackley",
    "Griewank",
    "Rastrigin",
    "Rosenbrock",
    "ShiftedRastrigin",
    "ShiftedSphere",
]
FLABEL = {"ShiftedRastrigin": "Shifted Rastrigin", "ShiftedSphere": "Shifted Sphere"}
DIMS = [5, 10, 20, 30, 50, 100]


def load(*paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    df = pd.DataFrame(rows)
    df["best"] = np.where(df.best < FLOOR, 0.0, df.best)
    return df


def tidy_log(ax):
    """Readable y-axis: linear when the panel spans less than one decade,
    log with 1-2-5 ticks for one to two decades, decade ticks otherwise."""
    lo, hi = ax.get_ylim()
    ratio = hi / max(lo, 1e-300)
    if ratio < 10:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(MaxNLocator(4))
    elif ratio < 100:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(), numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem)


def family_best(df, prefix, inst):
    """Per-instance best member of a family, chosen by median final error
    (the 0201 convention for FSS* and Single*)."""
    g = df[
        (df.method.str.startswith(prefix))
        & (df.function == inst[0])
        & (df.dim == inst[1])
    ]
    return g.groupby("method").best.median().idxmin()


main = load(R / "main_uphods.jsonl", R / "main_base.jsonl", R / "main_swucb.jsonl")
main = main[main.method.isin(CORE)]


# ------------------------------------------------------- box plots
def box_panel(ax, sub, show_x):
    data, cols = [], []
    for m in CORE:
        v = sub[sub.method == m].best.values
        data.append(np.maximum(v, FLOOR))
        cols.append(
            SLOT[0] if m == "UPHODS-v2" else SLOT[1] if m == "SW-UCB" else MUTED
        )
    bp = ax.boxplot(
        data,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=1.6, markeredgewidth=0, alpha=0.6),
        medianprops=dict(color=INK, linewidth=0.9),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        boxprops=dict(linewidth=0.6),
    )
    for patch, w, c in zip(bp["boxes"], range(len(CORE)), cols):
        patch.set_facecolor(c if c != MUTED else GRAY_FILL)
        patch.set_edgecolor(c if c != MUTED else MUTED)
    for k, c in enumerate(cols):
        ec = c
        for item in (
            bp["whiskers"][2 * k],
            bp["whiskers"][2 * k + 1],
            bp["caps"][2 * k],
            bp["caps"][2 * k + 1],
        ):
            item.set_color(ec)
        bp["fliers"][k].set_markerfacecolor(ec)
    ax.set_yscale("log")
    tidy_log(ax)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(1, len(CORE) + 1))
    if show_x:
        ax.set_xticklabels([NAME[m] for m in CORE], rotation=55, ha="right")
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)


def fig_boxes(dims, stem, width, row_h):
    fig, axes = plt.subplots(
        len(FUNCS), len(dims), figsize=(width, row_h * len(FUNCS)), squeeze=False
    )
    for i, fn in enumerate(FUNCS):
        for j, d in enumerate(dims):
            ax = axes[i, j]
            box_panel(
                ax, main[(main.function == fn) & (main.dim == d)], i == len(FUNCS) - 1
            )
            ax.set_title(f"{FLABEL.get(fn, fn)}, D = {d}", color=INK, pad=3)
            if j == 0:
                ax.set_ylabel("Final error")
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, stem)


fig_boxes([10, 100], "fig_boxplots_D10_D100", 6.9, 1.55)
fig_boxes(DIMS, "figS1_boxplots_all", 10.0, 1.55)


# ------------------------------------------------- convergence curves
SERIES = [  # (label, how to select the runs, colour, marker)
    ("UPHODS", "UPHODS-v2", SLOT[0], "o"),
    ("SW-UCB", "SW-UCB", SLOT[1], "s"),
    ("FSS* (best FSS per instance)", "FSS-", SLOT[2], "^"),
    ("Single* (best single solver per instance)", "Single-", SLOT[3], "D"),
    ("GP-PO", "GP-PO-v2", SLOT[4], "v"),
]


def fig_convergence(dims, stem):
    fig, axes = plt.subplots(
        len(FUNCS), len(dims), figsize=(6.9, 1.45 * len(FUNCS)), squeeze=False
    )
    handles = None
    for i, fn in enumerate(FUNCS):
        for j, d in enumerate(dims):
            ax = axes[i, j]
            hs = []
            for label, key, col, mk in SERIES:
                m = family_best(main, key, (fn, d)) if key.endswith("-") else key
                runs = main[
                    (main.method == m) & (main.function == fn) & (main.dim == d)
                ]
                C = np.maximum(np.vstack(runs.curve.values), FLOOR)
                x = 20 * np.arange(1, C.shape[1] + 1)
                med = np.median(C, axis=0)
                (h,) = ax.plot(
                    x,
                    med,
                    color=col,
                    linewidth=1.1,
                    marker=mk,
                    markersize=3,
                    markevery=(j * 2 + SERIES.index((label, key, col, mk)), 10),
                    markeredgecolor="#fcfcfb",
                    markeredgewidth=0.4,
                    label=label,
                )
                if key == "UPHODS-v2":
                    q1, q3 = np.percentile(C, [25, 75], axis=0)
                    ax.fill_between(x, q1, q3, color=col, alpha=0.15, linewidth=0)
                hs.append(h)
            handles = hs
            ax.set_yscale("log")
            tidy_log(ax)
            ax.set_xlim(0, 1000)
            ax.set_title(f"{FLABEL.get(fn, fn)}, D = {d}", color=INK, pad=3)
            if j == 0:
                ax.set_ylabel("Best error so far")
            if i == len(FUNCS) - 1:
                ax.set_xlabel("Function evaluations")
            else:
                ax.set_xticklabels([])
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
        labelcolor=INK,
    )
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, stem)


fig_convergence([10, 100], "fig_convergence_D10_D100")


# ------------------------------------------------ A12 effect-size map
def a12(x, y):
    """Probability that a run of UPHODS (x) ends with a lower error than a run
    of the baseline (y); ties count one half (Vargha-Delaney A)."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(np.mean((x < y) + 0.5 * (x == y)))


BASE = [m for m in CORE if m != "UPHODS-v2"]
A = np.zeros((len(BASE), len(DIMS)))
for i, b in enumerate(BASE):
    for j, d in enumerate(DIMS):
        vals = []
        for fn in FUNCS:
            sub = main[(main.function == fn) & (main.dim == d)]
            vals.append(
                a12(sub[sub.method == "UPHODS-v2"].best, sub[sub.method == b].best)
            )
        A[i, j] = np.mean(vals)
pd.DataFrame(A, index=[NAME[b] for b in BASE], columns=DIMS).round(3).to_csv(
    OUT / "a12_by_dim.csv"
)

cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(3.4, 2.7))
im = ax.imshow(A, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(range(len(DIMS)), [f"{d}D" for d in DIMS])
ax.set_yticks(range(len(BASE)), [NAME[b] for b in BASE])
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(DIMS)), minor=True)
ax.set_yticks(np.arange(-0.5, len(BASE)), minor=True)
ax.grid(which="minor", color="#fcfcfb", linewidth=1.5)
ax.tick_params(which="minor", length=0)
for i in range(len(BASE)):
    for j in range(len(DIMS)):
        v = A[i, j]
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=6.3,
            color="#ffffff" if abs(v - 0.5) > 0.33 else INK,
        )
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=6)
cb.set_label(
    "Mean $A_{12}$ over six functions\n(> 0.5: UPHODS better)", fontsize=6.5, color=INK2
)
ax.set_xlabel("Dimension")
save(fig, "fig_a12_by_dimension")


# ------------------------------------------------- sensitivity (9.2)
def groups(df, method):
    d = df[df.method == method]
    return {k: g.best.values for k, g in d.groupby(["function", "dim"])}


SENS_DIMS = (10, 30, 100)
dflt_df = main[(main.method == "UPHODS-v2") & (main.dim.isin(SENS_DIMS))]
dflt = groups(dflt_df, "UPHODS-v2")
DEFAULTS = {
    "kappa": 1.0,
    "eps": 0.2,
    "cooldown_evals": 10,
    "phase_half_life": 200,
    "lam_p": 0.5,
    "lam_m": 0.5,
    "inject_ratio": 0.4,
    "archive_delta": 0.05,
    "weight_window": 30,
    "slate_size": 5,
    "lengthscale": 0.5,
}
SYMBOL = {
    "kappa": r"$\kappa$ (exploration)",
    "eps": r"$\varepsilon$ (switch margin)",
    "cooldown_evals": r"$C$ (cool-down, evals)",
    "phase_half_life": r"$h$ (half-life, evals)",
    "lam_p": r"$\lambda_p$ (preference term)",
    "lam_m": r"$\lambda_m$ (metric term)",
    "inject_ratio": r"$\rho$ (elite injection)",
    "archive_delta": r"$\delta$ (archive crowding)",
    "weight_window": r"$W$ (weight window)",
    "slate_size": r"$M$ (slate size)",
    "lengthscale": r"$\ell$ (GP length-scale)",
}
rng = np.random.default_rng(20261102)
rows = []
for p in sorted((R / "sens").glob("*.jsonl")):
    key, val = p.stem.rsplit("_", 1)
    g = groups(load(p), "UPHODS-v2")
    lr = np.array(
        [
            np.log(max(np.median(g[k]), FLOOR) / max(np.median(dflt[k]), FLOOR))
            for k in g
        ]
    )
    boot = [np.mean(rng.choice(lr, len(lr))) for _ in range(4000)]
    ns = load(p).ns.mean()
    rows.append(
        (
            key,
            float(val),
            np.exp(lr.mean()),
            *np.exp(np.percentile(boot, [2.5, 97.5])),
            ns,
        )
    )
S = pd.DataFrame(rows, columns=["param", "value", "ratio", "lo", "hi", "ns"])
S.to_csv(OUT / "sensitivity.csv", index=False)
ns_default = dflt_df.ns.mean()

order = list(DEFAULTS)
ylab, yr, yl, yh, yns, ycol, yhead = [], [], [], [], [], [], []
for key in order:
    sub = S[S.param == key].sort_values("value")
    ylab.append(SYMBOL[key]), yr.append(np.nan), yl.append(np.nan), yh.append(np.nan)
    yns.append(0.0), ycol.append(None), yhead.append(True)
    for v in sorted([*sub.value, DEFAULTS[key]]):
        is_d = v == DEFAULTS[key]
        r = sub[sub.value == v]
        ylab.append(f"{v:g}" + (" (default)" if is_d else ""))
        yhead.append(False)
        if is_d:
            yr.append(1.0), yl.append(1.0), yh.append(1.0), yns.append(ns_default)
        else:
            yr.append(r.ratio.iloc[0]), yl.append(r.lo.iloc[0]), yh.append(r.hi.iloc[0])
            yns.append(r.ns.iloc[0])
        ycol.append(MUTED if is_d else SLOT[0])
ypos = np.arange(len(ylab), dtype=float)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 7.0), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.08))
a1.axvline(1.0, color=AXIS, linewidth=0.8)
for y, r, lo, hi, c, hd in zip(ypos, yr, yl, yh, ycol, yhead):
    if hd:
        continue
    a1.plot([lo, hi], [y, y], color=c, linewidth=1.0)
    a1.plot(r, y, "o", color=c, markersize=3.6, markeredgecolor="#fcfcfb", markeredgewidth=0.5)
a1.set_xscale("log")
xmax = max(2.2, float(np.nanmax(yh)) * 1.05)
a1.set_xlim(0.6, xmax)
a1.set_xticks([0.7, 1.0, 1.4, 2.0], ["0.7", "1.0", "1.4", "2.0"])
a1.minorticks_off()
a1.set_xlabel("Error ratio to the default, < 1 is better\n(geometric mean over 18 instances, 95% bootstrap CI)")
keep = [i for i, hd in enumerate(yhead) if not hd]
a2.barh(ypos[keep], np.array(yns)[keep], height=0.62,
        color=[GRAY_FILL if ycol[i] == MUTED else SLOT[0] for i in keep])
a2.set_xlabel("Mean number of switches per run")
a1.set_yticks(ypos, ylab)
for lab, hd in zip(a1.get_yticklabels(), yhead):
    if hd:
        lab.set_color(INK)
        lab.set_fontweight("bold")
a1.set_ylim(len(ylab) - 0.4, -0.6)
a1.tick_params(axis="y", length=0)
a2.tick_params(axis="y", length=0)
a1.grid(axis="y", visible=False)
a2.grid(axis="y", visible=False)
save(fig, "fig_sensitivity")


# --------------------------------------------------- GP timing (9.3)
T = json.loads((R / "gp_timing.json").read_text(encoding="utf-8"))
recs = pd.DataFrame(T["rows"]).rename(columns={"cold_median_s": "cold_median"})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
lo = main[main.method == "UPHODS-v2"].groupby("dim").n_pairs.mean()
ax.axvspan(lo.min(), lo.max(), color=GRAY_FILL, alpha=0.7, linewidth=0)
ax.text(np.sqrt(lo.min() * lo.max()), 30, "benchmark runs,\nB = 1000\n(mean per D)",
        ha="center", va="top", fontsize=5.8, color=INK2)
bud = load(R / "budget" / "uphods.jsonl")
n5k = bud[(bud.method == "UPHODS-v2") & (bud.dim == 30)].n_pairs.mean()
ax.axvline(n5k, ymax=0.3, color=MUTED, linewidth=0.7)
ax.text(n5k * 1.06, 0.0016, "30D,\nB = 5000", ha="left", va="bottom", fontsize=5.8, color=INK2)
for d, col, mk in ((10, SLOT[0], "o"), (100, SLOT[1], "s")):
    sub = recs[recs.dim == d].sort_values("n_pairs")
    ax.plot(sub.n_pairs, sub.cold_median, color=col, marker=mk, markersize=3.5, linewidth=1.1,
            markeredgecolor="#fcfcfb", markeredgewidth=0.4, label=f"D = {d}")
ref = recs[recs.dim == 10].sort_values("n_pairs")
x0, y0 = ref.n_pairs.iloc[-1], ref.cold_median.iloc[-1]
xs = np.array([500, 4000])
ax.plot(xs, y0 * (xs / x0) ** 3 / 3, color=MUTED, linewidth=0.7)
ax.text(1500, y0 * (1500 / x0) ** 3 / 3 / 2.2, r"slope $n^3$", color=INK2, fontsize=6.2, ha="left")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.0012, 40)
ax.set_xlabel("Number of pairwise comparisons $n$")
ax.set_ylabel("Time per GP fit (s)")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.4, 1.0))
save(fig, "fig_gp_timing")
