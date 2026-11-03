"""How large is the metric correction term in the solver score? (diagnostic)

For a few runs, decompose every phase-regression observation of UPHODS-v2 into
the log-progress part log(m1 + floor) and the correction lam_p*p + lam_m*(w'm - 0.5),
and report their spreads; with robust normalisation switched off (-noRN) the
metric part is unbounded.  Uses the frozen method code unchanged.

    uv run python experiments/diag_metric_term.py            # development seeds 1000-1004
    uv run python experiments/diag_metric_term.py 0 30       # final seeds 0-29 (reported)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from scenarios import make_scenario  # noqa: E402

from uphods.problems import make_problem  # noqa: E402
from uphods.runner import METHODS  # noqa: E402


def decompose(algo):
    obs = algo.phase_obs
    pos = [o[2] for o in obs if o[2] > 0]
    floor = algo.cfg.phase_floor * (float(np.median(pos)) if pos else 1.0)
    logp = np.array([np.log(o[2] + floor) for o in obs])
    extra = np.array([o[4] for o in obs])
    return logp, extra


def run(method, prob, seed):
    a = METHODS[method](prob, 1000, seed)
    a.run()
    return decompose(a)


cases = [
    ("benchmark Rastrigin 30D", lambda: make_problem("Rastrigin", 30)),
    (
        "penalty 1e6 Rastrigin 30D",
        lambda: make_scenario("penalty", "Rastrigin", 30, 1e6)[0],
    ),
    (
        "discrete K=10 Rosenbrock 30D",
        lambda: make_scenario("discrete", "Rosenbrock", 30, 10)[0],
    ),
]
lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1000, 1005)
print(
    f"Seeds {lo}-{hi - 1}; sd = mean over runs of the within-run sd; max = largest over runs."
)
print()
print(
    "| case | method | sd of log-progress part | sd of correction | max |correction| |"
)
print("|---|---|---|---|---|")
for name, mk in cases:
    for m in ["UPHODS-v2", "UPHODS-v2-noRN"]:
        L, E = [], []
        for s in range(lo, hi):
            lp, ex = run(m, mk(), s)
            L.append(np.std(lp))
            E.append((np.std(ex), np.max(np.abs(ex))))
        E = np.array(E)
        print(
            f"| {name} | {m} | {np.mean(L):.2f} | {E[:, 0].mean():.2f} | {E[:, 1].max():.2f} |"
        )
