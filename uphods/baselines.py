"""Baselines: single solvers, fixed switching strategies FSS-1..4 and a
sliding-window UCB adaptive algorithm selector (SW-UCB).

All hybrid baselines inherit the solver library, the elite archive and the
elite transfer of :class:`~uphods.controller.HybridBase`; they differ from
UPHODS only in the switching decision.  Without a preference model, designs
are ranked for the archive by their objective value.
"""

from __future__ import annotations

import numpy as np

from .controller import HybridBase
from .metrics import convergence_speed
from .solvers import ORDER


class Single(HybridBase):
    def __init__(self, problem, budget, seed, solver="GA", **kw):
        super().__init__(problem, budget, seed, initial=solver, **kw)
        self.method = f"Single-{solver}"


def _next(name):
    return ORDER[(ORDER.index(name) + 1) % len(ORDER)]


class FSS1(HybridBase):
    """Fixed iteration: switch to the next solver every ``period`` epochs."""

    method = "FSS-1"

    def __init__(self, problem, budget, seed, period=10, **kw):
        super().__init__(problem, budget, seed, **kw)
        self.period = period

    def decide(self):
        return _next(self.active) if len(self.tenure()) >= self.period else None


class FSS2(HybridBase):
    """Threshold: switch when the relative improvement over the last
    ``window`` epochs of the tenure falls below ``theta``."""

    method = "FSS-2"

    def __init__(self, problem, budget, seed, theta=1e-3, window=3, **kw):
        super().__init__(problem, budget, seed, **kw)
        self.theta, self.window = theta, window

    def decide(self):
        ten = self.tenure()
        if len(ten) < self.window:
            return None
        a, b = ten[-self.window]["best_before"], ten[-1]["best_after"]
        rel = (a - b) / (abs(a) + 1e-12)
        return _next(self.active) if rel < self.theta else None


class FSS3(HybridBase):
    """Cyclic: switch to the next solver every ``cycle`` function evaluations."""

    method = "FSS-3"

    def __init__(self, problem, budget, seed, cycle=150, **kw):
        super().__init__(problem, budget, seed, **kw)
        self.cycle = cycle

    def decide(self):
        start = self.tenure()[0]["evals_before"] if self.tenure() else 0
        return _next(self.active) if self.ev.n_evals - start >= self.cycle else None


class FSS4(HybridBase):
    """Stage-based: GA -> PSO -> GD -> SA at 20 %, 50 % and 80 % of the budget
    (global search first, local refinement last)."""

    method = "FSS-4"

    def __init__(self, problem, budget, seed, stages=(0.2, 0.5, 0.8), **kw):
        super().__init__(problem, budget, seed, **kw)
        self.stages = stages

    def decide(self):
        k = int(
            np.searchsorted(
                np.array(self.stages) * self.ev.budget, self.ev.n_evals, "right"
            )
        )
        return ORDER[min(k, len(ORDER) - 1)]


class SWUCB(HybridBase):
    """Sliding-window UCB over the solver library (adaptive operator
    selection with a non-stationary bandit, Garivier & Moulines 2011).
    Reward of an epoch: relative improvement per 100 evaluations, scaled by
    the largest reward seen so far."""

    method = "SW-UCB"

    def __init__(self, problem, budget, seed, window=20, c=0.5, **kw):
        super().__init__(problem, budget, seed, **kw)
        self.window, self.c = window, c
        self.hist = []  # (solver, raw reward)
        self.rmax = 1e-12

    def after_epoch(self, i0, i1, U_batch):
        e = self.epochs[-1]
        r = convergence_speed(
            e["best_before"], e["best_after"], e["evals_after"] - e["evals_before"]
        )
        self.rmax = max(self.rmax, r)
        self.hist.append((self.active, r))

    def decide(self):
        h = self.hist[-self.window :]
        t = len(h)
        best, best_v = None, -np.inf
        for name in ORDER:
            rs = [r / self.rmax for s, r in h if s == name]
            if not rs:
                return name  # untried within the window: explore it
            v = np.mean(rs) + self.c * np.sqrt(2.0 * np.log(max(t, 2)) / len(rs))
            if v > best_v:
                best, best_v = name, v
        return best
