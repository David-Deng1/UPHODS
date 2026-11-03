"""Benchmark problems and the evaluation counter.

Every call of the objective goes through :class:`Evaluator`, which enforces the
evaluation budget and records the best-so-far trace.  Solvers never call the
objective directly, so the budget accounting is identical for all methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------
# Vectorised test functions: X has shape (n, D), return shape (n,)
# --------------------------------------------------------------------------


def ackley(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    s1 = np.sqrt(np.sum(X * X, axis=1) / d)
    s2 = np.sum(np.cos(2.0 * np.pi * X), axis=1) / d
    return -20.0 * np.exp(-0.2 * s1) - np.exp(s2) + 20.0 + np.e


def griewank(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    i = np.sqrt(np.arange(1, d + 1, dtype=float))
    return np.sum(X * X, axis=1) / 4000.0 - np.prod(np.cos(X / i), axis=1) + 1.0


def rastrigin(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    return 10.0 * d + np.sum(X * X - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    a = X[:, 1:] - X[:, :-1] ** 2
    b = X[:, :-1] - 1.0
    return np.sum(100.0 * a * a + b * b, axis=1)


def sphere(X: np.ndarray) -> np.ndarray:
    return np.sum(X * X, axis=1)


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    lower: np.ndarray
    upper: np.ndarray
    func: Callable[[np.ndarray], np.ndarray]
    f_opt: float = 0.0
    shift: np.ndarray | None = None

    def raw(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.shift is not None:
            X = X - self.shift
        return self.func(X) - self.f_opt

    # normalised <-> raw coordinates (all solvers work in [0, 1]^D internally)
    def to_unit(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lower) / (self.upper - self.lower)

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        return self.lower + np.clip(U, 0.0, 1.0) * (self.upper - self.lower)


_BASE = {
    # name: (function, lower, upper, shifted, shift_bound)
    "Ackley": (ackley, -32.768, 32.768, False, 0.0),
    "Griewank": (griewank, -600.0, 600.0, False, 0.0),
    "Rastrigin": (rastrigin, -5.12, 5.12, False, 0.0),
    "Rosenbrock": (rosenbrock, -5.0, 10.0, False, 0.0),
    "ShiftedSphere": (sphere, -100.0, 100.0, True, 80.0),
    "ShiftedRastrigin": (rastrigin, -5.0, 5.0, True, 4.0),
}

FUNCTIONS = list(_BASE)
DIMS = [5, 10, 20, 30, 50, 100]
INSTANCE_SEED = 20261101  # fixes the shift vectors; never changed after registration


def make_problem(name: str, dim: int) -> Problem:
    func, lo, hi, shifted, sb = _BASE[name]
    shift = None
    if shifted:
        # one shift vector per (function, dimension), identical for all runs
        key = FUNCTIONS.index(name) * 1000 + dim
        rng = np.random.default_rng([INSTANCE_SEED, key])
        shift = rng.uniform(-sb, sb, dim)
    return Problem(name, dim, np.full(dim, lo), np.full(dim, hi), func, 0.0, shift)


# --------------------------------------------------------------------------
# Evaluation counter
# --------------------------------------------------------------------------


class BudgetExhausted(Exception):
    pass


@dataclass
class Evaluator:
    problem: Problem
    budget: int
    n_evals: int = 0
    best_f: float = np.inf
    best_x: np.ndarray | None = None
    trace_evals: list = field(default_factory=list)
    trace_best: list = field(default_factory=list)
    U_log: list = field(default_factory=list)  # every evaluated design (unit coords)
    f_log: list = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.budget - self.n_evals

    def __call__(self, U: np.ndarray) -> np.ndarray:
        """Evaluate designs given in unit coordinates.

        If the batch is larger than the remaining budget, only the first
        ``remaining`` rows are evaluated and the shorter array is returned.
        """
        U = np.atleast_2d(U)
        k = min(len(U), self.remaining)
        if k <= 0:
            raise BudgetExhausted
        f = self.problem.raw(self.problem.from_unit(U[:k]))
        for i in range(k):
            self.n_evals += 1
            if f[i] < self.best_f:
                self.best_f = float(f[i])
                self.best_x = U[i].copy()
            self.trace_evals.append(self.n_evals)
            self.trace_best.append(self.best_f)
            self.U_log.append(U[i].copy())
            self.f_log.append(float(f[i]))
        return f

    def curve(self, step: int = 20) -> np.ndarray:
        """Best-so-far at evaluations step, 2*step, ..., budget."""
        best = np.asarray(self.trace_best)
        idx = np.arange(step, self.budget + 1, step) - 1
        idx = idx[idx < len(best)]
        return best[idx]
