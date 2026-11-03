"""Mechanism scenarios of PROTOCOL.md section 10.

Only test problems are defined here; the method code in ``uphods/`` is frozen
(tag v2.0-frozen) and is not touched.

A. Static-penalty constraint ("penalty"), level = P:
       f_pen(x) = f(x) + P * (1 + g(x)) * [g(x) > 0],
       g(x) = RMS(u - c) - r   (u: unit coordinates, c ~ U[0.3, 0.7]^D fixed
       per instance, r = 0.1% quantile of the RMS distance of random designs).
B. Discrete design variables ("discrete"), level = K: every variable takes one
   of K equally spaced levels (cell centres), so the objective is piecewise
   constant.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from uphods.problems import FUNCTIONS, INSTANCE_SEED, Problem, make_problem

R_QUANTILE = 1e-3  # a uniformly random design is feasible with this probability


@lru_cache(maxsize=None)
def feasible_radius(name: str, dim: int) -> float:
    """0.1% quantile of the RMS distance of uniform random designs from c."""
    c = constraint_center(name, dim)
    U = np.random.default_rng([INSTANCE_SEED, 11, dim]).random((200_000, dim))
    return float(np.quantile(np.sqrt(np.mean((U - c) ** 2, axis=1)), R_QUANTILE))


def constraint_center(name: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng([INSTANCE_SEED, 10, FUNCTIONS.index(name), dim])
    return rng.uniform(0.3, 0.7, dim)


def violation(base: Problem, c: np.ndarray, r: float, X: np.ndarray) -> np.ndarray:
    U = base.to_unit(np.atleast_2d(X))
    return np.sqrt(np.mean((U - c) ** 2, axis=1)) - r


def penalty_problem(name: str, dim: int, P: float, r: float | None = None):
    base = make_problem(name, dim)
    c = constraint_center(name, dim)
    r = feasible_radius(name, dim) if r is None else r

    def func(X):
        v = violation(base, c, r, X)
        return base.raw(X) + np.where(v > 0, P * (1.0 + v), 0.0)

    prob = Problem(name, dim, base.lower, base.upper, func, 0.0, None)
    return prob, (lambda X: violation(base, c, r, X)), base


def quantize_unit(U: np.ndarray, K: int) -> np.ndarray:
    return (np.clip(np.floor(U * K), 0, K - 1) + 0.5) / K


def discrete_problem(name: str, dim: int, K: int):
    base = make_problem(name, dim)

    def func(X):
        Uq = quantize_unit(base.to_unit(np.atleast_2d(X)), K)
        return base.raw(base.from_unit(Uq))

    return Problem(name, dim, base.lower, base.upper, func, 0.0, None), base


def make_scenario(scenario: str, name: str, dim: int, level: float):
    """Returns (problem, extra) where ``extra(best_x_unit)`` gives scenario
    specific fields for the result record."""
    if scenario == "penalty":
        prob, g, base = penalty_problem(name, dim, float(level))

        def extra(u):
            x = prob.from_unit(np.atleast_2d(u))
            return dict(feasible=bool(g(x)[0] <= 0), best_obj=float(base.raw(x)[0]))

        return prob, extra
    if scenario == "discrete":
        prob, base = discrete_problem(name, dim, int(level))
        return prob, (lambda u: {})
    raise ValueError(scenario)
