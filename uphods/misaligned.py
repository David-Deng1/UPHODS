"""Preference-misaligned benchmark scenario (PROTOCOL.md section 8).

The simulated decision maker values the objective and, additionally, a
design property h(x) that the objective does not contain: closeness to a
reference design u_ref (e.g. an existing baseline that is easy to
manufacture).  Hidden utility

    u*(x) = -log(1 + f(x) - f*) - lam_h * ||u - u_ref||^2 / D.
"""

from __future__ import annotations

import numpy as np

from .problems import FUNCTIONS, INSTANCE_SEED, Problem


def u_ref(problem: Problem) -> np.ndarray:
    key = 7_000_000 + FUNCTIONS.index(problem.name) * 1000 + problem.dim
    return np.random.default_rng([INSTANCE_SEED, key]).uniform(0.1, 0.9, problem.dim)


def h(U: np.ndarray, problem: Problem) -> np.ndarray:
    U = np.atleast_2d(U)
    return np.mean((U - u_ref(problem)) ** 2, axis=1)


def dm_utility(f_err, U, problem: Problem, lam_h: float) -> np.ndarray:
    u = -np.log1p(np.maximum(np.asarray(f_err, dtype=float), 0.0))
    if lam_h > 0:
        u = u - lam_h * h(U, problem)
    return u
