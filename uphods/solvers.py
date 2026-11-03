"""Constituent solvers: GA, PSO, SA and GD.

All solvers work in unit coordinates u in [0, 1]^D and share one interface:

* ``start(U, f, ev)``  initialise (or warm-start) from designs ``U`` with
  objective values ``f`` (NaN = not yet evaluated); missing designs are
  filled randomly and evaluated.
* ``step(ev)``         run one native iteration (one *epoch*) and return the
  evaluated batch ``(U, f)``.
* ``propose(rng)``     designs the solver would evaluate next, generated
  without spending evaluations (used for look-ahead scoring).
* ``population()``     the current working set ``(U, f)`` used for elite
  transfer and diversity.

Evaluations always go through the :class:`~uphods.problems.Evaluator`, so the
budget accounting is the same for every method.
"""

from __future__ import annotations

import copy

import numpy as np

from .problems import Evaluator


def _eval(ev: Evaluator, U: np.ndarray):
    """Evaluate as many rows as the budget allows; return (U_eval, f)."""
    if ev.remaining <= 0 or len(U) == 0:
        return U[:0], np.empty(0)
    f = ev(U)
    return U[: len(f)], f


def _fill(U, f, n, dim, rng, ev):
    """Complete a start population to n designs and evaluate missing values."""
    if U is None or len(U) == 0:
        U = np.empty((0, dim))
        f = np.empty(0)
    U = np.array(U, dtype=float, copy=True)
    f = np.array(f, dtype=float, copy=True)
    if len(U) < n:
        extra = rng.random((n - len(U), dim))
        U = np.vstack([U, extra])
        f = np.concatenate([f, np.full(len(extra), np.nan)])
    U, f = U[:n], f[:n]
    miss = np.where(np.isnan(f))[0]
    if len(miss):
        Ue, fe = _eval(ev, U[miss])
        f[miss[: len(fe)]] = fe
        keep = ~np.isnan(f)
        U, f = U[keep], f[keep]
    return U, f


class Solver:
    name = "base"
    single_point = False

    def __init__(self, dim: int, rng: np.random.Generator, pop_size: int = 20):
        self.dim = dim
        self.rng = rng
        self.N = pop_size
        self.started = False

    def clone(self):
        c = copy.copy(self)
        c.__dict__ = {
            k: (v.copy() if isinstance(v, np.ndarray) else copy.copy(v))
            for k, v in self.__dict__.items()
        }
        c.rng = self.rng  # never used by propose(); proposals take their own rng
        return c


# ---------------------------------------------------------------------------
# Real-coded genetic algorithm (binary tournament, SBX, polynomial mutation,
# generational replacement with one elite)
# ---------------------------------------------------------------------------


def sbx(P1, P2, rng, eta=15.0, pc=0.9):
    n, d = P1.shape
    u = rng.random((n, d))
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 * (1.0 - u) + 1e-300)) ** (1.0 / (eta + 1.0)),
    )
    mask = (rng.random((n, d)) < 0.5) & (rng.random(n) < pc)[:, None]
    C1 = np.where(mask, 0.5 * ((1 + beta) * P1 + (1 - beta) * P2), P1)
    C2 = np.where(mask, 0.5 * ((1 - beta) * P1 + (1 + beta) * P2), P2)
    return np.clip(C1, 0, 1), np.clip(C2, 0, 1)


def poly_mutation(X, rng, eta=20.0, pm=None):
    pm = 1.0 / X.shape[1] if pm is None else pm
    u = rng.random(X.shape)
    delta = np.where(
        u < 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
        1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)),
    )
    mask = rng.random(X.shape) < pm
    return np.clip(X + mask * delta, 0, 1)


class GA(Solver):
    name = "GA"

    def start(self, U=None, f=None, ev=None):
        self.U, self.f = _fill(U, f, self.N, self.dim, self.rng, ev)
        self.started = True

    def _offspring(self, rng):
        n = len(self.U)
        a, b = rng.integers(0, n, (2, self.N))
        par = np.where(self.f[a] <= self.f[b], a, b)
        P = self.U[par]
        half = (self.N + 1) // 2
        C1, C2 = sbx(
            P[:half], P[half : 2 * half] if 2 * half <= len(P) else P[:half], rng
        )
        C = np.vstack([C1, C2])[: self.N - 1]
        return poly_mutation(C, rng)

    def propose(self, rng):
        return self._offspring(rng)

    def step(self, ev):
        C = self._offspring(self.rng)
        Ce, fe = _eval(ev, C)
        ib = int(np.argmin(self.f))
        # generational replacement with one elite; if the budget truncated the
        # offspring batch, the best remaining parents fill the population
        order = np.argsort(self.f)
        keep = [ib] + [i for i in order if i != ib][: self.N - 1 - len(fe)]
        self.U = np.vstack([self.U[keep], Ce])
        self.f = np.concatenate([self.f[keep], fe])
        return Ce, fe

    def population(self):
        return self.U, self.f


# ---------------------------------------------------------------------------
# Particle swarm optimisation (global-best topology, constriction weights)
# ---------------------------------------------------------------------------


class PSO(Solver):
    name = "PSO"
    w, c1, c2, vmax = 0.7298, 1.49618, 1.49618, 0.2

    def start(self, U=None, f=None, ev=None):
        U, f = _fill(U, f, self.N, self.dim, self.rng, ev)
        self.X, self.fx = U.copy(), f.copy()
        self.V = np.zeros_like(U)
        self.P, self.fp = U.copy(), f.copy()
        self.started = True

    def _move(self, rng):
        g = self.P[int(np.argmin(self.fp))]
        r1, r2 = rng.random(self.X.shape), rng.random(self.X.shape)
        V = (
            self.w * self.V
            + self.c1 * r1 * (self.P - self.X)
            + self.c2 * r2 * (g - self.X)
        )
        V = np.clip(V, -self.vmax, self.vmax)
        X = self.X + V
        out = (X < 0) | (X > 1)
        V[out] = 0.0
        return np.clip(X, 0, 1), V

    def propose(self, rng):
        return self._move(rng)[0]

    def step(self, ev):
        X, V = self._move(self.rng)
        Xe, fe = _eval(ev, X)
        k = len(fe)
        self.X[:k], self.V[:k], self.fx[:k] = Xe, V[:k], fe
        better = fe < self.fp[:k]
        self.P[:k][better] = Xe[better]
        self.fp[:k][better] = fe[better]
        return Xe, fe

    def population(self):
        return self.P, self.fp


# ---------------------------------------------------------------------------
# Simulated annealing (single chain, Metropolis acceptance, geometric cooling,
# step size adapted to the acceptance rate); one epoch = N proposals
# ---------------------------------------------------------------------------


class SA(Solver):
    name = "SA"
    single_point = True
    cool = 0.995  # per proposal
    s_init = 0.1  # Gaussian step (unit coordinates)

    def start(self, U=None, f=None, ev=None):
        U, f = _fill(U, f, 1 if U is None else max(1, len(U)), self.dim, self.rng, ev)
        i = int(np.argmin(f))
        self.x, self.fx = U[i].copy(), float(f[i])
        spread = float(np.median(f) - np.min(f)) if len(f) > 1 else 0.0
        self.T = 0.1 * (spread if spread > 0 else abs(self.fx)) + 1e-12
        if not hasattr(self, "s"):
            self.s = self.s_init
        self.buf_U, self.buf_f = U[[i]].copy(), f[[i]].copy()
        self.started = True

    def propose(self, rng):
        return np.clip(self.x + self.s * rng.standard_normal((self.N, self.dim)), 0, 1)

    def step(self, ev):
        Us, fs, acc = [], [], 0
        for _ in range(self.N):
            if ev.remaining <= 0:
                break
            y = np.clip(self.x + self.s * self.rng.standard_normal(self.dim), 0, 1)
            fy = float(ev(y[None])[0])
            Us.append(y)
            fs.append(fy)
            d = fy - self.fx
            if d < 0 or self.rng.random() < np.exp(-d / max(self.T, 1e-300)):
                self.x, self.fx = y, fy
                acc += 1
            self.T *= self.cool
        n = max(len(fs), 1)
        rate = acc / n
        if rate > 0.3:
            self.s = min(self.s * 1.5, 0.5)
        elif rate < 0.1:
            self.s = max(self.s * 0.6, 1e-6)
        Ue, fe = np.array(Us).reshape(-1, self.dim), np.array(fs)
        self.buf_U, self.buf_f = Ue, fe
        return Ue, fe

    def population(self):
        U = np.vstack([self.x[None], self.buf_U])
        f = np.concatenate([[self.fx], self.buf_f])
        return U, f


# ---------------------------------------------------------------------------
# Gradient descent with forward-difference gradients and a backtracking step
# on the normalised direction; one epoch = one gradient step (D + <=4 evals)
# ---------------------------------------------------------------------------


class GD(Solver):
    name = "GD"
    single_point = True
    h = 1e-6
    max_backtrack = 4

    def start(self, U=None, f=None, ev=None):
        U, f = _fill(U, f, 1 if U is None else max(1, len(U)), self.dim, self.rng, ev)
        i = int(np.argmin(f))
        self.x, self.fx = U[i].copy(), float(f[i])
        if not hasattr(self, "alpha"):
            self.alpha = 0.1 * np.sqrt(self.dim)
        self.d_last = None
        self.buf_U, self.buf_f = U[[i]].copy(), f[[i]].copy()
        self.started = True

    def propose(self, rng):
        if self.d_last is None:
            return self.x[None].copy()
        return np.clip(self.x + self.alpha * self.d_last, 0, 1)[None]

    def step(self, ev):
        D = self.dim
        Us, fs = [], []
        if ev.remaining <= 0:
            return np.empty((0, D)), np.empty(0)
        if ev.remaining == 1:
            # not enough budget for a gradient: one last trial step along the
            # previous direction (or a random direction)
            d = (
                self.d_last
                if self.d_last is not None
                else self.rng.standard_normal(D) / np.sqrt(D)
            )
            y = np.clip(self.x + self.alpha * d, 0, 1)
            fy = float(ev(y[None])[0])
            if fy < self.fx:
                self.x, self.fx = y, fy
            self.buf_U, self.buf_f = y[None], np.array([fy])
            return y[None], np.array([fy])
        # forward differences on as many coordinates as the budget allows
        k = min(D, ev.remaining - 1)
        coords = np.arange(D) if k == D else self.rng.choice(D, k, replace=False)
        sgn = np.where(self.x[coords] + self.h <= 1.0, 1.0, -1.0)
        S = np.repeat(self.x[None], k, axis=0)
        S[np.arange(k), coords] += sgn * self.h
        Se, fe = _eval(ev, S)
        Us.append(Se)
        fs.append(fe)
        g = np.zeros(D)
        g[coords[: len(fe)]] = (fe - self.fx) / (sgn[: len(fe)] * self.h)
        gn = np.linalg.norm(g)
        d = -g / gn if gn > 0 else self.rng.standard_normal(D) / np.sqrt(D)
        for _ in range(self.max_backtrack):
            if ev.remaining <= 0:
                break
            y = np.clip(self.x + self.alpha * d, 0, 1)
            fy = float(ev(y[None])[0])
            Us.append(y[None])
            fs.append(np.array([fy]))
            if fy < self.fx:
                self.x, self.fx = y, fy
                self.alpha = min(2.0 * self.alpha, 0.5 * np.sqrt(D))
                self.d_last = d
                break
            self.alpha = max(0.5 * self.alpha, 1e-12)
        Ue, fe = np.vstack(Us), np.concatenate(fs)
        self.buf_U, self.buf_f = Ue, fe
        return Ue, fe

    def population(self):
        return np.vstack([self.x[None], self.buf_U[-self.N :]]), np.concatenate(
            [[self.fx], self.buf_f[-self.N :]]
        )


SOLVERS = {"GA": GA, "PSO": PSO, "SA": SA, "GD": GD}
ORDER = ["GA", "PSO", "GD", "SA"]  # order used by the fixed schedules (0201 code)


def make_library(dim, rng, pop_size=20):
    # each solver gets its own random stream so that adding/removing one does
    # not change the others' behaviour
    seeds = rng.integers(0, 2**63 - 1, len(ORDER))
    return {
        n: SOLVERS[n](dim, np.random.default_rng(s), pop_size)
        for n, s in zip(ORDER, seeds)
    }
