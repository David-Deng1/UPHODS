"""Objective performance metrics (Sec. 3.1.3) and online robust normalisation
(Sec. 3.1.2, Eqs. 2-4)."""

from __future__ import annotations

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Raw metrics
# ---------------------------------------------------------------------------


def convergence_speed(best_before: float, best_after: float, n_evals: int) -> float:
    """m1, revised after R2-5: windowed relative improvement per 100 evaluations.

    ``best_before`` is the best-so-far value at the start of the window and
    ``best_after`` at its end; ``n_evals`` is the number of evaluations spent in
    the window.  Dividing by the window cost (not by the absolute iteration
    index t, as in Eq. 5 of the 0201 version) keeps the metric comparable
    between early and late phases and between cheap and expensive solvers.
    """
    if n_evals <= 0:
        return 0.0
    rel = (best_before - best_after) / (abs(best_before) + 1e-12)
    return max(rel, 0.0) * 100.0 / n_evals


def diversity(U: np.ndarray) -> float:
    """m2 (Eq. 6): mean pairwise Euclidean distance in unit coordinates,
    divided by sqrt(D) so that the value lies in [0, 1]."""
    n, d = U.shape
    if n < 2:
        return 0.0
    sq = np.sum(U * U, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * U @ U.T, 0.0)
    iu = np.triu_indices(n, 1)
    return float(np.mean(np.sqrt(D2[iu])) / np.sqrt(d))


def robustness(utility_traj) -> float:
    """m3_rob = 1 / (1 + CV) of the recent population-utility trajectory (Eq. 7).

    Utilities are taken on the probability scale (Phi of the latent mean), so
    they are positive and the coefficient of variation is well defined.
    """
    u = np.asarray(utility_traj, dtype=float)
    if len(u) < 2:
        return 0.5
    cv = np.std(u) / (abs(np.mean(u)) + 1e-12)
    return float(1.0 / (1.0 + cv))


# ---------------------------------------------------------------------------
# Online robust normalisation
# ---------------------------------------------------------------------------


class RobustNormalizer:
    """Exponentially weighted median/MAD standardisation with a Hampel
    pre-filter, mapped to [0, 1] by tanh (Eqs. 2-4).

    Parameters follow Table 2 of the revision: lambda (EW rate), W (window),
    tau_H (Hampel threshold) and zeta (tanh scale).
    """

    def __init__(self, lam=0.2, window=10, hampel_tau=3.0, zeta=2.0, enabled=True):
        self.lam, self.tau, self.zeta = lam, hampel_tau, zeta
        self.hist = deque(maxlen=window)
        self.m = None
        self.s = None
        self.enabled = enabled

    @staticmethod
    def _med_mad(a):
        med = float(np.median(a))
        return med, 1.4826 * float(np.median(np.abs(a - med)))

    def update(self, x: float) -> None:
        x_upd = x
        if len(self.hist) >= 3:
            mh, sh = self._med_mad(np.asarray(self.hist))
            if abs(x - mh) / (sh + self._eps(mh)) > self.tau:
                x_upd = mh  # spike: do not let it move the location/scale
        self.hist.append(x_upd)
        med, mad = self._med_mad(np.asarray(self.hist))
        if self.m is None:
            self.m, self.s = med, mad
        else:
            self.m = (1 - self.lam) * self.m + self.lam * med
            self.s = (1 - self.lam) * self.s + self.lam * mad

    @staticmethod
    def _eps(m):
        return 1e-3 * abs(m) + 1e-12

    def transform(self, x: float) -> float:
        if not self.enabled:
            return float(x)
        if self.m is None or len(self.hist) < 3:
            return 0.5
        z = (x - self.m) / (self.s + self._eps(self.m))
        return float(0.5 * (1.0 + np.tanh(z / self.zeta)))
