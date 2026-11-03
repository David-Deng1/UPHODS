"""Simulated decision maker for the benchmark suite.

The hidden utility is u*(x) = -log(1 + f(x) - f*), i.e. the DM prefers lower
objective error and judges differences on a relative scale.  Rankings are
Thurstonian: u* plus Gaussian noise (sigma), so a pair is ordered correctly
with probability Phi(delta u* / (sqrt(2) sigma)), consistent with the probit
likelihood of Eq. (1).  For the preference-noise study a fraction ``flip`` of
the resulting pairs is additionally reversed at random.
"""

from __future__ import annotations

import numpy as np


class SimulatedDM:
    def __init__(
        self,
        rng: np.random.Generator,
        sigma: float = 0.05,
        flip: float = 0.0,
        utility=None,
    ):
        self.rng, self.sigma, self.flip = rng, sigma, flip
        # optional hidden utility u(f, U) (misaligned scenario); default -log(1 + f)
        self.utility = utility
        self.n_queries = 0
        self.n_pairs = 0

    def rank(self, f_err: np.ndarray, U=None) -> np.ndarray:
        """Indices of the slate from most to least preferred."""
        if self.utility is not None:
            u = np.asarray(self.utility(f_err, U), dtype=float)
        else:
            u = -np.log1p(np.maximum(np.asarray(f_err, dtype=float), 0.0))
        u = u + self.sigma * self.rng.standard_normal(len(u))
        return np.argsort(-u, kind="stable")

    def pairs(self, ids, f_err, U=None):
        """All ordered pairs (winner_id, loser_id) implied by one ranking."""
        ids = np.asarray(ids)
        order = ids[self.rank(f_err, U)]
        out = []
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                w, l = order[a], order[b]
                if self.flip > 0 and self.rng.random() < self.flip:
                    w, l = l, w
                out.append((int(w), int(l)))
        self.n_queries += 1
        self.n_pairs += len(out)
        return out
