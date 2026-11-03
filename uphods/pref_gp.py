"""Hybrid preference model (Sec. 3.1.4): latent utility

    u(x) = w^T m(x) + r(x),   r ~ GP(0, k),

with a probit (Thurstone / Bradley-Terry-probit) likelihood for pairwise
comparisons (Eq. 1) and a Gaussian objective channel N(u_i | w^T m_i, tau^2)
(Eq. 9).  Because u_i - w^T m_i = r_i, the objective channel is a Gaussian
factor on the residual at every compared design.  The probit factors depend
on the residual through the projection s_k = r_w - r_l plus the offset
o_k = w^T (m_w - m_l); they are approximated with Expectation Propagation.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _ratio(z):
    """phi(z) / Phi(z), numerically stable."""
    return np.exp(-0.5 * z * z - _LOG_SQRT_2PI - log_ndtr(z))


class PreferenceGP:
    def __init__(
        self,
        dim,
        lengthscale=0.5,
        amplitude=1.0,
        noise=1.0 / np.sqrt(2.0),
        tau=1.0,
        jitter=1e-6,
        damping=0.5,
        max_sweeps=50,
        tol=1e-5,
    ):
        # inputs are unit coordinates; distances are divided by sqrt(D) so that
        # one length-scale value is meaningful in every dimension
        self.ls = lengthscale * np.sqrt(dim)
        self.amp2 = amplitude**2
        self.c2 = 2.0 * noise**2  # Phi((u_i - u_j) / (sqrt(2) sigma))
        self.tau2 = tau**2
        self.jitter, self.damping = jitter, damping
        self.max_sweeps, self.tol = max_sweeps, tol
        self.X = None
        self.alpha = None
        self.L = None
        self.fit_seconds = []
        self._sites = {}  # pair key -> (tau_tilde, nu_tilde), warm start between fits

    def kernel(self, A, B):
        sa = np.sum(A * A, axis=1)[:, None]
        sb = np.sum(B * B, axis=1)[None, :]
        d2 = np.maximum(sa + sb - 2.0 * A @ B.T, 0.0)
        return self.amp2 * np.exp(-0.5 * d2 / self.ls**2)

    def fit(self, X, pairs, offsets=None, pair_keys=None):
        """X: (n, D) compared designs; pairs: (m, 2) int (winner, loser) row
        indices into X; offsets: (m,) metric-channel offsets w^T(m_w - m_l)."""
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        n, m = len(X), len(pairs)
        o = np.zeros(m) if offsets is None else np.asarray(offsets, dtype=float)
        K = self.kernel(X, X) + self.jitter * np.eye(n)
        wi, li = pairs[:, 0], pairs[:, 1]
        keys = pair_keys if pair_keys is not None else [None] * m
        tt = np.array([self._sites.get(k, (0.0, 0.0))[0] for k in keys])
        nt = np.array([self._sites.get(k, (0.0, 0.0))[1] for k in keys])

        def posterior(tt, nt):
            Lam = np.eye(n) / self.tau2
            np.add.at(Lam, (wi, wi), tt)
            np.add.at(Lam, (li, li), tt)
            np.add.at(Lam, (wi, li), -tt)
            np.add.at(Lam, (li, wi), -tt)
            eta = np.zeros(n)
            np.add.at(eta, wi, nt)
            np.add.at(eta, li, -nt)
            Lc = cho_factor(Lam)
            Lam_inv = cho_solve(Lc, np.eye(n))
            B = K + Lam_inv
            Lb = cho_factor(B)
            alpha = cho_solve(Lb, Lam_inv @ eta)  # = K^-1 mu
            mu = K @ alpha
            Sigma = K - K @ cho_solve(Lb, K)
            return mu, Sigma, alpha, Lb

        mu, Sigma, alpha, Lb = posterior(tt, nt)
        for _ in range(self.max_sweeps if m else 0):
            sm = mu[wi] - mu[li]
            sv = Sigma[wi, wi] + Sigma[li, li] - 2.0 * Sigma[wi, li]
            tc = 1.0 / sv - tt
            nc = sm / sv - nt
            ok = tc > 1e-10
            vc = np.where(ok, 1.0 / np.where(ok, tc, 1.0), 1.0)
            mc = nc * vc
            den = np.sqrt(self.c2 + vc)
            z = (mc + o) / den
            r = _ratio(z)
            mh = mc + vc * r / den
            vh = vc - vc**2 * r * (z + r) / (self.c2 + vc)
            vh = np.maximum(vh, 1e-12)
            tt_new = np.maximum(1.0 / vh - tc, 1e-10)
            nt_new = mh / vh - nc
            tt_new = np.where(ok, (1 - self.damping) * tt + self.damping * tt_new, tt)
            nt_new = np.where(ok, (1 - self.damping) * nt + self.damping * nt_new, nt)
            delta = np.max(np.abs(tt_new - tt)) + np.max(np.abs(nt_new - nt))
            tt, nt = tt_new, nt_new
            mu, Sigma, alpha, Lb = posterior(tt, nt)
            if delta < self.tol:
                break
        for k, a, b in zip(keys, tt, nt):
            if k is not None:
                self._sites[k] = (a, b)
        self.X, self.alpha, self.L = X, alpha, Lb
        self.mu_train = mu
        self.fit_seconds.append(time.perf_counter() - t0)
        return self

    def predict(self, Xs):
        """Posterior mean and standard deviation of the residual r at Xs."""
        Xs = np.atleast_2d(Xs)
        if self.X is None:
            return np.zeros(len(Xs)), np.full(len(Xs), np.sqrt(self.amp2))
        Ks = self.kernel(Xs, self.X)
        mean = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        var = np.maximum(self.amp2 - np.sum(Ks * v.T, axis=1), 1e-12)
        return mean, np.sqrt(var)


class SimplexWeights:
    """Metric weights on the probability simplex (Sec. 3.2.1, Eqs. 10-12).

    Loss: mean squared residual between the posterior utility means and the
    metric projection, plus a symmetric Dirichlet(alpha) prior.  The Euclidean
    gradient is preconditioned by the inverse Fisher information of the
    simplex (natural gradient) and applied as a mirror-descent (softmax) step
    with step size eta_t = eta0 / (1 + gamma t).
    """

    def __init__(
        self, k=3, eta0=0.05, gamma=1e-4, alpha=1.2, sigma_o=1.0, adaptive=True
    ):
        self.w = np.full(k, 1.0 / k)
        self.eta0, self.gamma, self.alpha, self.so2 = eta0, gamma, alpha, sigma_o**2
        self.t = 0
        self.adaptive = adaptive
        self.history = [self.w.copy()]

    def step(self, u_hat, M):
        if not self.adaptive or len(u_hat) == 0:
            return self.w
        n = len(u_hat)
        resid = u_hat - M @ self.w
        g = -(M.T @ resid) / (self.so2 * n)  # Eq. 11 (averaged)
        g -= (self.alpha - 1.0) / (n * self.w)  # Dirichlet prior
        g_nat = self.w * (g - self.w @ g)  # inverse-Fisher preconditioning
        eta = self.eta0 / (1.0 + self.gamma * self.t)
        logits = np.log(self.w) - eta * g_nat
        logits -= logits.max()
        w = np.exp(logits)
        self.w = w / w.sum()
        self.t += 1
        self.history.append(self.w.copy())
        return self.w
