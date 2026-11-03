"""UPHODS v2: solver-level hybrid utility.

Why a v2 is needed (dev-seed pilot, results/dev/v1.summary.md): the switching
score of the 0201 version compares the *population-averaged utility of
designs* (Eqs. 13-18).  Every solver started from the same elites produces
designs of similar absolute quality, so the confidence condition of Eq. (18)
is almost never met; the faithful implementation switched about once per run
and behaved like the initial solver (GA).

v2 keeps the components of the paper (preference GP, robust normalisation,
simplex weights learned from preferences, challenger score mu + kappa sigma,
hysteresis with cool-down, elite transfer) but applies the hybrid utility to
what a switch decision is about: how much each *solver* currently
contributes.

* After every epoch the active solver receives an observation
  y = w^T m~ of the weighted, normalised metric vector of that epoch
  (m1: log-improvement of the best objective per 100 evaluations, m2: batch
  diversity, m3: robustness of the utility trajectory).
* Each solver's utility is estimated with exponential discounting in
  evaluation time (non-stationary), giving a mean and a standard error.
* An initial probing round runs every solver for one epoch.
* Challenger: highest score S = mu + kappa sigma.  Switch when
  S_B - S_A > eps (hysteresis margin) and the cool-down, counted in
  evaluations, has elapsed.  ``rule="lcb"`` gives the 0201 rule of Eq. (18)
  for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .controller import UPHODS, UPHODSConfig
from .metrics import diversity, robustness
from .solvers import ORDER


@dataclass
class V2Config(UPHODSConfig):
    probe: bool = True
    eta0: float = 2.0  # 0201 Table 2: 0.05 (too small to move the weights; DEVLOG round 2)
    rule: str = "margin"  # "margin": S_B - S_A > eps ; "lcb": Eq. (18) ; "ts": Thompson
    beta: float = 1.0
    kappa: float = 1.0
    eps: float = 0.2
    cooldown_evals: int = 10
    half_life: float = 200.0  # evaluations; discounting of old observations
    prior_mean: float = 0.5
    prior_sd: float = 0.3
    obs_sd: float = 0.2
    m1_scale: str = "median"  # how log-progress is mapped to [0, 1]
    signal: str = "metrics"  # "metrics" (UPHODS) or "preference" (GP-PO)
    min_epochs: int = 1  # minimum consecutive epochs of a tenure
    w_init: tuple | None = None  # initial metric weights (default uniform)
    weight_target: str = "epoch"  # "epoch": preference gain per epoch; "design": Eq. (10)
    weight_window: int = 30  # epochs used in the weight objective
    estimator: str = "phase"  # "discounted" or "phase" (solver effect + shared time trend)
    phase_half_life: float = 200.0  # evaluations, discounting in the phase regression
    phase_floor: float = 0.01  # zero-progress floor, fraction of the median positive m1
    phase_prior_sd: float = 1.5  # prior sd of solver effects around the common mean
    # "logm1": z = log(m1 + floor) (objective progress only)
    # "hybrid": y = w^T m~ + p, the solver-level form of Eq. (8): objective
    #   channel w^T m~ (metrics robustly normalised by Eqs. 2-4, weights learned
    #   from preferences) plus subjective channel p (preference GP probability
    #   that the epoch produced a more preferred design)
    phase_target: str = "hybrid2"
    phase_slope_sd: float = 0.5  # >0: solver-specific decay slopes shrunk to the shared one
    phase_age: bool = False  # tenure-age covariate: progress right after a (re)start vs. later
    # "hybrid2": z = log(m1 + floor) + lam_p * p + lam_m * (w^T m~ - 0.5):
    #   objective progress in log units, plus the subjective channel and the
    #   preference-weighted secondary metrics as bounded additive terms
    lam_p: float = 0.5
    lam_m: float = 0.5
    # design-level utility used for the elite archive and the recommendation:
    # "metrics": u(x) = w^T m(x) + r(x) as in the 0201 Eq. (8), where m(x) are the
    #   metrics of the epoch that produced x (all designs of one epoch share them)
    # "objective": u(x) = s(x) + r(x), s = rank-normal score of the objective
    #   value of x itself, r = GP residual learned from the pairwise preferences
    design_util: str = "objective"


def log_progress(best_before, best_after, n_evals, delta=1e-12):
    """Log-improvement of the best objective per 100 evaluations."""
    if n_evals <= 0:
        return 0.0
    g = np.log(max(best_before, 0.0) + delta) - np.log(max(best_after, 0.0) + delta)
    return max(g, 0.0) * 100.0 / n_evals


class UPHODS2(UPHODS):
    method = "UPHODS-v2"

    def __init__(self, problem, budget, seed, cfg: V2Config | None = None, method=None):
        cfg = cfg or V2Config()
        super().__init__(problem, budget, seed, cfg, method)
        self.obs = {n: [] for n in ORDER}  # (eval time, y)
        self.m1_hist = []
        self.probe_queue = [n for n in ORDER if n != cfg.initial] if cfg.probe else []
        self.last_switch_evals = 0
        self.best_pref_id = None
        self.epoch_feats, self.epoch_pref = [], []
        self.phase_obs = []  # (solver, mid-epoch evaluation count, raw m1)
        if cfg.w_init is not None:
            w0 = np.asarray(cfg.w_init, dtype=float) + 1e-6
            self.w.w = w0 / w0.sum()
            self.w.history = [self.w.w.copy()]

    # ------------------------------------------------------------ metrics
    def _raw_metrics(self, U_batch):
        e = self.epochs[-1]
        m1 = log_progress(
            e["best_before"], e["best_after"], e["evals_after"] - e["evals_before"]
        )
        m2 = diversity(U_batch) if len(U_batch) > 1 else 0.0
        m3 = robustness(self.util_traj[self.active][-self.cfg.robust_window :])
        return np.array([m1, m2, m3])

    def _metric_vector(self, raw):
        """m1 is mapped with a run-level scale (median of all positive
        log-progress values so far), so values stay comparable between
        solvers and over time; m2 and m3 already lie in [0, 1].  With robust
        normalisation enabled the Hampel filter guards the scale estimate."""
        c = self.cfg
        m1, m2, m3 = raw
        pos = [v for v in self.m1_hist if v > 0]
        if not pos:
            s = 1.0
        elif c.robust_norm:
            a = np.asarray(pos)
            med = np.median(a)
            mad = 1.4826 * np.median(np.abs(a - med))
            a = a[np.abs(a - med) <= 3.0 * mad + 1e-12] if len(a) >= 5 else a  # Hampel
            s = float(np.median(a))
        else:
            s = float(np.mean(pos))
        m1n = 1.0 - np.exp(-m1 / max(s, 1e-12))
        return np.array([m1n, m2, m3])

    def _preference_signal(self):
        """GP-PO signal: probability (preference GP) that the most preferred
        design after the epoch beats the one before it, mapped to [0, 1]."""
        if not self.pairs:
            return 0.0
        util = self.utility_all()
        new_best = int(np.argmax(util))
        old = self.best_pref_id
        self.best_pref_id = new_best
        if old is None or old == new_best:
            return 0.0
        mu, sd = self.gp.predict(self.U_all()[[new_best, old]])
        if self.cfg.design_util == "objective" and self.cfg.metric_channel:
            mu = mu + self.objective_score()[[new_best, old]]
        z = (mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2 + self.gp.c2)
        return float(max(2.0 * norm.cdf(z) - 1.0, 0.0))

    # ---------------------------------------------------------- tracking
    def estimate(self, name):
        """Discounted mean and standard error of a solver's utility."""
        c = self.cfg
        t = self.ev.n_evals
        obs = self.obs[name]
        # prior acts as one pseudo-observation at the start of the run
        ts = np.array([0.0] + [o[0] for o in obs])
        ys = np.array([c.prior_mean] + [o[1] for o in obs])
        wts = 0.5 ** ((t - ts) / c.half_life)
        wts[0] *= (c.obs_sd / c.prior_sd) ** 2
        n_eff = wts.sum()
        mean = float(np.sum(wts * ys) / n_eff)
        sd = float(c.obs_sd / np.sqrt(n_eff))
        return mean, sd

    def after_epoch(self, i0, i1, U_batch):
        e = self.epochs[-1]
        self.m1_hist.append(
            log_progress(
                e["best_before"], e["best_after"], e["evals_after"] - e["evals_before"]
            )
        )
        super().after_epoch(i0, i1, U_batch)
        c = self.cfg
        p = self._preference_signal()
        if c.phase_target in ("hybrid", "hybrid2"):
            m = UPHODS._metric_vector(self, self.last_raw[self.active])  # Eqs. 2-4
        else:
            m = self._metric_vector(self.last_raw[self.active])
        self.epoch_feats.append(m)
        self.epoch_pref.append(p)
        if c.weight_target == "epoch" and self.w.adaptive and self.pairs:
            k = c.weight_window
            y = np.array(self.epoch_pref[-k:])
            F = np.array(self.epoch_feats[-k:])
            if len(y) >= 3:
                # centred regression: weights follow the covariance between a
                # metric and the preference gain, not the metric's mean level
                self.w.step(y - y.mean(), F - F.mean(axis=0))
        y = p if c.signal == "preference" else float(m @ self.w.w)
        self.obs[self.active].append((self.ev.n_evals, y))
        e = self.epochs[-1]
        self.phase_obs.append((self.active, 0.5 * (e["evals_before"] + e["evals_after"]),
                               self.m1_hist[-1], float(m @ self.w.w) + p,
                               c.lam_p * p + c.lam_m * (float(m @ self.w.w) - 0.5),
                               len(self.tenure()) - 1))

    def _fit(self):
        """GP fit; with weight_target="epoch" the design-level weight step of
        Eq. (10) is replaced by the epoch-level step in after_epoch."""
        if self.cfg.weight_target == "design":
            return super()._fit()
        if not self.pairs:
            return
        pos = {d: i for i, d in enumerate(self.train_ids)}
        P = np.array([(pos[a], pos[b]) for a, b, _ in self.pairs])
        keys = [k for _, _, k in self.pairs]
        if self.cfg.design_util == "objective":
            sc = self.objective_score()[self.train_ids] if self.cfg.metric_channel else                 np.zeros(len(self.train_ids))
            off = sc[P[:, 0]] - sc[P[:, 1]]
        else:
            Mt = self.M_all[self.train_ids]
            off = (Mt[P[:, 0]] - Mt[P[:, 1]]) @ self._w()
        self.gp.fit(self.U_all()[self.train_ids], P, off, keys)

    def objective_score(self):
        """Rank-normal score of every evaluated design (higher = lower f)."""
        from scipy.stats import norm as _n, rankdata
        f = self.f_all()
        r = rankdata(-f, method="average")
        return _n.ppf((r - 0.5) / len(f))

    def utility_all(self):
        if self.cfg.design_util != "objective":
            return super().utility_all()
        s = self.objective_score() if self.cfg.metric_channel else 0.0
        if not self.pairs:
            return s if self.cfg.metric_channel else super().utility_all()
        r, _ = self.gp.predict(self.U_all())
        return s + r

    # --------------------------------------------------------- switching
    def phase_estimates(self):
        """Phase-adjusted solver utility.

        Weighted Bayesian linear regression of z = log(m1 + floor) on solver
        indicators and a shared trend in log(evaluations):
            z_e = a_s(e) + b * log(t_e) + noise.
        The shared trend absorbs the fact that progress is easier early in a
        run, so observations of different solvers made at different times
        become comparable.  Returns, per solver, the predicted log-progress
        at the current time and its posterior standard deviation."""
        c = self.cfg
        t_now = max(self.ev.n_evals, 1)
        obs = self.phase_obs
        pos = [o[2] for o in obs if o[2] > 0]
        floor = c.phase_floor * (float(np.median(pos)) if pos else 1.0)
        k = len(ORDER)
        if not obs:
            return {n: (0.0, c.phase_prior_sd) for n in ORDER}
        slopes = c.phase_slope_sd > 0
        p_dim = k + 1 + (k if slopes else 0) + (1 if c.phase_age else 0)
        X = np.zeros((len(obs), p_dim))
        z = np.empty(len(obs))
        wt = np.empty(len(obs))
        hybrid = c.phase_target == "hybrid"
        lt_now = np.log(t_now)
        for i, (n, t, v, yh, extra, age) in enumerate(obs):
            j = ORDER.index(n)
            lt = np.log(max(t, 1.0)) - lt_now        # centred: effects are levels at t_now
            X[i, j] = 1.0
            X[i, k] = lt
            if slopes:
                X[i, k + 1 + j] = lt
            if c.phase_age:
                X[i, -1] = np.log1p(age)
            if hybrid:
                z[i] = yh
            else:
                z[i] = np.log(v + floor) + (extra if c.phase_target == "hybrid2" else 0.0)
            wt[i] = 0.5 ** ((t_now - t) / c.phase_half_life)
        m0 = float(np.sum(wt * z) / np.sum(wt))
        mu0 = np.zeros(p_dim)
        mu0[:k] = m0
        prior_var = np.ones(p_dim)
        prior_var[:k] = c.phase_prior_sd ** 2
        if slopes:
            prior_var[k + 1: 2 * k + 1] = c.phase_slope_sd ** 2
        P0inv = np.diag(1.0 / prior_var)
        sig2 = 1.0
        for _ in range(2):
            A = X.T @ (wt[:, None] * X) / sig2 + P0inv
            bvec = X.T @ (wt * z) / sig2 + P0inv @ mu0
            Sig = np.linalg.inv(A)
            beta = Sig @ bvec
            r = z - X @ beta
            s_floor = 0.05 if hybrid else 0.3
            sig2 = max(float(np.sum(wt * r * r) + s_floor ** 2) / (np.sum(wt) + 1.0), s_floor ** 2)
        out = {}
        for j, n in enumerate(ORDER):
            x = np.zeros(p_dim)
            x[j] = 1.0
            if c.phase_age:
                # incumbent continues its tenure; a challenger would start afresh
                age_next = len(self.tenure()) if n == self.active else 0
                x[-1] = np.log1p(age_next)
            out[n] = (float(x @ beta), float(np.sqrt(x @ Sig @ x)))
        return out

    def scores(self):
        out = {}
        est = self.phase_estimates() if self.cfg.estimator == "phase" else None
        for n in ORDER:
            m, s = est[n] if est else self.estimate(n)
            out[n] = (m, s, m + self.cfg.kappa * s)
        return out

    def decide(self):
        c = self.cfg
        if self.probe_queue:
            return self.probe_queue.pop(0)
        if c.hysteresis and (
            self.ev.n_evals - self.last_switch_evals < c.cooldown_evals
            or len(self.tenure()) < c.min_epochs
        ):
            return None
        sc = self.scores()
        A = self.active
        if c.rule == "ts":
            # Thompson sampling: one draw per solver from its utility estimate
            draw = {n: sc[n][0] + sc[n][1] * self.rng_prop.standard_normal() for n in ORDER}
            B = max((n for n in ORDER if n != A), key=lambda n: draw[n])
            gap = draw[B] - draw[A]
            self.decisions.append((self.epoch, A, B, gap))
            return B if gap > (c.eps if c.hysteresis else 0.0) else None
        B = max((n for n in ORDER if n != A), key=lambda n: sc[n][2])
        if not c.hysteresis:
            gap = sc[B][2] - sc[A][2]
            thr = 0.0
        elif c.rule == "lcb":
            gap = (sc[B][0] - c.beta * sc[B][1]) - (sc[A][0] + c.beta * sc[A][1])
            thr = c.eps
        else:
            gap = sc[B][2] - sc[A][2]
            thr = c.eps
        self.decisions.append((self.epoch, A, B, gap))
        return B if gap > thr else None

    def switch(self, nxt, reason=""):
        super().switch(nxt, reason)
        self.last_switch_evals = self.ev.n_evals

    def result(self):
        r = super().result()
        r["est"] = {
            n: [round(v, 4) for v in self.estimate(n)] + [len(self.obs[n])]
            for n in ORDER
        }
        return r
