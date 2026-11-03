"""Hybrid controllers.

:class:`HybridBase` holds what all hybrid methods share: the solver library,
the evaluation counter, the elite archive and the elite transfer at a switch.
Subclasses only decide *whether and where* to switch (``decide``) and may
update a model after each epoch (``after_epoch``).

:class:`UPHODS` implements Algorithm 1 of the paper; the ablation variants
and the GP-PO baseline are configurations of the same class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.stats import norm

from .archive import EliteArchive, lookahead_population, transfer
from .metrics import RobustNormalizer, convergence_speed, diversity, robustness
from .oracle import SimulatedDM
from .pref_gp import PreferenceGP, SimplexWeights
from .problems import Evaluator, Problem
from .solvers import ORDER, make_library


class HybridBase:
    method = "hybrid"

    def __init__(
        self,
        problem: Problem,
        budget: int,
        seed: int,
        initial="GA",
        elite_transfer=True,
        inject_ratio=0.4,
        pop_size=20,
        archive_q=0.95,
        archive_delta=0.05,
        archive_max=50,
        handover="archive",
        dm_h_lambda=0.0,
        transfer_pool="own",
    ):
        self.transfer_pool = transfer_pool  # "own" (0201 text) or "incumbent"
        self.dm_h_lambda = dm_h_lambda  # >0: preference-misaligned scenario
        # handover at a switch: "archive" = elite archive + injection (UPHODS,
        # Sec. 3.2.2); "incumbent" = the incoming solver takes over the
        # outgoing solver's current population (classical sequential hybrid);
        # "none" = the incoming solver resumes its own previous state
        self.handover = handover if elite_transfer else "none"
        self.problem = problem
        self.seed = seed
        self.rng = np.random.default_rng([seed, 0])
        self.rng_prop = np.random.default_rng([seed, 1])  # look-ahead proposals
        self.ev = Evaluator(problem, budget)
        self.lib = make_library(problem.dim, np.random.default_rng([seed, 2]), pop_size)
        self.active = initial
        self.elite_transfer = elite_transfer
        self.inject_ratio = inject_ratio
        self.archive = EliteArchive(archive_q, archive_delta, archive_max)
        self.epoch = 0
        self.epochs = []  # dict per epoch
        self.switches = []  # dict per switch
        self.usage = {n: 0 for n in ORDER}
        self.tenure_start = 0  # index into self.epochs where the current tenure began

    # ------------------------------------------------------------------ data
    def U_all(self):
        return np.asarray(self.ev.U_log)

    def f_all(self):
        return np.asarray(self.ev.f_log)

    def utility_all(self):
        """Utility used to rank designs for the elite archive.  Methods
        without a preference model rank by the objective."""
        return -np.log1p(np.maximum(self.f_all(), 0.0))

    def elites(self):
        idx = self.archive.rebuild(self.U_all(), self.utility_all())
        return self.U_all()[idx], self.f_all()[idx]

    # -------------------------------------------------------------- tenure
    def tenure(self):
        return self.epochs[self.tenure_start :]

    def window_improvement(self, W=3):
        """Relative improvement per 100 evaluations over the last W epochs of
        the current tenure (metric m1)."""
        ten = self.tenure()[-W:]
        if not ten:
            return 0.0
        return convergence_speed(
            ten[0]["best_before"],
            ten[-1]["best_after"],
            ten[-1]["evals_after"] - ten[0]["evals_before"],
        )

    # ---------------------------------------------------------------- loop
    def switch(self, nxt, reason=""):
        U_e, f_e = (
            self.elites()
            if self.elite_transfer
            else (np.empty((0, self.problem.dim)), np.empty(0))
        )
        n0 = self.ev.n_evals
        if self.handover == "incumbent":
            U_i, f_i = self.lib[self.active].population()
            self.lib[nxt].start(U_i, f_i, self.ev)
        else:
            pool = None
            if self.transfer_pool == "incumbent":
                pool = self.lib[self.active].population()
            elif self.transfer_pool == "random" or (
                self.transfer_pool == "adaptive" and self._converged(nxt)
            ):
                # diversity-preserving transfer: elites plus fresh random designs
                pool = (np.empty((0, self.problem.dim)), np.empty(0))
            transfer(
                self.lib[nxt], U_e, f_e, self.ev, self.inject_ratio, self.elite_transfer, pool
            )
        self.usage[nxt] += self.ev.n_evals - n0
        self.switches.append(
            dict(
                epoch=self.epoch,
                evals=self.ev.n_evals,
                frm=self.active,
                to=nxt,
                best=self.ev.best_f,
                reason=reason,
            )
        )
        self.active = nxt
        self.tenure_start = len(self.epochs)

    def _converged(self, name, thr=0.05):
        """True if solver ``name``'s retained population has collapsed
        (normalised mean pairwise distance below ``thr``)."""
        from .metrics import diversity
        s = self.lib[name]
        if not s.started or s.single_point:
            return False
        U, _ = s.population()
        return diversity(U) < thr

    def after_epoch(self, i0, i1, U_batch):
        pass

    def decide(self):
        return None

    def run(self):
        t0 = time.perf_counter()
        ev = self.ev
        first = True
        while ev.remaining > 0:
            i0, best0 = ev.n_evals, ev.best_f
            s = self.lib[self.active]
            if first:
                s.start(None, None, ev)
                first = False
            Ub, _ = s.step(ev)
            if ev.n_evals == i0:
                raise RuntimeError(
                    f"{self.active} epoch spent no evaluation "
                    f"(remaining={ev.remaining}); refusing to loop forever"
                )
            self.usage[self.active] += ev.n_evals - i0
            if best0 == np.inf:
                best0 = float(np.max(ev.f_log))  # reference for the first epoch
            self.epoch += 1
            self.epochs.append(
                dict(
                    epoch=self.epoch,
                    solver=self.active,
                    evals_before=i0,
                    evals_after=ev.n_evals,
                    best_before=best0,
                    best_after=ev.best_f,
                )
            )
            self.after_epoch(i0, ev.n_evals, Ub)
            if ev.remaining <= 0:
                break
            nxt = self.decide()
            if nxt is not None and nxt != self.active:
                self.switch(nxt)
        self.wall_seconds = time.perf_counter() - t0
        return self.result()

    def result(self):
        best_at = [s["best"] for s in self.switches] + [self.ev.best_f]
        gains = [
            max(best_at[j] - best_at[j + 1], 0.0) for j in range(len(self.switches))
        ]
        return dict(
            method=self.method,
            function=self.problem.name,
            dim=self.problem.dim,
            seed=self.seed,
            best=self.ev.best_f,
            evals=self.ev.n_evals,
            curve=self.ev.curve(20).tolist(),
            ns=len(self.switches),
            aip=float(np.mean(gains)) if gains else 0.0,
            switches=[(s["evals"], s["frm"], s["to"]) for s in self.switches],
            usage=self.usage,
            epochs=self.epoch,
            wall_seconds=self.wall_seconds,
            **self.dm_evaluation(),
        )

    def dm_evaluation(self, top_k=10):
        """Misaligned scenario: the DM picks, by its hidden utility, the best
        of the method's own top-k designs (PROTOCOL.md section 8)."""
        if self.dm_h_lambda <= 0:
            return {}
        from .misaligned import dm_utility

        U, f = self.U_all(), self.f_all()
        u_star = dm_utility(f, U, self.problem, self.dm_h_lambda)
        top = np.argsort(-self.utility_all(), kind="stable")[:top_k]
        pick = top[int(np.argmax(u_star[top]))]
        return dict(
            dm_pick_u=float(u_star[pick]),
            dm_pick_f=float(f[pick]),
            dm_best_u=float(u_star.max()),
        )


# ---------------------------------------------------------------------------
# UPHODS
# ---------------------------------------------------------------------------


@dataclass
class UPHODSConfig:
    # switching (Table 2)
    cooldown: int = 10
    beta: float = 1.96
    eps: float = 0.0
    kappa: float = 0.2
    hysteresis: bool = True  # False -> UPHODS-noHys (beta = 0, no cool-down)
    # elite transfer
    elite_transfer: bool = True  # False -> UPHODS-noET
    inject_ratio: float = 0.4
    archive_q: float = 0.95
    archive_delta: float = 0.05
    archive_max: int = 50
    # preference model
    metric_channel: bool = True  # False -> GP-PO (preference-only utility)
    lengthscale: float = 0.5
    noise: float = 1.0 / np.sqrt(2.0)
    tau: float = 1.0
    slate_size: int = 5
    # metric weights
    adaptive_weights: bool = True  # False -> UPHODS-fixedW
    eta0: float = 0.05
    gamma: float = 1e-4
    dirichlet_alpha: float = 1.2
    # robust normalisation
    robust_norm: bool = True  # False -> UPHODS-noRN
    rn_lambda: float = 0.2
    rn_window: int = 10
    hampel_tau: float = 3.0
    tanh_zeta: float = 2.0
    metric_window: int = 3  # W in the revised m1
    robust_window: int = 5  # h in m3
    # simulated decision maker
    dm_sigma: float = 0.05
    dm_flip: float = 0.0
    initial: str = "GA"
    pop_size: int = 20
    handover: str = "archive"
    dm_h_lambda: float = 0.0  # >0: preference-misaligned scenario (PROTOCOL 8)
    transfer_pool: str = "own"


class UPHODS(HybridBase):
    method = "UPHODS"

    def __init__(
        self, problem, budget, seed, cfg: UPHODSConfig | None = None, method=None
    ):
        cfg = cfg or UPHODSConfig()
        self.cfg = cfg
        super().__init__(
            problem,
            budget,
            seed,
            cfg.initial,
            cfg.elite_transfer,
            cfg.inject_ratio,
            cfg.pop_size,
            cfg.archive_q,
            cfg.archive_delta,
            cfg.archive_max,
            cfg.handover,
            cfg.dm_h_lambda,
            cfg.transfer_pool,
        )
        if method:
            self.method = method
        dm_u = None
        if cfg.dm_h_lambda > 0:
            from .misaligned import dm_utility

            dm_u = lambda f, U, _p=problem, _l=cfg.dm_h_lambda: dm_utility(f, U, _p, _l)
        self.dm = SimulatedDM(
            np.random.default_rng([seed, 3]), cfg.dm_sigma, cfg.dm_flip, dm_u
        )
        self.gp = PreferenceGP(problem.dim, cfg.lengthscale, 1.0, cfg.noise, cfg.tau)
        self.w = SimplexWeights(
            3,
            cfg.eta0,
            cfg.gamma,
            cfg.dirichlet_alpha,
            adaptive=cfg.adaptive_weights and cfg.metric_channel,
        )
        self.norms = [
            RobustNormalizer(
                cfg.rn_lambda,
                cfg.rn_window,
                cfg.hampel_tau,
                cfg.tanh_zeta,
                cfg.robust_norm,
            )
            for _ in range(3)
        ]
        self.M_all = np.zeros((0, 3))  # metric vector of each evaluated design
        self.train_ids = []  # design ids that appear in comparisons
        self.pairs = []  # (winner_id, loser_id, key)
        self.last_query_epoch = -(10**9)
        self.last_raw = {}  # solver -> latest raw metric vector
        self.util_traj = {n: [] for n in ORDER}
        self.cool = 0
        self.decisions = []

    # ---------------------------------------------------------- metrics
    def _metric_vector(self, raw):
        return np.array([self.norms[k].transform(raw[k]) for k in range(3)])

    def _raw_metrics(self, U_batch):
        c = self.cfg
        m1 = self.window_improvement(c.metric_window)
        m2 = diversity(U_batch) if len(U_batch) > 1 else 0.0
        traj = self.util_traj[self.active][-c.robust_window :]
        m3 = robustness(traj)
        return np.array([m1, m2, m3])

    # ---------------------------------------------------------- utility
    def _w(self):
        return self.w.w if self.cfg.metric_channel else np.zeros(3)

    def utility(self, U, M):
        r, sd = self.gp.predict(U)
        return M @ self._w() + r, sd

    def utility_all(self):
        if not self.pairs:
            return super().utility_all()
        return self.utility(self.U_all(), self.M_all)[0]

    # ---------------------------------------------------------- queries
    def _query_due(self):
        frac = self.ev.n_evals / self.ev.budget
        interval = 1 + int(4 * frac)  # queries become rarer as the run proceeds
        return self.epoch - self.last_query_epoch >= interval

    def _query(self, i0, i1):
        f = self.f_all()
        batch = np.arange(i0, i1)
        if len(batch) == 0:
            return
        k = self.cfg.slate_size
        top = batch[np.argsort(f[batch])][:2]
        rest = np.setdiff1d(batch, top)
        slate = list(top)
        elite = [i for i in self.archive.idx if i not in slate]
        if elite:
            slate.append(elite[0])
            if len(elite) > 1:
                slate.append(int(self.rng.choice(elite[1:])))
        n_rand = max(0, k - len(slate))
        if len(rest):
            slate += list(self.rng.choice(rest, min(n_rand, len(rest)), replace=False))
        slate = list(dict.fromkeys(int(s) for s in slate))[:k]
        if len(slate) < 2:
            return
        q = self.dm.n_queries
        for a, b in self.dm.pairs(slate, f[slate], self.U_all()[slate]):
            self.pairs.append((a, b, (q, a, b)))
        for s in slate:
            if s not in self.train_ids:
                self.train_ids.append(s)
        self.last_query_epoch = self.epoch

    def _fit(self):
        if not self.pairs:
            return
        pos = {d: i for i, d in enumerate(self.train_ids)}
        P = np.array([(pos[a], pos[b]) for a, b, _ in self.pairs])
        keys = [k for _, _, k in self.pairs]
        Mt = self.M_all[self.train_ids]
        w = self._w()
        off = (Mt[P[:, 0]] - Mt[P[:, 1]]) @ w
        self.gp.fit(self.U_all()[self.train_ids], P, off, keys)
        if self.w.adaptive:
            u_hat = Mt @ w + self.gp.mu_train
            self.w.step(u_hat, Mt)

    # ---------------------------------------------------------- epoch hook
    def after_epoch(self, i0, i1, U_batch):
        raw = self._raw_metrics(U_batch)
        if self.cfg.robust_norm:
            for k in range(3):
                self.norms[k].update(raw[k])
        m = self._metric_vector(raw)
        self.last_raw[self.active] = raw
        n_new = i1 - len(self.M_all)
        self.M_all = np.vstack([self.M_all, np.repeat(m[None], n_new, axis=0)])
        if self._query_due():
            self._query(i0, i1)
        self._fit()
        self.archive.rebuild(self.U_all(), self.utility_all())
        if len(U_batch):
            u, _ = self.utility(U_batch, np.repeat(m[None], len(U_batch), axis=0))
            self.util_traj[self.active].append(float(norm.cdf(np.mean(u))))

    # ---------------------------------------------------------- switching
    def _profile(self, name, U_prop):
        """Metric vector attached to the designs solver ``name`` would produce."""
        raw = self.last_raw.get(name)
        m2 = diversity(U_prop) if len(U_prop) > 1 else 0.0
        if raw is None:
            m = np.full(3, 0.5)
            if not self.cfg.robust_norm:
                obs = np.array(list(self.last_raw.values()))
                m = obs.mean(axis=0) if len(obs) else np.zeros(3)
            m[1] = self.norms[1].transform(m2)
            return m
        raw = raw.copy()
        raw[1] = m2
        return self._metric_vector(raw)

    def scores(self):
        idx = self.archive.idx
        U_e, f_e = self.U_all()[idx], self.f_all()[idx]
        out = {}
        for name in ORDER:
            s = self.lib[name]
            if name == self.active:
                P = s.propose(self.rng_prop)
            else:
                P = lookahead_population(
                    s,
                    U_e,
                    f_e,
                    self.rng_prop,
                    self.cfg.inject_ratio,
                    self.cfg.elite_transfer,
                )
            P = np.unique(P, axis=0)
            M = np.repeat(self._profile(name, P)[None], len(P), axis=0)
            mu, sd = self.utility(P, M)
            mbar = float(np.mean(mu))
            sbar = float(np.sqrt(np.sum(sd**2)) / len(P))  # Eq. 14
            out[name] = (mbar, sbar, mbar + self.cfg.kappa * sbar)  # Eq. 15
        return out

    def decide(self):
        c = self.cfg
        if not self.pairs:
            return None
        sc = self.scores()
        A = self.active
        B = max((n for n in ORDER if n != A), key=lambda n: sc[n][2])
        beta = c.beta if c.hysteresis else 0.0
        gap = (sc[B][0] - beta * sc[B][1]) - (sc[A][0] + beta * sc[A][1])  # Eq. 18
        self.decisions.append((self.epoch, A, B, gap))
        if c.hysteresis and self.cool > 0:
            self.cool -= 1
            return None
        if gap > c.eps:
            self.cool = c.cooldown if c.hysteresis else 0
            return B
        return None

    def result(self):
        r = super().result()
        r.update(
            n_pairs=len(self.pairs),
            n_train=len(self.train_ids),
            gp_seconds=float(np.sum(self.gp.fit_seconds)),
            gp_fits=len(self.gp.fit_seconds),
            gp_max_seconds=float(np.max(self.gp.fit_seconds))
            if self.gp.fit_seconds
            else 0.0,
            w_final=self.w.w.tolist(),
            w_traj=[
                h.tolist() for h in self.w.history[:: max(1, len(self.w.history) // 20)]
            ],
        )
        return r
