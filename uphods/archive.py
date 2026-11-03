"""Elite archive, handshake mapping and elite injection (Sec. 3.2.2)."""

from __future__ import annotations

import numpy as np


class EliteArchive:
    """Designs whose utility exceeds the q-quantile of all designs observed so
    far, filtered for crowding (normalised distance >= delta) and capped at
    ``max_size``.  The archive is rebuilt every epoch because utilities
    change as the preference model is updated."""

    def __init__(self, q=0.95, delta=0.05, max_size=50):
        self.q, self.delta, self.max_size = q, delta, max_size
        self.idx = np.empty(0, dtype=int)

    def rebuild(self, U_all, util_all):
        n, d = U_all.shape
        if n == 0:
            self.idx = np.empty(0, dtype=int)
            return self.idx
        thr = np.quantile(util_all, self.q)
        cand = np.where(util_all >= thr)[0]
        cand = cand[np.argsort(-util_all[cand], kind="stable")]
        keep = []
        for i in cand:
            if keep:
                dist = np.linalg.norm(U_all[keep] - U_all[i], axis=1) / np.sqrt(d)
                if dist.min() < self.delta:
                    continue
            keep.append(i)
            if len(keep) >= self.max_size:
                break
        self.idx = np.array(keep, dtype=int)
        return self.idx


def fit_affine(S_src, S_dst):
    """Least-squares affine map S_dst ~ S_src @ R.T + b (Eq. 19).

    With fewer handshake pairs than dimensions the problem is
    under-determined; the minimum-norm solution of lstsq is used.  When both
    solvers share the same encoding (all four solvers in this study use unit
    coordinates) the fitted map is the identity.
    """
    n, d = S_src.shape
    A = np.hstack([S_src, np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(A, S_dst, rcond=None)
    R, b = coef[:d].T, coef[d]
    return R, b


def handshake(U_elite, enc_src, enc_dst, n_pairs=10):
    """Map elites from the outgoing solver's representation to the incoming
    one (Eq. 19).  ``enc_*`` encode unit coordinates into each solver's
    representation; the affine map is fitted on ``n_pairs`` handshake elites
    and returned in the incoming representation.

    All four solvers in this study use unit coordinates, so the encodings
    coincide and the map is the identity; the general branch is kept for
    solvers with different encodings."""
    if len(U_elite) == 0:
        return U_elite
    H = U_elite[:n_pairs]
    S_src, S_dst = enc_src(H), enc_dst(H)
    if S_src.shape == S_dst.shape and np.allclose(S_src, S_dst):
        return U_elite.copy()
    R, b = fit_affine(S_src, S_dst)
    return enc_src(U_elite) @ R.T + b


def identity_encoding(U):
    return U


def transfer(challenger, elites_U, elites_f, ev, inject_ratio=0.4, enabled=True, pool=None):
    """Warm-start ``challenger`` at a switch.

    The default start population of the challenger is its own retained state
    (or a random population if it has never run).  With elite transfer
    enabled, ``inject_ratio`` of that population is replaced by the
    handshake-mapped elites; single-trajectory solvers (SA, GD) start from the
    best of their retained point and the elites.  Injected designs keep their
    recorded objective values (deterministic objective, unchanged by the
    identity map), so no re-evaluation is needed.
    """
    own_U, own_f = challenger.population() if challenger.started else (None, None)
    if pool is not None and enabled:
        # default start population taken from the outgoing solver's current
        # population instead of the challenger's own (possibly stale) state
        own_U, own_f = pool
    if not enabled or len(elites_U) == 0:
        challenger.start(own_U, own_f, ev)
        return
    E = handshake(elites_U, identity_encoding, identity_encoding)
    if challenger.single_point:
        U = E[:1] if own_U is None else np.vstack([E[:1], own_U[:1]])
        f = elites_f[:1] if own_f is None else np.concatenate([elites_f[:1], own_f[:1]])
        challenger.start(U, f, ev)
        return
    N = challenger.N
    n_inj = min(len(E), max(1, int(round(inject_ratio * N))))
    if own_U is None:
        U, f = E[:n_inj], elites_f[:n_inj]
    else:
        order = np.argsort(own_f)[: N - n_inj]  # drop the worst members
        U = np.vstack([own_U[order], E[:n_inj]])
        f = np.concatenate([own_f[order], elites_f[:n_inj]])
    challenger.start(U, f, ev)  # missing members are filled randomly


def lookahead_population(
    challenger, elites_U, elites_f, rng, inject_ratio=0.4, enabled=True
):
    """Designs the challenger would evaluate in its first epoch after a
    switch, generated without spending evaluations (random fill omitted)."""
    c = challenger.clone()
    own_U, own_f = c.population() if c.started else (None, None)
    if enabled and len(elites_U):
        if c.single_point:
            parts = [(elites_U[:1], elites_f[:1])]
            if own_U is not None:
                parts.append((own_U[:1], own_f[:1]))
        else:
            n_inj = min(len(elites_U), max(1, int(round(inject_ratio * c.N))))
            parts = [(elites_U[:n_inj], elites_f[:n_inj])]
            if own_U is not None:
                order = np.argsort(own_f)[: c.N - n_inj]
                parts.append((own_U[order], own_f[order]))
        U = np.vstack([p[0] for p in parts])
        f = np.concatenate([p[1] for p in parts])
    elif own_U is not None:
        U, f = own_U, own_f
    else:
        return rng.random((c.N if not c.single_point else 1, c.dim))
    _start_no_eval(c, U, f)
    return c.propose(rng)


def _start_no_eval(c, U, f):
    """Initialise a cloned solver from fully evaluated designs only."""

    class _NoEval:
        remaining = 0

    n = len(U)
    if not c.single_point and n < c.N:
        # pad by repetition so the solver's arrays have the usual size
        rep = np.resize(np.arange(n), c.N)
        U, f = U[rep], f[rep]
    c.start(U, f, _NoEval())
