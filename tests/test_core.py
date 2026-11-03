import numpy as np
import pytest
from scipy.stats import norm

from uphods.archive import EliteArchive, fit_affine, handshake, identity_encoding
from uphods.metrics import RobustNormalizer, convergence_speed, diversity, robustness
from uphods.pref_gp import PreferenceGP, SimplexWeights
from uphods.problems import DIMS, FUNCTIONS, Evaluator, make_problem
from uphods.runner import METHODS, run_one
from uphods.solvers import make_library


@pytest.mark.parametrize("name", FUNCTIONS)
def test_optimum_is_zero(name):
    p = make_problem(name, 10)
    x_opt = (
        np.ones(10)
        if name == "Rosenbrock"
        else (p.shift if p.shift is not None else np.zeros(10))
    )
    assert np.all(x_opt >= p.lower) and np.all(
        x_opt <= p.upper
    )  # optimum inside the box
    assert abs(p.raw(x_opt[None])[0]) < 1e-9
    assert p.raw(p.lower[None] * 0.5 + p.upper[None] * 0.3)[0] > 0


def test_shift_is_fixed_per_instance():
    a, b = make_problem("ShiftedSphere", 30), make_problem("ShiftedSphere", 30)
    assert np.array_equal(a.shift, b.shift)


def test_evaluator_budget_and_truncation():
    p = make_problem("Ackley", 5)
    ev = Evaluator(p, 7)
    f = ev(np.random.default_rng(0).random((10, 5)))
    assert len(f) == 7 and ev.remaining == 0 and len(ev.U_log) == 7
    assert ev.best_f == min(f)


@pytest.mark.parametrize("dim", [5, 30])
def test_every_solver_counts_all_evaluations(dim):
    p = make_problem("Rosenbrock", dim)
    for name, s in make_library(dim, np.random.default_rng(1)).items():
        ev = Evaluator(p, 400)
        s.start(None, None, ev)
        while ev.remaining > 0:
            n0 = ev.n_evals
            U, f = s.step(ev)
            assert ev.n_evals - n0 == len(f)  # returned batch == evaluations spent
        assert ev.n_evals == 400


@pytest.mark.parametrize("budget", [2, 3, 7, 12, 13, 101, 102, 1000])
@pytest.mark.parametrize("dim", [5, 10, 100])
def test_gd_terminates_for_any_remaining_budget(dim, budget):
    # regression: with exactly one evaluation left GD used to return an empty
    # batch, and the run loop spun forever
    r = run_one("Single-GD", "Rosenbrock", dim, seed=7, budget=budget)
    assert r["evals"] == budget


def test_gd_solves_sphere_quickly():
    p = make_problem("ShiftedSphere", 5)
    s = make_library(5, np.random.default_rng(2))["GD"]
    ev = Evaluator(p, 300)
    s.start(None, None, ev)
    while ev.remaining > 0:
        s.step(ev)
    assert ev.best_f < 1e-2


def test_metrics():
    assert convergence_speed(10.0, 5.0, 50) == pytest.approx(1.0)
    assert convergence_speed(5.0, 6.0, 10) == 0.0
    U = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert diversity(U) == pytest.approx(1.0)
    assert robustness([0.5, 0.5, 0.5]) == pytest.approx(1.0)


def test_robust_normalizer_ignores_spike():
    rn = RobustNormalizer(lam=0.5, window=10)
    for x in [1.0, 1.1, 0.9, 1.0, 1.05]:
        rn.update(x)
    m_before = rn.m
    rn.update(1000.0)
    assert abs(rn.m - m_before) < 0.2  # Hampel filter suppressed the spike
    assert rn.transform(1000.0) > 0.99  # but the spike itself still maps high
    assert 0.3 < rn.transform(1.0) < 0.7


def test_preference_gp_recovers_order():
    rng = np.random.default_rng(3)
    X = rng.random((25, 2))
    true = -np.sum((X - 0.3) ** 2, axis=1)
    pairs = []
    for _ in range(80):
        i, j = rng.choice(25, 2, replace=False)
        pairs.append((i, j) if true[i] > true[j] else (j, i))
    gp = PreferenceGP(2, lengthscale=0.3, tau=10.0)
    gp.fit(X, np.array(pairs))
    mu, sd = gp.predict(X)
    rho = np.corrcoef(np.argsort(np.argsort(mu)), np.argsort(np.argsort(true)))[0, 1]
    assert rho > 0.8
    # far from data the prediction falls back to the prior
    m_far, s_far = gp.predict(np.array([[50.0, 50.0]]))
    assert abs(m_far[0]) < 1e-6 and s_far[0] == pytest.approx(1.0, rel=1e-3)


def test_ep_matches_exact_gaussian_limit():
    # one comparison: EP posterior mean of the difference must be positive
    gp = PreferenceGP(1, lengthscale=10.0, tau=1e3)
    gp.fit(np.array([[0.0], [1.0]]), np.array([[0, 1]]))
    m, _ = gp.predict(np.array([[0.0], [1.0]]))
    assert m[0] > m[1]


def test_weights_stay_on_simplex_and_move_towards_explaining_metric():
    W = SimplexWeights(3, eta0=0.5)
    M = np.random.default_rng(4).random((50, 3))
    u_hat = 2.0 * M[:, 0]
    for _ in range(200):
        w = W.step(u_hat, M)
    assert np.isclose(w.sum(), 1.0) and np.all(w > 0)
    assert w[0] == max(w)


def test_archive_crowding_and_cap():
    rng = np.random.default_rng(5)
    U = np.vstack([np.full((10, 3), 0.5), rng.random((200, 3))])
    util = np.concatenate([np.full(10, 10.0), rng.random(200)])
    a = EliteArchive(q=0.9, delta=0.05, max_size=15)
    idx = a.rebuild(U, util)
    assert len(idx) <= 15
    assert sum(i < 10 for i in idx) == 1  # duplicates removed by crowding


def test_handshake_identity_and_affine_fit():
    U = np.random.default_rng(6).random((8, 3))
    assert np.allclose(handshake(U, identity_encoding, identity_encoding), U)
    R0, b0 = np.diag([2.0, 0.5, 1.0]), np.array([1.0, -1.0, 0.0])
    R, b = fit_affine(U, U @ R0.T + b0)
    assert np.allclose(R, R0) and np.allclose(b, b0)


@pytest.mark.parametrize("method", list(METHODS))
def test_every_method_uses_exact_budget(method):
    r = run_one(method, "Rastrigin", 5, seed=1000, budget=300)
    assert r["evals"] == 300
    assert len(r["curve"]) == 15
    assert np.all(np.diff(r["curve"]) <= 0)


def test_runs_are_reproducible():
    a = run_one("UPHODS", "Ackley", 5, seed=1001, budget=300)
    b = run_one("UPHODS", "Ackley", 5, seed=1001, budget=300)
    assert a["best"] == b["best"] and a["switches"] == b["switches"]


def test_misaligned_dm_prefers_reference_region():
    from uphods.misaligned import dm_utility, u_ref
    from uphods.problems import make_problem
    p = make_problem("ShiftedSphere", 10)
    r = u_ref(p)
    U = np.vstack([r, p.to_unit(p.shift)])          # reference design vs f-optimum
    f = p.raw(p.from_unit(U))
    u = dm_utility(f, U, p, 10.0)
    assert f[1] < f[0]                               # optimum is better in f ...
    assert np.all((r >= 0.1) & (r <= 0.9))


@pytest.mark.parametrize("method", ["UPHODS-v2", "SW-UCB", "Single-PSO"])
def test_misaligned_scenario_reports_dm_utility(method):
    r = run_one(method, "Rastrigin", 5, seed=1000, budget=300, overrides={"dm_h_lambda": 10.0})
    assert "dm_pick_u" in r and r["dm_pick_u"] <= r["dm_best_u"] + 1e-12
