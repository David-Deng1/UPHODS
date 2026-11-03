"""Method registry and a single-run entry point."""

from __future__ import annotations

from dataclasses import replace

from .baselines import FSS1, FSS2, FSS3, FSS4, SWUCB, Single
from .controller import UPHODS, UPHODSConfig
from .controller_v2 import UPHODS2, V2Config
from .problems import make_problem

BUDGET = 1000


def _uphods(name, cls=UPHODS, cfg_cls=UPHODSConfig, **changes):
    def make(problem, budget, seed, overrides=None, **kw):
        cfg = replace(cfg_cls(), **changes, **(overrides or {}))
        return cls(problem, budget, seed, cfg, method=name)

    return make


def _v2(name, **changes):
    return _uphods(name, UPHODS2, V2Config, **changes)


def _plain(cls, *args):
    def make(problem, budget, seed, overrides=None, **kw):
        return cls(problem, budget, seed, *args, **(overrides or {}))

    return make


METHODS = {
    # v1: faithful reproduction of the 0201 description
    "UPHODS": _uphods("UPHODS"),
    "GP-PO": _uphods("GP-PO", metric_channel=False),
    "UPHODS-noHys": _uphods("UPHODS-noHys", hysteresis=False),
    "UPHODS-noET": _uphods("UPHODS-noET", elite_transfer=False),
    "UPHODS-fixedW": _uphods("UPHODS-fixedW", adaptive_weights=False),
    "UPHODS-noRN": _uphods("UPHODS-noRN", robust_norm=False),
    # v2: solver-level hybrid utility (see controller_v2.py)
    "UPHODS-v2": _v2("UPHODS-v2"),
    "GP-PO-v2": _v2("GP-PO-v2", metric_channel=False, signal="preference", estimator="discounted"),
    "UPHODS-v2-noHys": _v2("UPHODS-v2-noHys", hysteresis=False),
    "UPHODS-v2-noET": _v2("UPHODS-v2-noET", elite_transfer=False),
    "UPHODS-v2-fixedW": _v2("UPHODS-v2-fixedW", adaptive_weights=False),
    "UPHODS-v2-noRN": _v2("UPHODS-v2-noRN", robust_norm=False),
    # baselines
    "FSS-1": _plain(FSS1),
    "FSS-2": _plain(FSS2),
    "FSS-3": _plain(FSS3),
    "FSS-4": _plain(FSS4),
    "SW-UCB": _plain(SWUCB),
    "Single-GA": _plain(Single, "GA"),
    "Single-PSO": _plain(Single, "PSO"),
    "Single-SA": _plain(Single, "SA"),
    "Single-GD": _plain(Single, "GD"),
}

V1_METHODS = [m for m in METHODS if "v2" not in m]


def run_one(
    method: str,
    function: str,
    dim: int,
    seed: int,
    budget: int = BUDGET,
    overrides=None,
):
    problem = make_problem(function, dim)
    algo = METHODS[method](problem, budget, seed, overrides=overrides)
    return algo.run()
