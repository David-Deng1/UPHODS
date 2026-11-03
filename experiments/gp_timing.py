"""GP fit time versus the number of pairwise comparisons (PROTOCOL.md 9.3).

Pairs come from rankings of 5-design slates, as in the default query, and the
number of compared designs is 0.43 times the number of pairs (the ratio seen in
the main runs).  Two timings per setting:

  cold  - EP from scratch on all pairs (upper bound),
  warm  - EP warm-started from the sites of the previous fit after one new slate
          (10 new pairs), which is what happens inside a run.

Single-threaded BLAS, as in the experiment grid.  Median of ``--reps`` repeats.

    uv run python experiments/gp_timing.py --out results/final/gp_timing.json
"""

from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uphods.pref_gp import PreferenceGP  # noqa: E402

SLATE = 5
PAIRS_PER_SLATE = SLATE * (SLATE - 1) // 2
RATIO = 0.43


def make_data(n_pairs, dim, rng):
    n_items = max(SLATE, int(round(RATIO * n_pairs)))
    X = rng.random((n_items, dim))
    target = rng.random(dim)
    util = -np.sum((X - target) ** 2, axis=1) / dim
    n_slates = int(np.ceil(n_pairs / PAIRS_PER_SLATE))
    # every design is shown at least once (first a permutation of all designs),
    # remaining slots are random; no design twice in one slate
    slots = np.concatenate(
        [
            rng.permutation(n_items),
            rng.integers(0, n_items, max(0, SLATE * n_slates - n_items)),
        ]
    )
    pairs = []
    for s in range(n_slates):
        ids = list(dict.fromkeys(slots[SLATE * s : SLATE * (s + 1)].tolist()))
        while len(ids) < SLATE:
            c = int(rng.integers(n_items))
            if c not in ids:
                ids.append(c)
        ids = np.array(ids)
        order = ids[np.argsort(-(util[ids] + 0.05 * rng.standard_normal(SLATE)))]
        for a in range(SLATE):
            for b in range(a + 1, SLATE):
                pairs.append((order[a], order[b]))
    pairs = np.array(pairs[:n_pairs])
    # only designs that appear in a comparison enter the GP
    used = np.unique(pairs)
    remap = -np.ones(n_items, dtype=int)
    remap[used] = np.arange(len(used))
    return X[used], remap[pairs]


def time_fit(X, pairs, dim):
    keys = [(int(a), int(b), k) for k, (a, b) in enumerate(pairs)]
    gp = PreferenceGP(dim)
    t0 = time.perf_counter()
    gp.fit(X, pairs, pair_keys=keys)
    cold = time.perf_counter() - t0
    # warm start: refit after one more slate's worth of pairs
    m = len(pairs) - PAIRS_PER_SLATE
    gp2 = PreferenceGP(dim)
    sub = pairs[:m]
    used = np.unique(sub)
    remap = -np.ones(len(X), dtype=int)
    remap[used] = np.arange(len(used))
    gp2.fit(X[used], remap[sub], pair_keys=keys[:m])
    t0 = time.perf_counter()
    gp2.fit(X, pairs, pair_keys=keys)
    warm = time.perf_counter() - t0
    return cold, warm, len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="100,250,500,1000,2000,4000")
    ap.add_argument("--dims", default="10,100")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for dim in map(int, a.dims.split(",")):
        for n in map(int, a.pairs.split(",")):
            cold, warm, items = [], [], 0
            for r in range(a.reps):
                X, P = make_data(n, dim, np.random.default_rng([n, dim, r]))
                c, w, items = time_fit(X, P, dim)
                cold.append(c)
                warm.append(w)
            row = dict(
                dim=dim,
                n_pairs=n,
                n_designs=items,
                cold_median_s=float(np.median(cold)),
                warm_median_s=float(np.median(warm)),
                cold_all=cold,
                warm_all=warm,
            )
            rows.append(row)
            print(
                f"D={dim:3d} pairs={n:5d} designs={items:5d} "
                f"cold={row['cold_median_s']:.3f}s warm={row['warm_median_s']:.3f}s",
                flush=True,
            )
    info = dict(
        machine=platform.machine(),
        processor=platform.processor(),
        system=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        logical_cpus=os.cpu_count(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(dict(info=info, rows=rows), indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
