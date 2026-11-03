"""Does the recommendation of UPHODS coincide with its best evaluated design?

UPHODS recommends the evaluated design with the highest posterior utility
u(x) = s(x) + mu_r(x) (Algorithm 1), while the benchmark tables report, for
every method, the error of the best design evaluated in a run.  This script
replays the reported UPHODS runs (frozen code, same seeds, hence identical
runs) and records, per run, the error of the recommended design and the rank
of the lowest-error design under u.  Nothing in uphods/ is changed.

    uv run python experiments/check_recommendation.py results/final/recommendation.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import numpy as np  # noqa: E402

from run_grid import _low_priority, git_rev  # noqa: E402
from uphods.problems import FUNCTIONS, make_problem  # noqa: E402
from uphods.runner import METHODS  # noqa: E402


def job(args):
    fn, d, seed = args
    algo = METHODS["UPHODS-v2"](make_problem(fn, d), 1000, seed)
    r = algo.run()
    f = algo.f_all()
    u = algo.utility_all()
    rec = int(np.argmax(u))
    best = int(np.argmin(f))
    rank_best = int(np.sum(u > u[best])) + 1  # 1 = highest utility
    return dict(
        function=fn,
        dim=d,
        seed=seed,
        best=r["best"],
        rec_err=float(f[rec]),
        rec_is_best=bool(f[rec] == f[best]),
        rank_of_best=rank_best,
        in_top10=bool(rank_best <= 10),
    )


def main():
    out = Path(sys.argv[1])
    jobs = [
        (fn, d, s)
        for fn in FUNCTIONS
        for d in (5, 10, 20, 30, 50, 100)
        for s in range(30)
    ]
    _low_priority()
    rev = git_rev()
    with (
        ProcessPoolExecutor(12, initializer=_low_priority) as pool,
        out.open("w", encoding="utf-8") as fh,
    ):
        for k, fu in enumerate(as_completed([pool.submit(job, j) for j in jobs]), 1):
            r = fu.result()
            r["code"] = rev
            fh.write(json.dumps(r) + "\n")
            if k % 108 == 0:
                print(f"{k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
