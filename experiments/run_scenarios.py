"""Run the mechanism scenarios of PROTOCOL.md section 10 in parallel.

Same conventions as run_grid.py (JSON lines, git revision, resumable, low
priority workers); the problem is built by experiments/scenarios.py.

    uv run python experiments/run_scenarios.py --scenario penalty --level 1e3 \
        --methods UPHODS-v2,UPHODS-v2-noRN --seeds 1000-1019 --out results/dev/scen/pen1e3.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_grid import _low_priority, git_rev, parse_seeds  # noqa: E402
from scenarios import make_scenario  # noqa: E402

from uphods.problems import FUNCTIONS  # noqa: E402
from uphods.runner import METHODS  # noqa: E402


def _job(args):
    method, scenario, level, fn, dim, seed, budget, cfg = args
    prob, extra = make_scenario(scenario, fn, dim, level)
    algo = METHODS[method](prob, budget, seed, overrides=cfg or None)
    r = algo.run()
    r.update(scenario=scenario, level=level, **extra(algo.ev.best_x))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["penalty", "discrete"])
    ap.add_argument("--level", type=float, required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--functions", default=",".join(FUNCTIONS))
    ap.add_argument("--dims", default="10,30")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) // 2 - 4))
    )
    a = ap.parse_args()
    cfg = json.loads(a.cfg) if a.cfg else {}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["method"], r["function"], r["dim"], r["seed"]))
    jobs = [
        (m, a.scenario, a.level, f, int(d), s, a.budget, cfg)
        for m in a.methods.split(",")
        for f in a.functions.split(",")
        for d in a.dims.split(",")
        for s in parse_seeds(a.seeds)
        if (m, f, int(d), s) not in done
    ]
    rev = git_rev()
    print(
        f"{a.scenario} {a.level:g}: {len(jobs)} runs to do ({len(done)} done), code {rev}",
        flush=True,
    )
    t0 = time.time()
    _low_priority()
    with (
        ProcessPoolExecutor(a.workers, initializer=_low_priority) as pool,
        out.open("a", encoding="utf-8") as fh,
    ):
        futs = [pool.submit(_job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            r.update(code=rev, tag=a.tag, cfg=cfg)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if k % max(1, len(jobs) // 10) == 0 or k == len(jobs):
                el = time.time() - t0
                print(
                    f"  {k}/{len(jobs)} {100 * k / len(jobs):.0f}%  {el:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
