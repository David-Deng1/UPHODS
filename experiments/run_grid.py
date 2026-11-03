"""Run a grid of (method, function, dimension, seed) in parallel.

Results are appended to a JSON-lines file, one line per run, together with the
git commit of the code, so every number can be traced to a code version and a
seed.  Already finished runs are skipped, so an interrupted grid can resume.

Example:
    uv run python experiments/run_grid.py --seeds 1000-1009 --out results/dev/v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uphods.problems import DIMS, FUNCTIONS  # noqa: E402
from uphods.runner import METHODS, run_one  # noqa: E402


def git_rev():
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "uphods"]
        )
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _low_priority():
    """Run workers below normal priority so the machine stays responsive."""
    try:
        if os.name == "nt":
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:
        pass


def _job(args):
    method, fn, dim, seed, budget, cfg = args
    return run_one(method, fn, dim, seed, budget, overrides=cfg or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--functions", default=",".join(FUNCTIONS))
    ap.add_argument("--dims", default=",".join(map(str, DIMS)))
    ap.add_argument("--seeds", default="1000-1009")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument(
        "--cfg",
        default="",
        help="JSON dict of config overrides (UPHODS config fields, or baseline keyword args)",
    )
    ap.add_argument("--tag", default="", help="label stored with each result")
    ap.add_argument("--out", required=True)
    # default: well under half of the logical cores, leaving room for other work
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
        (m, f, int(d), s, a.budget, cfg)
        for m in a.methods.split(",")
        for f in a.functions.split(",")
        for d in a.dims.split(",")
        for s in parse_seeds(a.seeds)
        if (m, f, int(d), s) not in done
    ]
    rev = git_rev()
    print(f"{len(jobs)} runs to do ({len(done)} already done), code {rev}", flush=True)
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
            if k % max(1, len(jobs) // 20) == 0 or k == len(jobs):
                el = time.time() - t0
                bar = "#" * int(20 * k / len(jobs))
                print(
                    f"[{bar:<20}] {k}/{len(jobs)} {100 * k / len(jobs):.0f}%  "
                    f"{el:.0f}s elapsed, ~{el / k * (len(jobs) - k):.0f}s left",
                    flush=True,
                )


if __name__ == "__main__":
    main()
