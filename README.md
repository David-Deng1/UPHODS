# UPHODS: code and benchmark results

This repository contains the implementation and the benchmark results of

> R. Deng, L. Jia, K. Shen, J. Hao. *User Preference-Driven Hybrid Optimization with Dynamic Switching for Engineering Design.* Manuscript under review at *Applied Soft Computing*.

UPHODS is an online controller that switches among complete, unmodified solvers (GA, PSO, SA and gradient descent) during a run. Each switching decision uses a phase-adjusted solver utility estimated by discounted Bayesian regression, with an optimism term, a margin and a cooldown (hysteresis). Designs are ranked by a two-level utility, an objective score plus a probit Gaussian-process residual learned from the designer's pairwise preferences. An elite archive transfers good designs to the incoming solver.

The vehicle-body case study of the paper is **not** part of this repository: its models and data are subject to third-party restrictions. Everything else reported in Section 4 of the paper (benchmark experiments) and in the Supplementary Material can be reproduced from this repository.

## Contents

| Path | Content |
|---|---|
| `uphods/` | Method and baselines. `controller_v2.py` is the method described in the paper; `pref_gp.py` the preference model (EP inference); `archive.py` the elite archive; `baselines.py` the fixed switching schedules FSS-1 to FSS-4, the sliding-window UCB selector SW-UCB and the single solvers; `oracle.py` and `misaligned.py` the simulated designer; `problems.py` the benchmark functions. |
| `tests/` | Unit tests (`uv run pytest`). |
| `experiments/` | Scripts that run the experiments and produce every table and figure from the raw results. |
| `results/final/` | Raw results (one JSON line per run) and the generated tables and figures. |

`uphods/` is identical to the code that produced the results (tag `v2.0-frozen`, commit `a13a68d`, of the development repository). Each result line stores in its `code` field the development-repository commit at run time; commits after `a13a68d` changed only analysis scripts, never `uphods/`.

Code comments that mention the "0201 version" refer to the originally submitted manuscript (February 2026), whose description is also implemented (`controller.py`, methods without the `-v2` suffix) for reference. Some comments refer to sections of an internal experiment protocol that is not included here.

## Method names

| Name in the code | Name in the paper |
|---|---|
| `UPHODS-v2` | UPHODS |
| `UPHODS-v2-noHys`, `-noET`, `-fixedW`, `-noRN` | ablations: no hysteresis, no elite transfer, fixed metric weights, no robust normalisation |
| `GP-PO-v2` | GP-PO (preference-only controller) |
| `FSS-1` ... `FSS-4`; FSS* | fixed switching schedules; FSS* is the best of the four on each instance |
| `Single-GA`, `-PSO`, `-SA`, `-GD`; Single* | single solvers; Single* is the best of the four on each instance |
| `SW-UCB` | sliding-window UCB algorithm selector |

## Requirements

Python 3.12 or later and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

```bash
uv sync
uv run pytest
```

## Reproducing the results

The reported runs use seeds 0-29. Default settings were chosen in preliminary runs whose seeds (1000 and above) are disjoint from these. All runs are deterministic given the seed. The full set comprises tens of thousands of runs and takes many hours; `run_grid.py` uses up to 12 worker processes (`--workers`).

**1. Run the experiments** (writes JSON lines into `results/final/`):

```bash
bash experiments/run_suite.sh 0-29 results/final final            # main, controlled and misaligned-preference comparisons
bash experiments/run_supplement.sh 0-29 results/final final       # preference noise, sensitivity, budget 5000, GP timing
bash experiments/run_scenarios_suite.sh 0-29 results/final/scen final   # penalty-constraint and discrete-variable scenarios
uv run python experiments/check_recommendation.py results/final/recommendation.jsonl
uv run python experiments/diag_metric_term.py 0 30 > results/final/diag_metric_term.md
```

A single run: `uv run python experiments/run_grid.py --methods UPHODS-v2 --functions Ackley --dims 10 --seeds 0 --out test.jsonl`.

**2. Generate the tables and figures** from the raw results:

```bash
bash experiments/combine_results.sh results/final
uv run python experiments/paper_tables.py results/final > results/final/paper_tables.md
uv run python experiments/supplementary_tables.py results/final > results/final/supplementary_tables.md
uv run python experiments/supplement_report.py results/final > results/final/supplement.summary.md
uv run python experiments/scenario_report.py results/final/scen > results/final/scen/summary.md
uv run --with matplotlib python experiments/make_figures.py results/final
uv run python experiments/summarize.py results/final/main_all.jsonl --target UPHODS-v2
uv run python experiments/summarize.py results/final/ctrl_all.jsonl --target UPHODS-v2
uv run python experiments/ablation.py results/final/main_uphods.jsonl > results/final/ablation_main.txt
uv run python experiments/posthoc_final.py > results/final/posthoc.txt
uv run python experiments/summarize_misaligned.py results/final/mis_uphods.jsonl results/final/mis_base.jsonl results/final/mis_swucb.jsonl > results/final/mis.summary.txt
```

`combine_results.sh` concatenates the per-group result files into `main_all.jsonl`, `ctrl_all.jsonl` and `gppo.jsonl`, which are not stored in the repository. Step 2 alone takes one to two minutes, so the tables and figures can be checked against the stored raw results without rerunning the experiments.

## Where the paper's numbers come from

| Paper | File |
|---|---|
| Section 4 comparison tables (main and controlled comparison, results by dimension, ablations, misaligned preferences, switching frequency and preference-model cost) | `results/final/paper_tables.md` |
| Preference noise, sensitivity, budget 5000, GP fit time | `results/final/supplement.summary.md` |
| Penalty-constraint and discrete-variable scenarios | `results/final/scen/summary.md` |
| Size of the metric correction term | `results/final/diag_metric_term.md` |
| Recommendation versus best evaluated design | `results/final/recommendation.jsonl` |
| Supplementary Tables S1-S3 | `results/final/supplementary_tables.md`, `results/final/tableS1_per_instance.csv` |
| Figures 4-8 and Supplementary Fig. S1 | `results/final/figures/` |

## Format of a result line

Each line of a `.jsonl` file is one run: `method`, `function`, `dim`, `seed`, `best` (lowest error found), `evals`, `curve` (best error after every 20 evaluations), `switches` and `usage` (number of switches and evaluations per solver), per-method diagnostics such as the solver estimates (`est`) and the preference-model cost (`gp_seconds`, `gp_max_seconds`), `code` (commit), `tag` and `cfg` (configuration overrides).

## Use of AI tools

The code was written with the assistance of an AI coding tool (Claude, Anthropic) under the authors' direction. The authors specified the method and the experimental design and reviewed the code, the unit tests and the results.

## License

MIT, see `LICENSE`.
