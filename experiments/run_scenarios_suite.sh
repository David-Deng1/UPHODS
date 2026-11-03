#!/usr/bin/env bash
# All settings of PROTOCOL.md section 10.  Usage: run_scenarios_suite.sh <seeds> <outdir> <tag>
#   development check: run_scenarios_suite.sh 1000-1019 results/dev/scen dev
#   final:             run_scenarios_suite.sh 0-29      results/final/scen final
set -e
SEEDS=$1; OUT=$2; TAG=$3
mkdir -p $OUT
G="uv run python experiments/run_scenarios.py --seeds $SEEDS --tag $TAG"
UP="UPHODS-v2,GP-PO-v2,UPHODS-v2-noHys,UPHODS-v2-noET,UPHODS-v2-fixedW,UPHODS-v2-noRN"
BASE="FSS-1,FSS-2,FSS-3,FSS-4,Single-GA,Single-PSO,Single-SA,Single-GD"
SW='"c": 0.25, "initial": "GA"'
for S in "penalty 1000" "penalty 1000000" "discrete 10" "discrete 30"; do
  set -- $S; SC=$1; LV=$2; F=$OUT/${SC}_${LV}
  $G --scenario $SC --level $LV --methods $UP --out ${F}_uphods.jsonl
  $G --scenario $SC --level $LV --methods $BASE --cfg '{"handover": "incumbent"}' --out ${F}_base.jsonl
  $G --scenario $SC --level $LV --methods SW-UCB --cfg "{$SW, \"handover\": \"incumbent\"}" --out ${F}_swucb.jsonl
done
echo SCEN_DONE
