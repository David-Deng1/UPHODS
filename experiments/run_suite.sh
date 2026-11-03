#!/usr/bin/env bash
# Full evaluation suite for one seed set.  Usage: run_suite.sh <seeds> <outdir> <tag>
#   validation: run_suite.sh 1020-1039 results/val val
#   final:      run_suite.sh 0-29      results/final final
set -e
SEEDS=$1; OUT=$2; TAG=$3
mkdir -p $OUT
G="uv run python experiments/run_grid.py --seeds $SEEDS --tag $TAG"
UP="UPHODS-v2,GP-PO-v2,UPHODS-v2-noHys,UPHODS-v2-noET,UPHODS-v2-fixedW,UPHODS-v2-noRN"
SINGLES="Single-GA,Single-PSO,Single-SA,Single-GD"
SW='"c": 0.25, "initial": "GA"'
# 1) main table: methods as originally defined
$G --methods $UP --out $OUT/main_uphods.jsonl
$G --methods FSS-1,FSS-2,FSS-3,FSS-4,$SINGLES --cfg '{"handover": "incumbent"}' --out $OUT/main_base.jsonl
$G --methods SW-UCB --cfg "{$SW, \"handover\": \"incumbent\"}" --out $OUT/main_swucb.jsonl
# 2) controlled table: all hybrids with the elite-archive transfer
$G --methods FSS-1,FSS-2,FSS-3,FSS-4 --cfg '{"handover": "archive"}' --out $OUT/ctrl_fss.jsonl
$G --methods SW-UCB --cfg "{$SW, \"handover\": \"archive\"}" --out $OUT/ctrl_swucb.jsonl
# 3) misaligned-preference scenario (PROTOCOL 8)
MIS="--functions Ackley,Griewank,Rastrigin,Rosenbrock,ShiftedRastrigin,ShiftedSphere --dims 10,30"
$G --methods $UP $MIS --cfg '{"dm_h_lambda": 10.0}' --out $OUT/mis_uphods.jsonl
$G --methods FSS-1,FSS-2,FSS-3,FSS-4,$SINGLES $MIS --cfg '{"handover": "incumbent", "dm_h_lambda": 10.0}' --out $OUT/mis_base.jsonl
$G --methods SW-UCB $MIS --cfg "{$SW, \"handover\": \"incumbent\", \"dm_h_lambda\": 10.0}" --out $OUT/mis_swucb.jsonl
echo SUITE_DONE
