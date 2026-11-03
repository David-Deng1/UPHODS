#!/usr/bin/env bash
# Supplementary experiments of PROTOCOL.md section 9 (frozen code, final seeds).
# Usage: run_supplement.sh <seeds> <outdir> <tag>
#   final: run_supplement.sh 0-29 results/final final
set -e
SEEDS=$1; OUT=$2; TAG=$3
mkdir -p $OUT/noise $OUT/sens $OUT/budget
G="uv run python experiments/run_grid.py --seeds $SEEDS --tag $TAG"
PREF="UPHODS-v2,GP-PO-v2"
MIS="--functions Ackley,Griewank,Rastrigin,Rosenbrock,ShiftedRastrigin,ShiftedSphere --dims 10,30"
SW='"c": 0.25, "initial": "GA"'

# 9.1 preference noise: flip probability p on top of the Thurstone noise
for P in 0.1 0.2 0.3; do
  $G --methods $PREF --cfg "{\"dm_flip\": $P}" --out $OUT/noise/main_p$P.jsonl
  $G --methods $PREF $MIS --cfg "{\"dm_flip\": $P, \"dm_h_lambda\": 10.0}" --out $OUT/noise/mis_p$P.jsonl
done

# 9.2 one-at-a-time sensitivity of UPHODS-v2 (defaults come from the main table)
SENS="--methods UPHODS-v2 --dims 10,30,100"
while read KEY VALUES; do
  for V in $VALUES; do
    $G $SENS --cfg "{\"$KEY\": $V}" --out $OUT/sens/${KEY}_$V.jsonl
  done
done <<'EOF'
kappa 0.5 2.0
eps 0.1 0.4
cooldown_evals 5 20 40
phase_half_life 100.0 400.0
lam_p 0.25 1.0
lam_m 0.25 1.0
inject_ratio 0.2 0.7
archive_delta 0.02 0.1
weight_window 15 60
slate_size 3 8
lengthscale 0.25 1.0
EOF

# 9.4 budget extension (about 5x the preference data of the main runs)
BUD="--budget 5000 --dims 30,100"
$G $BUD --methods UPHODS-v2 --out $OUT/budget/uphods.jsonl
$G $BUD --methods FSS-1,FSS-2,FSS-3,FSS-4,Single-GA,Single-PSO,Single-SA,Single-GD --cfg '{"handover": "incumbent"}' --out $OUT/budget/base.jsonl
$G $BUD --methods SW-UCB --cfg "{$SW, \"handover\": \"incumbent\"}" --out $OUT/budget/swucb.jsonl

# 9.3 GP timing benchmark (single process, no seeds involved)
uv run python experiments/gp_timing.py --out $OUT/gp_timing.json
echo SUPPLEMENT_DONE
