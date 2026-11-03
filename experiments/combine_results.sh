#!/usr/bin/env bash
# Rebuild the combined result files used by summarize.py (they are not stored in the repository).
# Usage: bash experiments/combine_results.sh [results/final]
set -e
R=${1:-results/final}
cat "$R/main_uphods.jsonl" "$R/main_base.jsonl" "$R/main_swucb.jsonl" > "$R/main_all.jsonl"
{
  grep '"method": "UPHODS-v2"' "$R/main_uphods.jsonl"
  grep '"method": "Single-' "$R/main_base.jsonl"
  cat "$R/ctrl_fss.jsonl" "$R/ctrl_swucb.jsonl"
} > "$R/ctrl_all.jsonl"
grep '"method": "GP-PO-v2"' "$R/main_uphods.jsonl" > "$R/gppo.jsonl"
