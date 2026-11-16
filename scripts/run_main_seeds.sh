#!/usr/bin/env bash
# Scratch, Transfer and Zero-shot only; seeds 0 through 9 by default.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${WESN_MODE:-fspl_residual}"
args=()
while (( $# )); do
  case "$1" in
    --mode)
      if (( $# < 2 )); then
        printf '%s\n' '--mode requires fspl_residual or direct_pl.' >&2
        exit 2
      fi
      mode="$2"
      shift 2 ;;
    --mode=*) mode="${1#*=}"; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash scripts/run_main_seeds.sh [--mode fspl_residual|direct_pl] [main.py options]

Runs Scratch, Transfer and Zero-shot on every scene, with seeds 0-9.
The default mode is fspl_residual; WESN_MODE can also select the mode.
Use --mode to override WESN_MODE. No baselines are trained.

Examples:
  bash scripts/run_main_seeds.sh --mode fspl_residual --device cuda
  bash scripts/run_main_seeds.sh --mode direct_pl --device cuda
  bash scripts/run_main_seeds.sh --mode direct_pl --scenes Scene10 --device cpu \
    --pretrain-steps 2 --target-steps 2 --max-eval-points 16

Environment: PYTHON, CONFIG, TRAIN_FRACTION (default 0.1), OUT_ROOT,
             SEED_START (default 0), SEED_END (default 9), WESN_MODE.
Default output: runs/main-<mode>-minmax-train-<TRAIN_FRACTION>/
Results: seed_N/, logs/seed_N.log, seed_summary/{summary.json,per_seed.csv}.
USAGE
      exit 0 ;;
    --include-baselines|--include-baselines=*|--baseline-only|--baseline-only=*|--baselines|--baselines=*|--methods|--methods=*|--wesn-mode|--wesn-mode=*|--prediction-mode|--prediction-mode=*)
      printf '%s\n' 'This script runs the three main experiments only; select their mode with --mode.' >&2
      exit 2 ;;
    *) args+=("$1"); shift ;;
  esac
done

case "$mode" in
  fspl_residual|direct_pl) ;;
  *) printf '%s\n' '--mode must be fspl_residual or direct_pl.' >&2; exit 2 ;;
esac

export WESN_MODE="$mode"
export TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
export OUT_ROOT="${OUT_ROOT:-runs/main-${mode}-minmax-train-${TRAIN_FRACTION}}"
export INCLUDE_BASELINES=0
exec bash "$SCRIPT_DIR/run_seeds.sh" \
  --methods E2_scratch E3_transfer E4_zero_shot "${args[@]}"
