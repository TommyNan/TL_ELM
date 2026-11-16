#!/usr/bin/env bash
# Ten repetitions by default: seeds 0 through 9, each with its own 10% split.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  printf '%s\n' 'No Python interpreter found. Activate your server environment or set PYTHON.' >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python interpreter is unavailable: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-9}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
CONFIG="${CONFIG:-configs/icassp.yml}"
# An unset WESN_MODE preserves model.prediction_mode from the selected YAML.
WESN_MODE="${WESN_MODE:-}"
OUT_ROOT="${OUT_ROOT:-runs/wesn-minmax-seeds-10pct}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

MODE_ARGS=()
if [[ -n "$WESN_MODE" ]]; then
  case "$WESN_MODE" in
    fspl_residual|direct_pl) MODE_ARGS=(--wesn-mode "$WESN_MODE") ;;
    *) printf '%s\n' 'WESN_MODE must be fspl_residual or direct_pl.' >&2; exit 2 ;;
  esac
fi

BASELINE_ARGS=()
case "${INCLUDE_BASELINES:-1}" in
  1) BASELINE_ARGS=(--include-baselines) ;;
  0) ;;
  *) printf '%s\n' 'INCLUDE_BASELINES must be 0 or 1.' >&2; exit 2 ;;
esac

if [[ -n "${RECEIVER_MASK:-}" ]]; then
  printf '%s\n' 'Coordinate/frequency experiments do not select receivers by material; RECEIVER_MASK is unsupported.' >&2
  exit 2
fi
if [[ ! "$SEED_START" =~ ^[0-9]+$ || ! "$SEED_END" =~ ^[0-9]+$ ]]; then
  printf '%s\n' 'SEED_START and SEED_END must be nonnegative integers.' >&2
  exit 2
fi
# Interpret user-supplied IDs as decimal even if they have leading zeroes.
SEED_START=$((10#$SEED_START))
SEED_END=$((10#$SEED_END))
if (( SEED_END < SEED_START )); then
  printf '%s\n' 'SEED_END must be at least SEED_START.' >&2
  exit 2
fi
# The script owns these flags so each saved seed and output path is unambiguous.
for arg in "$@"; do
  case "$arg" in
    --receiver-mask|--receiver-mask=*)
      printf '%s\n' 'Coordinate/frequency experiments do not select receivers by material; --receiver-mask is unsupported.' >&2
      exit 2 ;;
    --seed|--seed=*|--train-fraction|--train-fraction=*|--output-dir|--output-dir=*|--config|--config=*)
      printf '%s\n' 'Set seed range, fraction, config and output through environment variables, not forwarded flags.' >&2
      exit 2 ;;
    --inspect|--predict-checkpoint|--predict-checkpoint=*|--prediction-output|--prediction-output=*)
      printf '%s\n' 'This script runs training experiments; inspection or prediction flags are not accepted.' >&2
      exit 2 ;;
  esac
done
# Refuse an existing experiment directory before doing any work. In particular,
# do not aggregate an old seed folder together with newly generated runs.
for (( seed=SEED_START; seed<=SEED_END; seed++ )); do
  if [[ -e "$OUT_ROOT/seed_$seed" || -e "$OUT_ROOT/logs/seed_$seed.log" ]]; then
    printf 'Seed output already exists: %s\n' "$OUT_ROOT/seed_$seed" >&2
    exit 2
  fi
done
if [[ -e "$OUT_ROOT/seed_summary" ]]; then
  printf 'Seed summary output already exists: %s\n' "$OUT_ROOT/seed_summary" >&2
  exit 2
fi
mkdir -p -- "$OUT_ROOT/logs"
for (( seed=SEED_START; seed<=SEED_END; seed++ )); do
  printf 'Running seed %s (%s..%s), train_fraction=%s\n' "$seed" "$SEED_START" "$SEED_END" "$TRAIN_FRACTION"
  "$PYTHON_BIN" main.py --config "$CONFIG" "${BASELINE_ARGS[@]}" "${MODE_ARGS[@]}" "$@" \
    --seed "$seed" --train-fraction "$TRAIN_FRACTION" --output-dir "$OUT_ROOT/seed_$seed" \
    2>&1 | tee "$OUT_ROOT/logs/seed_$seed.log"
done
"$PYTHON_BIN" scripts/aggregate_seeds.py --root "$OUT_ROOT" \
  --seed-start "$SEED_START" --seed-end "$SEED_END"
