#!/usr/bin/env bash
# ICASSP baselines only: every selected scene, seeds 0 through 9 by default.
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
  printf '%s\n' 'No Python interpreter found. Activate your environment or set PYTHON.' >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python interpreter is unavailable: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi

RUN_MODULE="baselines.compare_baselines"
CONFIG="${CONFIG:-configs/icassp.yml}"
OUT_ROOT="${OUT_ROOT:-${ROOT_OUTPUT_DIR:-runs/baseline-comparison-10pct}}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-9}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Ask the Python entry point for help once, without starting a sweep.
for arg in "$@"; do
  if [[ "$arg" == '--help' || "$arg" == '-h' ]]; then
    exec "$PYTHON_BIN" -m "$RUN_MODULE" "$@"
  fi
done

if [[ -n "${RECEIVER_MASK:-}" ]]; then
  printf '%s\n' 'Material receiver filtering is unsupported; unset RECEIVER_MASK.' >&2
  exit 2
fi
if [[ ! "$SEED_START" =~ ^[0-9]+$ || ! "$SEED_END" =~ ^[0-9]+$ ]]; then
  printf '%s\n' 'SEED_START and SEED_END must be nonnegative integers.' >&2
  exit 2
fi
SEED_START=$((10#$SEED_START))
SEED_END=$((10#$SEED_END))
if (( SEED_START < 0 || SEED_END < SEED_START )); then
  printf '%s\n' 'SEED_END must be at least SEED_START; seed IDs must be nonnegative.' >&2
  exit 2
fi

# These flags belong to the sweep, so saved seed IDs and paths cannot diverge.
for arg in "$@"; do
  case "$arg" in
    --seed|--seed=*|--seed-start|--seed-start=*|--seed_start|--seed_start=*|--seed-end|--seed-end=*|--seed_end|--seed_end=*|\
    --config|--config=*|--train-fraction|--train-fraction=*|--train_fraction|--train_fraction=*|--train-ratio|--train-ratio=*|--train_ratio|--train_ratio=*|\
    --output-dir|--output-dir=*|--output_dir|--output_dir=*|--root|--root=*|--aggregate-only|--aggregate_only|--aggregate-only=*|--aggregate_only=*)
      printf '%s\n' 'Set config, seed range, training fraction and output through CONFIG, SEED_START/SEED_END, TRAIN_FRACTION and OUT_ROOT.' >&2
      exit 2 ;;
  esac
done

for (( seed=SEED_START; seed<=SEED_END; seed++ )); do
  if [[ -e "$OUT_ROOT/seed_$seed" || -e "$OUT_ROOT/logs/seed_$seed.log" ]]; then
    printf 'Seed output already exists: %s\n' "$OUT_ROOT/seed_$seed" >&2
    exit 2
  fi
done
if [[ -e "$OUT_ROOT/sweep_summary" ]]; then
  printf 'Sweep summary already exists: %s\n' "$OUT_ROOT/sweep_summary" >&2
  exit 2
fi

mkdir -p -- "$OUT_ROOT/logs"
for (( seed=SEED_START; seed<=SEED_END; seed++ )); do
  printf 'Baselines only: seed %s (%s..%s), train_fraction=%s\n' "$seed" "$SEED_START" "$SEED_END" "$TRAIN_FRACTION"
  "$PYTHON_BIN" -m "$RUN_MODULE" --config "$CONFIG" "$@" \
    --seed "$seed" --train-fraction "$TRAIN_FRACTION" --output-dir "$OUT_ROOT/seed_$seed" \
    2>&1 | tee "$OUT_ROOT/logs/seed_$seed.log"
done

"$PYTHON_BIN" -m "$RUN_MODULE" --aggregate-only --root "$OUT_ROOT" \
  --seed-start "$SEED_START" --seed-end "$SEED_END"
