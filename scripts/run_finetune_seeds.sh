#!/usr/bin/env bash
# Fine-tuning label budgets on a fixed test set; seeds 0 through 9 by default.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/icassp.yml}"
OUT_ROOT="${OUT_ROOT:-runs/finetune-budget}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-9}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  exec "$PYTHON" scripts/finetune_budget.py "$@"
fi

if (( SEED_START < 0 || SEED_END < SEED_START )); then
  printf '%s\n' 'Choose a nonnegative seed range with SEED_END >= SEED_START.' >&2
  exit 2
fi
if [[ -e "$OUT_ROOT/seed_summary" ]]; then
  printf 'Seed summary already exists: %s\n' "$OUT_ROOT/seed_summary" >&2
  exit 2
fi

mkdir -p -- "$OUT_ROOT/logs"
for (( seed=SEED_START; seed<=SEED_END; seed++ )); do
  if [[ -e "$OUT_ROOT/seed_$seed" || -e "$OUT_ROOT/logs/seed_$seed.log" ]]; then
    printf 'Seed output already exists: %s\n' "$OUT_ROOT/seed_$seed" >&2
    exit 2
  fi
  printf 'Fine-tuning budgets: seed %s (%s..%s), training pool=%s\n' \
    "$seed" "$SEED_START" "$SEED_END" "$TRAIN_FRACTION"
  "$PYTHON" scripts/finetune_budget.py --config "$CONFIG" "$@" \
    --seed "$seed" --train-fraction "$TRAIN_FRACTION" --output-dir "$OUT_ROOT/seed_$seed" \
    2>&1 | tee "$OUT_ROOT/logs/seed_$seed.log"
done

"$PYTHON" scripts/aggregate_finetune_budget.py --root "$OUT_ROOT" \
  --seed-start "$SEED_START" --seed-end "$SEED_END"
