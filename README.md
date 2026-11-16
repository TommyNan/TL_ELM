# ICASSP Cross-Frequency Path-Loss Prediction

This project compares spatial baselines and NeRF-WESN transfer learning on 24 paired radio-map scenes from 15 buildings. Each scene is trained and evaluated independently. The default experiment uses 10% of valid receiver locations for training and the remaining 90% for testing, with no validation or separate support set. Repetitions use seeds 0 through 9.

## Methods

| Method | Experiment ID | Training labels |
| --- | --- | --- |
| FSPL | `E1_fspl` | None; analytical free-space path loss |
| KNN | `B1_knn` | Target-band PL at the shared training locations |
| MLP | `B2_mlp` | Target-band PL at the shared training locations |
| NeRF2 | `B3_nerf2` | Target-band PL at the shared training locations |
| NeRF-WESN Scratch | `E2_scratch` | Target-band training labels only |
| NeRF-WESN Transfer | `E3_transfer` | Source pretraining, followed by target fine-tuning |
| NeRF-WESN Zero-shot | `E4_zero_shot` | The same source model, without target fine-tuning |

The source band is **f2 = 1.8 GHz** and the target band is **f3 = 3.5 GHz**. Source and target training coordinates are identical. All methods use the same held-out test locations. FSPL consumes no PL labels; the other baselines and Scratch use the same target-label budget. Transfer additionally uses source labels and pretraining computation.

`MLP` takes seven features: Rx xyz, Tx xyz, and `log(f_GHz / 1 GHz)`. Its x/y coordinates are normalized by the scene spans, while z remains in metres. KNN uses a uniform mean of the nearest `min(k, N_train)` raw PL observations, with `k=5` by default. MLP and NeRF2 directly predict complete PL, without subtracting or adding FSPL.

The standalone `NeRF2` implementation is entirely in `baselines/nerf2.py`; it does not import `Models`. Its attenuation MLP defaults to eight hidden layers of width 256, with an input skip after zero-based layer 4.

The ray models are **NeRF2-inspired adaptations for scalar PL**, not exact reproductions of the original RSSI/CSI pipeline. The current renderer coherently sums complex contributions within each ray, then averages power across rays and converts the result to dB. The official NeRF2 RSSI renderer instead sums complex contributions across directions before taking the magnitude. PL supervision alone does not uniquely identify latent phase or CSI.

## WESN Training Modes

Select the mode in `configs/icassp.yml`:

```yaml
model:
  attenuation: wesn
  prediction_mode: fspl_residual  # or direct_pl
```

- `fspl_residual`: predict `FSPL(d, f) + learned_residual_db`. The residual may be positive or negative.
- `direct_pl`: directly predict complete PL, without calculating or adding FSPL.

Both modes use the original complete PL labels and MSE loss in dB squared. The source model, Scratch, Transfer, and Zero-shot use the same mode within a run. Transfer fine-tunes a separate copy of the source weights; Zero-shot retains the unchanged source model. The WESN reservoir stays fixed. Checkpoints save the model settings and restore the selected mode automatically.

## Data and Coordinates

Only Tx/Rx coordinates, carrier frequency, and their deterministic encodings are model inputs. Material images and measured-label context are not used.

The dataset directory contains two metadata CSV files and paired PL images:

```text
datasets/ICASSP2025_Paired_Scenes/
  positions.csv
  building_details.csv
  Scene01/
    f2_path_loss.png
    f3_path_loss.png
  ...
  Scene24/
    f2_path_loss.png
    f3_path_loss.png
```

PL labels retain their original quantized dB values; they are not divided by 255 or treated as RSSI. The PNG width and height determine the receiver grid:

- `x = column * 0.25 m`, `y = row * 0.25 m`.
- Tx coordinates are read directly from the already-localized `positions.csv`.
- Rx height equals the Tx z coordinate; z is the antenna height, not the map's y span.
- Negative original building bounds are metadata; they do not shift Tx/Rx again.
- Only actual coincident Tx/Rx locations are excluded. There is no material-based or PL-value filtering.
- FSPL uses the true positive Euclidean distance, without a 0.25 m or 1 m distance floor.

For each scene and seed, the training count is `floor(N_valid * train_fraction)`. All remaining valid locations form the test set.

## Repository Layout

```text
Models/                 WESN attenuation and NeRF-WESN ray model
baselines/              FSPL, KNN, MLP, standalone NeRF2, comparison scripts
configs/icassp.yml      Dataset, model, optimizer, and split settings
datasets/               Paired PL maps, metadata, and dataset documentation
scripts/                Seed aggregation and saved-model visualization
tests/                  Model, data, runner, and reporting tests
main.py                 Training, inspection, and checkpoint prediction
propagation.py          Frequency features and FSPL formula
experiment_utils.py     Metrics, timing helpers, and reproducibility utilities
requirements-icassp.txt  Python dependencies
```

Experiment outputs are created under the chosen output directory and are not part of the source tree.

## Installation

Use Python 3.12. For a GPU server, install a PyTorch 2.5.1 CUDA wheel compatible with its NVIDIA driver before installing the remaining dependencies. See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v251).

For example, on a server compatible with CUDA 12.4:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-icassp.txt
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For CPU-only use, install PyTorch from `https://download.pytorch.org/whl/cpu` instead and pass `--device cpu` to the commands below. Run Python commands from the repository root.

## Checks and Short Runs

Inspect all scenes and run the tests:

```bash
python main.py --config configs/icassp.yml --inspect
python -m unittest discover -s tests -v
bash -n scripts/run_seeds.sh baselines/sweep_baselines.sh
```

Check both WESN modes on one GPU scene, including all four baselines:

```bash
for mode in fspl_residual direct_pl; do
  CUDA_VISIBLE_DEVICES=0 python main.py --scenes Scene10 --include-baselines \
    --wesn-mode "$mode" --pretrain-steps 2 --target-steps 2 \
    --max-eval-points 32 --device cuda --output-dir "runs/smoke-$mode"
done
```

Short runs verify the execution path, not convergence or prediction accuracy. Formal experiments use `max_eval_points: 0` to score the complete held-out test set. The default training budget is 2000 optimizer steps per stage, not epochs; set the budget before evaluating results. Output directories must be new.

The final source was checked with Python 3.12 and PyTorch 2.5.1 on CPU: 86 automated tests passed, all 24 scenes loaded, and both WESN modes plus the baseline-only workflow completed short runs on two scenes with seeds 0-9. Checkpoint prediction, seed aggregation, and nine-panel visualization were also verified. These checks do not constitute a converged experiment or a CUDA benchmark.

## Run Experiments

Run one seed with all seven methods:

```bash
python main.py --include-baselines --seed 0 --train-fraction 0.1 \
  --wesn-mode fspl_residual --device cuda --output-dir runs/residual-seed0
python main.py --include-baselines --seed 0 --train-fraction 0.1 \
  --wesn-mode direct_pl --device cuda --output-dir runs/direct-seed0
```

Run all 24 scenes with seeds **0 through 9**, separately for each WESN mode:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python WESN_MODE=fspl_residual \
  TRAIN_FRACTION=0.1 OUT_ROOT=runs/wesn-residual-10pct \
  bash scripts/run_seeds.sh --device cuda

CUDA_VISIBLE_DEVICES=0 PYTHON=python WESN_MODE=direct_pl \
  TRAIN_FRACTION=0.1 OUT_ROOT=runs/wesn-direct-10pct \
  bash scripts/run_seeds.sh --device cuda
```

Compare only the four baselines, without WESN or source pretraining:

```bash
python -m baselines.compare_baselines --seed 0 --device cuda \
  --output-dir runs/baselines-seed0

CUDA_VISIBLE_DEVICES=0 PYTHON=python TRAIN_FRACTION=0.1 \
  OUT_ROOT=runs/baselines-10pct \
  bash baselines/sweep_baselines.sh --device cuda
```

Both sweep scripts accept `PYTHON`, `CONFIG`, `TRAIN_FRACTION`, `OUT_ROOT`, `SEED_START`, and `SEED_END`. The defaults are seeds 0-9 inclusive. `WESN_MODE` applies to the main sweep; when unset, the YAML mode is used. Set `TRAIN_FRACTION=0.2` and choose a different output root for a 20%/80% experiment. Add `--scenes Scene10 Scene11` to select scenes. Different modes and training budgets should have separate output roots.

## Metrics and Outputs

Performance reports contain four metrics:

| Field | Meaning |
| --- | --- |
| `rmse_db` | Held-out PL RMSE in dB |
| `mae_db` | Held-out PL MAE in dB |
| `train_time_sec` | Training or fitting time in seconds |
| `test_time_sec` | Held-out prediction time in seconds |

Scratch training time includes target training. Transfer includes source pretraining plus target fine-tuning; Zero-shot includes source pretraining. The shared source stage is executed once per scene and seed, but its cost is attributed to each method that requires it. FSPL training time is zero; KNN fitting time measures tree construction. Timings exclude model initialization and result-file writes, with CUDA synchronization at timing boundaries. Training MSE remains a diagnostic loss rather than an additional performance metric.

Each seed run saves its configuration, runtime information, common splits, per-scene CSV metrics, predictions, and neural checkpoints. Baseline-only runs also save `baseline_summary.json`. The main sweep writes `seed_summary/summary.json` and `per_seed.csv`; the baseline sweep writes these and per-scene seed statistics under `sweep_summary/`.

Statistics first average scenes equally within each seed, then report the mean and sample standard deviation across seeds. Receiver counts do not determine scene weights.

Reaggregate existing main experiments into a new directory:

```bash
python scripts/aggregate_seeds.py --root runs/wesn-residual-10pct \
  --seed-start 0 --seed-end 9 --output-dir runs/wesn-residual-10pct/rechecked
```

Generate a shared-scale comparison figure from saved models:

```bash
python scripts/visualize.py --run-dir runs/wesn-residual-10pct/seed_0 \
  --scene Scene10 --output runs/figures/Scene10-residual.png --device cuda
```

A complete seven-method run produces nine panels: the original target PL map, sparse training measurements, and seven prediction maps. Dense predictions include training locations, while titles use saved held-out RMSE. Supported figure formats are PNG, PDF, and SVG, with a JSON sidecar. Visualization restores saved models without retraining.

Export one checkpoint's dense prediction:

```bash
python main.py --predict-checkpoint \
  runs/wesn-residual-10pct/seed_0/Scene10/E4_zero_shot/final.pt \
  --prediction-output runs/figures/Scene10-zero-shot.npz --device cuda
```

## References

- [NeRF2 official implementation](https://github.com/XPengZhao/NeRF2)
- [ICASSP Indoor Radio Map Challenge dataset](https://indoorradiomapchallenge.github.io/dataset.html)
- Local dataset documentation: `datasets/Documentation.pdf`.
