# ICASSP Cross-Frequency Path-Loss Prediction

This project compares spatial baselines and NeRF-WESN transfer learning on 24 paired radio-map scenes from 15 buildings. Each scene is trained and evaluated independently. The default split uses 10% of valid receiver locations for training and the remaining 90% for testing, with no validation or separate support set. Repetitions use seeds 0 through 9.

MLP trains on min-max normalized PL targets. NeRF2 and NeRF-WESN train directly on the original PL values in dB. Reported RMSE and MAE always use physical PL in dB.

## Methods

| Method | Experiment ID | Training labels | Training units |
| --- | --- | --- | --- |
| FSPL | `E1_fspl` | None; analytical free-space path loss | Not fitted |
| KNN | `B1_knn` | Target-band PL at the shared training locations | Raw PL dB |
| MLP | `B2_mlp` | Target-band PL at the shared training locations | Min-max PL |
| NeRF2 | `B3_nerf2` | Target-band PL at the shared training locations | Raw PL dB |
| NeRF-WESN Scratch | `E2_scratch` | Target-band training labels only | Raw PL dB |
| NeRF-WESN Transfer | `E3_transfer` | Source pretraining, followed by target fine-tuning | Raw PL dB |
| NeRF-WESN Zero-shot | `E4_zero_shot` | The same source model, without target fine-tuning | Raw source PL dB |

The source band is **f2 = 1.8 GHz** and the target band is **f3 = 3.5 GHz**. Source and target training coordinates are identical. All methods use the same held-out test locations. Transfer additionally uses source labels and pretraining computation.

`MLP` takes seven features: Rx xyz, Tx xyz, and `log(f_GHz / 1 GHz)`. Its x/y coordinates are normalized by the scene spans, while z remains in metres. KNN uses a uniform mean of the nearest `min(k, N_train)` raw PL observations, with `k=5` by default. FSPL and KNN continue to operate directly in dB; no label normalization is applied to them.

The standalone `NeRF2` implementation is entirely in `baselines/nerf2.py`; it does not import `Models`. Its attenuation MLP defaults to eight hidden layers of width 256, with an input skip after zero-based layer 4. NeRF2 fits raw PL with MSE in dB-squared and returns dB directly, without label scaling or inverse scaling. Its coordinate encoding is retained; coordinate scaling and PL-label normalization are separate choices.

The ray models are **NeRF2-inspired adaptations for scalar PL**, not exact reproductions of the original RSSI/CSI pipeline. The current renderer coherently sums complex contributions within each ray, then averages latent power across rays. Its negative logarithm is fitted in dB: as complete PL in direct mode, or as a signed correction to FSPL in residual mode. The latent power is a learned representation rather than an independently calibrated channel measurement. The official NeRF2 RSSI renderer instead sums complex contributions across directions before taking the magnitude. PL supervision alone does not uniquely identify latent phase or CSI.

## Per-Band Min-Max Normalization

This normalization applies only to MLP. FSPL, KNN, NeRF2 and NeRF-WESN operate directly in dB. `datasets/data_loader.py` reads the original PL images as float32 dB values and computes the normalization statistics when MLP is selected. `datasets/normalization.py` applies its forward and inverse transformations.

For each frequency band, the minimum and maximum are computed over **all PL pixels in all dataset scenes, including both training and test locations**. The statistics are not fitted separately for a scene, seed, mini-batch, or training split. Restricting a run with `--scenes` does not restrict this dataset-wide statistics pass.

The current supplied dataset has:

| Band | Frequency | Minimum PL | Maximum PL | Range |
| --- | --- | --- | --- | --- |
| f2 | 1.8 GHz | 20 dB | 160 dB | 140 dB |
| f3 | 3.5 GHz | 26 dB | 160 dB | 134 dB |

For band `f`, let `m_f` and `M_f` be these bounds and `s_f = M_f - m_f`:

```text
z_f = (PL_db - m_f) / s_f
PL_db = m_f + s_f * z_f
```

MLP fits normalized labels with MSE in normalized units. Its predictions use an unconstrained regression output: no sigmoid or clipping to `[0, 1]` is applied. Evaluation denormalizes MLP predictions before calculating RMSE and MAE against the original PL labels. NeRF2 and NeRF-WESN predictions are already in dB. Saved prediction maps, sparse measurements, and figure color scales are all in dB. MLP training logs use `train_mse_normalized`; NeRF2 and NeRF-WESN use `train_mse_db2`. These diagnostic losses are separate from the four performance metrics.

For MLP constructed with saved normalization statistics, `model(rx_m, tx_m, frequency_ghz)` returns its normalized training prediction. Use `model.predict_path_loss_db(rx_m, tx_m, frequency_ghz)` for the denormalized dB prediction. Both interfaces return raw PL dB for NeRF2 and NeRF-WESN. The training and evaluation runners select these interfaces automatically.

MLP checkpoints save the complete per-band statistics and restore them without recomputing them from inference labels. Runs containing MLP also save `label_normalization.json`; the run summary records the same statistics as `label_normalization`. Main-only NeRF-WESN runs and NeRF2-only runs do not compute or save Min-Max statistics.

The MLP baseline uses a **known-range calibration assumption**: its extrema include eventual test labels. NeRF-WESN source pretraining, target fine-tuning and Zero-shot do not use these ranges. Zero-shot uses the source model directly at the target frequency, without target training labels or target PL normalization statistics.

Use new output directories when changing label preprocessing. New NeRF2 and NeRF-WESN checkpoints train on raw PL and do not save label-normalization statistics in their model parameters. The checkpoint prediction and visualization tools still decode older normalized checkpoints using their saved statistics and selected mode. Do not combine older normalized results with new raw-PL results across seeds. Normalization can change optimization, but does not by itself guarantee better accuracy.

## WESN Training Modes

Select the mode in `configs/icassp.yml`:

```yaml
model:
  attenuation: wesn
  prediction_mode: fspl_residual  # or direct_pl
```

Both modes use the original PL labels in dB:

```text
fspl_residual:
    PL_hat_db = FSPL_db + R_theta_db

direct_pl:
    PL_hat_db = model_score_db
```

The residual `R_theta_db` is learned in dB and may be positive or negative. Both modes fit the complete raw PL target with MSE in dB-squared. Residual mode adds the FSPL prior before computing this loss; direct mode does not evaluate FSPL. No Min-Max transformation or inverse scaling is used in either mode.

The source model, Scratch, Transfer, and Zero-shot use the same mode within a run. Source pretraining and target training both use raw PL dB. Transfer fine-tunes a separate copy of the source weights; Zero-shot retains the unchanged source model. The WESN reservoir stays fixed. Checkpoints save the selected mode.

## Data and Coordinates

Only Tx/Rx coordinates, carrier frequency, and their deterministic encodings are model inputs. Material images and measured-label context are not used.

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

The PNG files already contain the original quantized PL values in dB and are not rewritten. Loading them does not divide values by 255 or treat them as RSSI. Only MLP label normalization is applied in memory and reversed for evaluation; NeRF2 and NeRF-WESN use the original dB labels.

- `x = column * 0.25 m`, `y = row * 0.25 m`.
- Tx coordinates come directly from the already-localized `positions.csv`.
- Rx height equals the Tx z coordinate; z is the antenna height, not the map's y span.
- PNG width and height determine the receiver grid. Negative original building bounds do not shift Tx/Rx again.
- Only actual coincident Tx/Rx locations are excluded from training and evaluation. There is no material-based or PL-value filtering.
- FSPL uses the true positive Euclidean distance, without a 0.25 m or 1 m distance floor.

For each scene and seed, the training count is `floor(N_valid * train_fraction)`. All remaining valid locations form the test set.

## Repository Layout

```text
Models/                  WESN attenuation and NeRF-WESN ray model
baselines/               FSPL, KNN, MLP, standalone NeRF2, comparison scripts
configs/icassp.yml       Dataset, model, optimizer, and split settings
datasets/data_loader.py Raw PL loading and dataset-wide per-band statistics
datasets/normalization.py Min-max transforms and inverse transforms
datasets/                Paired PL maps, metadata, and dataset documentation
scripts/                 Seed aggregation and saved-model visualization
tests/                   Model, data, runner, and reporting tests
main.py                  Training, inspection, and checkpoint prediction
propagation.py           Frequency features and FSPL formula
experiment_utils.py      Metrics, timing helpers, and reproducibility utilities
requirements-icassp.txt   Python dependencies
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

For CPU-only use, install PyTorch from `https://download.pytorch.org/whl/cpu` instead and pass `--device cpu` below. Run Python commands from the repository root.

## Checks and Short Runs

```bash
python main.py --config configs/icassp.yml --inspect
python -m unittest discover -s tests -v
bash -n scripts/run_main_seeds.sh scripts/run_seeds.sh baselines/sweep_baselines.sh
```

Check both WESN modes on one GPU scene, including all four baselines:

```bash
for mode in fspl_residual direct_pl; do
  CUDA_VISIBLE_DEVICES=0 python main.py --scenes Scene10 --include-baselines \
    --wesn-mode "$mode" --pretrain-steps 2 --target-steps 2 \
    --max-eval-points 32 --device cuda --output-dir "runs/raw-pl-smoke-$mode"
done
```

Short runs verify execution, not convergence or prediction accuracy. Formal experiments use `max_eval_points: 0` to score the complete held-out test set. The default budget is 2000 optimizer steps per stage, not epochs; set it before evaluating results. Output directories must be new.

Automated checks cover per-band ranges, raw and normalized training losses,
physical dB metrics, checkpoint restoration, seed statistics and visualization.
These checks are not a CUDA benchmark or a convergence result.

The ray renderers use float64 for the coherent remainder's squared magnitude and logarithm. NeRF2 also evaluates density softplus and ray compositing in float64, then casts PL back to the network dtype; network parameters remain float32 by default. Gradient clipping retries its norm in float64 when finite float32 gradients overflow the norm calculation. Actual NaN/Inf gradients stop the optimizer, with the scene, method, step, and offending parameter reported. These numerical safeguards retain the existing machine-tiny thresholds.

## Run Experiments

Run one seed with all seven methods:

```bash
python main.py --include-baselines --seed 0 --train-fraction 0.1 \
  --wesn-mode fspl_residual --device cuda --output-dir runs/wesn-raw-residual-seed0
python main.py --include-baselines --seed 0 --train-fraction 0.1 \
  --wesn-mode direct_pl --device cuda --output-dir runs/wesn-raw-direct-seed0
```

Run only Scratch, Transfer, and Zero-shot on all scenes with seeds **0 through 9**:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python OUT_ROOT=runs/main-raw-residual-10pct \
  bash scripts/run_main_seeds.sh --mode fspl_residual --device cuda

CUDA_VISIBLE_DEVICES=0 PYTHON=python OUT_ROOT=runs/main-raw-direct-10pct \
  bash scripts/run_main_seeds.sh --mode direct_pl --device cuda
```

Each output root contains `seed_0/` through `seed_9/`, `logs/seed_N.log`, and the automatically generated `seed_summary/summary.json` and `seed_summary/per_seed.csv`. The wrapper accepts `WESN_MODE`, defaults to `fspl_residual`, and lets `--mode` override that environment setting. Training failures stop the sweep before aggregation; existing results are not overwritten.

Run all seven methods, including the four baselines, with seeds **0 through 9**:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python WESN_MODE=fspl_residual \
  TRAIN_FRACTION=0.1 OUT_ROOT=runs/all-wesn-raw-residual-10pct \
  bash scripts/run_seeds.sh --device cuda

CUDA_VISIBLE_DEVICES=0 PYTHON=python WESN_MODE=direct_pl \
  TRAIN_FRACTION=0.1 OUT_ROOT=runs/all-wesn-raw-direct-10pct \
  bash scripts/run_seeds.sh --device cuda
```

Compare only the four baselines [For Alex to Run]:

```bash
python -m baselines.compare_baselines --seed 0 --device cuda \
  --output-dir runs/baselines-mlp-minmax-nerf2-raw-seed0

CUDA_VISIBLE_DEVICES=0 TRAIN_FRACTION=0.1 \
  OUT_ROOT=runs/baselines-mlp-minmax-nerf2-raw-10pct \
  bash baselines/sweep_baselines.sh --device cuda
```

The baseline sweep uses `python` from your active environment. With the default settings, run `bash baselines/sweep_baselines.sh` to compare all four baselines over seeds 0-9 and generate the summary.

The sweep scripts accept `PYTHON`, `CONFIG`, `TRAIN_FRACTION`, `OUT_ROOT`, `SEED_START`, and `SEED_END`. Defaults are seeds 0-9 inclusive. For `run_seeds.sh`, an unset `WESN_MODE` preserves the YAML mode. Set `TRAIN_FRACTION=0.2` and choose a new output root for a 20%/80% experiment. Add `--scenes Scene10 Scene11` to select scenes. When MLP is selected, its statistics still cover the complete dataset. Different modes, preprocessing settings, and training budgets should have separate output roots.

## Metrics and Outputs

| Field | Meaning |
| --- | --- |
| `rmse_db` | Held-out PL RMSE in dB, after inverse scaling where required |
| `mae_db` | Held-out PL MAE in dB, after inverse scaling where required |
| `train_time_sec` | Training or fitting time in seconds |
| `test_time_sec` | Held-out prediction time in seconds |

Scratch training time includes target training. Transfer includes source pretraining plus target fine-tuning; Zero-shot includes source pretraining. The shared source stage executes once per scene and seed, but its cost is attributed to each method that requires it. FSPL training time is zero; KNN fitting time measures tree construction. Timings exclude model initialization and result-file writes, with CUDA synchronization at timing boundaries. Training MSE, whether normalized or in dB-squared, is diagnostic rather than an additional performance metric.

Each seed run saves its configuration, runtime information, common splits, per-scene dB metrics, dB predictions, and neural checkpoints. Runs containing MLP save `label_normalization.json`, and summaries record its calibration statistics. FSPL, KNN, NeRF2 and NeRF-WESN do not apply label normalization. Baseline-only runs also save `baseline_summary.json`. The main sweep writes `seed_summary/summary.json` and `per_seed.csv`; the baseline sweep adds per-scene seed statistics under `sweep_summary/`.

Statistics first average scenes equally within each seed, then report the mean and sample standard deviation across seeds. Receiver counts do not determine scene weights. Aggregation checks that per-method label preprocessing and bounds match across seeds; do not mix older normalized NeRF2/WESN results with their new raw-label results.

Reaggregate existing main experiments into a new directory:

```bash
python scripts/aggregate_seeds.py --root runs/main-raw-residual-10pct \
  --seed-start 0 --seed-end 9 --output-dir runs/main-raw-residual-10pct/rechecked
```

Generate a shared-scale comparison figure from saved models:

```bash
python scripts/visualize.py --run-dir runs/main-raw-residual-10pct/seed_0 \
  --scene Scene10 --output runs/raw-pl-figures/Scene10-residual.png --device cuda
```

A complete seven-method run produces nine panels: the original target PL map, sparse training measurements, and seven prediction maps. A main-only run has five panels. Dense predictions include training locations, while titles use saved held-out RMSE in dB. Supported formats are PNG, PDF, and SVG, with a JSON sidecar that includes available normalization statistics. Visualization restores each checkpoint's saved output units without retraining.

Export a checkpoint's dense prediction in physical PL dB:

```bash
python main.py --predict-checkpoint \
  runs/main-raw-residual-10pct/seed_0/Scene10/E4_zero_shot/final.pt \
  --prediction-output runs/raw-pl-figures/Scene10-zero-shot.npz --device cuda
```

## References

- [NeRF2 official implementation](https://github.com/XPengZhao/NeRF2)
- [ICASSP Indoor Radio Map Challenge dataset](https://indoorradiomapchallenge.github.io/dataset.html)
- Local dataset documentation: `datasets/Documentation.pdf`.
