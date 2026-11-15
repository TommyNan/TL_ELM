"""Sequential f2->f3 transfer with matched receiver splits (default 10%/90%).

E2: target scratch. E3: source pretrain then target fine-tune.
E4: the very same source checkpoint, evaluated at target frequency before
fine-tuning. Baselines are analytical FSPL, direct-PL KNN, MLP and NeRF2-MLP.
Neural inputs are Tx/Rx coordinates, derived features and carrier frequency.
Material maps are never read. WESN can use an FSPL residual prior or directly
predict PL; MLP and the independent NeRF2 baseline always predict PL directly.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from Models.model import RadioMapField
from baselines.fspl import FSPL
from baselines.knn import KNN
from baselines.mlp import MLP
from baselines.nerf2 import NeRF2Field
from datasets.data_loader import discover_scenes, load_scene
from experiment_utils import (aggregate_results, choose_device, metrics,
                              reproducibility_metadata, scene_seed, seed_everything, sync_if_cuda)
from propagation import WESN_PREDICTION_MODES, require_wesn_prediction_mode


PROJECT_ROOT = Path(__file__).resolve().parent
METHODS = ("E2_scratch", "E3_transfer", "E4_zero_shot")
BASELINES = ("E1_fspl", "B1_knn", "B2_mlp", "B3_nerf2")
NEURAL_BASELINES = ("B2_mlp", "B3_nerf2")


def make_train_test_split(valid_mask, train_fraction, seed):
    if not math.isfinite(float(train_fraction)) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    indices = np.flatnonzero(np.asarray(valid_mask).reshape(-1))
    count = math.floor(len(indices) * train_fraction)
    if not 0 < count < len(indices):
        raise ValueError("Train and test must both contain receivers")
    order = np.random.default_rng(seed).permutation(indices)
    return {"train": np.sort(order[:count]), "test": np.sort(order[count:])}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def copy_state(model):
    return {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}


def new_model(params, seed, device, model_type="radio_map"):
    seed_everything(seed)
    factories = {"radio_map": RadioMapField, "plain_mlp": PlainMLP, "nerf2": NeRF2Field}
    return factories[model_type](params).to(device)


def fit_field(model, rx_m, tx_m, frequency_ghz, labels_db,
              training, steps, lr, batch_seed, stage=None):
    """Fit from explicitly sliced training arrays; no held-out labels accepted.

    A new Adam optimizer is created for every stage, including fine-tuning.
    E2/E3 use the same batch seed, target step count and default learning rate.
    All trainable weights adapt; WESN's fixed reservoir is never optimized.
    """
    device = next(model.parameters()).device
    rx = torch.as_tensor(rx_m, dtype=torch.float32, device=device)
    tx = torch.as_tensor(tx_m, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_db, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(lr), weight_decay=float(training["weight_decay"]))
    rng = np.random.default_rng(batch_seed)
    history = []
    sync_if_cuda(device)
    start = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        batch = rng.choice(len(rx), min(training["batch_size"], len(rx)), replace=False)
        index = torch.as_tensor(batch, device=device)
        prediction = model(rx[index], tx.expand(len(index), -1), frequency_ghz)
        loss = (prediction - labels[index]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"],
                                       error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % training["log_every"] == 0 or step == steps:
            entry = {"step": step, "train_mse_db2": float(loss.item())}
            history.append(entry)
            if stage:
                print(f'{stage}: step {step}/{steps}, train MSE={loss.item():.4f} dB^2', flush=True)
    sync_if_cuda(device)
    return history, time.perf_counter() - start


def predict_field(model, rx_m, tx_m, frequency_ghz, batch_size):
    """Pure prediction; neither source nor target path-loss labels are inputs."""
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rx_m), batch_size):
            rx = torch.as_tensor(rx_m[start:start + batch_size], dtype=torch.float32, device=device)
            tx = torch.as_tensor(tx_m, dtype=torch.float32, device=device).expand(len(rx), -1)
            prediction = model(rx, tx, frequency_ghz)
            result.append(prediction.cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


def scene_from_config(config, scene_id):
    data = config["data"]
    root = Path(data["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return load_scene(root, scene_id, data["source_band"], data["target_band"],
                      band_frequencies_ghz={data["source_band"]: data["source_frequency_ghz"],
                                            data["target_band"]: data["target_frequency_ghz"]})


def save_checkpoint(path, model, params, config, scene_id, method, initial_digest, record,
                    model_type="radio_map", provenance=None):
    state = copy_state(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state,
                "state_sha256": state_digest(state), "initial_state_sha256": initial_digest,
                "model_params": params, "model_type": model_type,
                "config": config, "scene_id": scene_id,
                "reproducibility": provenance,
                "method": method, "metrics": record}, path)


def record_prediction(dest, record, indices, prediction, target):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    np.savez_compressed(dest / "test_predictions.npz", indices=indices,
                        prediction_db=prediction, target_db=target)


def run_scene(scene, split, methods, config, out, device, seed, baseline_names, provenance):
    data, training = config["data"], config["train"]
    prediction_mode = config["model"].get("prediction_mode", "fspl_residual")
    train_indices, test_indices = split["train"], split["test"]
    limit = training["max_eval_points"]
    if limit and len(test_indices) > limit:
        test_indices = np.sort(np.random.default_rng(seed + 1).choice(test_indices, limit, replace=False))
    scene_dir = out / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scene_dir / "split.npz", **split,
                        source_train=train_indices, target_train=train_indices,
                        test_evaluation=test_indices)
    params = dict(config["model"], width_m=scene.width_m, height_m=scene.height_m,
                  prediction_mode=prediction_mode)
    source_state, source_hash = None, None
    source_seconds = 0.0
    if "E3_transfer" in methods or "E4_zero_shot" in methods:
        source = new_model(params, seed, device)
        initial_digest = state_digest(source.state_dict())
        history, source_seconds = fit_field(
            source, scene.rx_m[train_indices], scene.tx_m,
            data["source_frequency_ghz"], scene.source_pl_db[train_indices], training,
            training["pretrain_steps"], training["pretrain_lr"], seed + 17,
            stage=f'{scene.scene_id}/source_pretrain')
        source_state = copy_state(source)
        source_hash = state_digest(source_state)
        source_record = dict(train_time_sec=source_seconds, f2_label_count=len(train_indices),
                             f3_label_count=0, source_steps=training["pretrain_steps"])
        save_checkpoint(scene_dir / "source_pretrain/final.pt", source, params, config,
                        scene.scene_id, "source_pretrain", initial_digest, source_record,
                        provenance=provenance)
        (scene_dir / "source_pretrain/history.json").write_text(json.dumps(history, indent=2))
        del source

    rows = []
    for method in methods:
        model = new_model(params, seed, device)
        source_steps = training["pretrain_steps"] if method != "E2_scratch" else 0
        target_steps = 0 if method == "E4_zero_shot" else training["target_steps"]
        # Fine-tune a separate instance; the source state is shared with E4.
        if source_steps:
            model.load_state_dict(source_state)
        initial_digest = state_digest(model.state_dict())
        history, target_seconds = [], 0.0
        if target_steps:
            lr = training["target_lr"] if method == "E2_scratch" else training["finetune_lr"]
            history, target_seconds = fit_field(
                model, scene.rx_m[train_indices], scene.tx_m,
                data["target_frequency_ghz"], scene.target_pl_db[train_indices], training,
                target_steps, lr, seed + 17, stage=f'{scene.scene_id}/{method}')
        sync_if_cuda(device)
        test_start = time.perf_counter()
        prediction = predict_field(model, scene.rx_m[test_indices], scene.tx_m,
                                   data["target_frequency_ghz"], training["eval_batch_size"])
        sync_if_cuda(device)
        test_seconds = time.perf_counter() - test_start
        record = dict(scene_id=scene.scene_id, building_id=scene.building_id,
                      approach=method, seed=training["seed"], scene_seed=seed,
                      train_fraction=data["train_fraction"], n_train=len(train_indices),
                      n_test_total=len(split["test"]),
                      f2_label_count=len(train_indices) if source_steps else 0,
                      f3_label_count=len(train_indices) if target_steps else 0,
                      source_steps=source_steps, target_steps=target_steps,
                      source_state_sha256=source_hash if source_steps else None,
                      wesn_prediction_mode=prediction_mode,
                      train_time_sec=target_seconds + (source_seconds if source_steps else 0.0),
                      test_time_sec=test_seconds,
                      **metrics(prediction, scene.target_pl_db[test_indices]))
        dest = scene_dir / method
        save_checkpoint(dest / "final.pt", model, params, config, scene.scene_id,
                        method, initial_digest, record, provenance=provenance)
        (dest / "history.json").write_text(json.dumps(history, indent=2))
        record_prediction(dest, record, test_indices, prediction, scene.target_pl_db[test_indices])
        rows.append(record)
        print(f'{scene.scene_id} {method}: RMSE={record["rmse_db"]:.3f}, MAE={record["mae_db"]:.3f} dB', flush=True)
        del model

    baseline_rows = []
    for name in baseline_names:
        dest = scene_dir / "baselines" / name
        history, model, initial_digest = [], None, None
        train_seconds, target_steps = 0.0, 0
        measured_count = len(train_indices)
        if name == "E1_fspl":
            parameters = dict(tx_m=scene.tx_m.tolist(), frequency_ghz=data["target_frequency_ghz"])
            baseline = FSPL(**parameters)
            model_type, measured_count = "fspl", 0
        elif name == "B1_knn":
            parameters = dict(config["baseline"]["knn"])
            baseline = KNN(**parameters)
            train_start = time.perf_counter()
            baseline.fit(scene.rx_m[train_indices], scene.target_pl_db[train_indices])
            train_seconds = time.perf_counter() - train_start
            model_type = "knn"
        else:
            model_type = "plain_mlp" if name == "B2_mlp" else "nerf2"
            defaults = ({"hidden_dim": 128, "num_layers": 4} if model_type == "plain_mlp"
                        else {"hidden_dim": 256, "mlp_num_layers": 8, "mlp_skips": [4]})
            baseline_params = dict(params, **defaults)
            baseline_params.update(config.get("baseline_models", {}).get(name, {}))
            model = new_model(baseline_params, seed, device, model_type)
            initial_digest = state_digest(model.state_dict())
            target_steps = training["target_steps"]
            history, train_seconds = fit_field(
                model, scene.rx_m[train_indices], scene.tx_m,
                data["target_frequency_ghz"], scene.target_pl_db[train_indices], training,
                target_steps, training["target_lr"], seed + 17, stage=f'{scene.scene_id}/{name}')
        sync_if_cuda(device)
        test_start = time.perf_counter()
        prediction = (baseline.predict(scene.rx_m[test_indices]) if model is None else
                      predict_field(model, scene.rx_m[test_indices], scene.tx_m,
                                    data["target_frequency_ghz"], training["eval_batch_size"]))
        sync_if_cuda(device)
        test_seconds = time.perf_counter() - test_start
        record = dict(scene_id=scene.scene_id, building_id=scene.building_id,
                      approach=name, seed=training["seed"], scene_seed=seed,
                      f2_label_count=0, f3_label_count=measured_count, n_train=measured_count,
                      n_test_total=len(split["test"]), train_fraction=data["train_fraction"],
                      source_steps=0, target_steps=target_steps, model_type=model_type,
                      train_time_sec=train_seconds, test_time_sec=test_seconds,
                      **metrics(prediction, scene.target_pl_db[test_indices]))
        record_prediction(dest, record, test_indices, prediction, scene.target_pl_db[test_indices])
        if model is None:
            (dest / "parameters.json").write_text(json.dumps(parameters, indent=2))
            if model_type == "knn":
                np.savez_compressed(dest / "observations.npz", indices=train_indices,
                                    rx_m=scene.rx_m[train_indices], path_loss_db=scene.target_pl_db[train_indices])
        else:
            save_checkpoint(dest / "final.pt", model, baseline_params, config, scene.scene_id,
                            name, initial_digest, record, model_type, provenance=provenance)
            (dest / "history.json").write_text(json.dumps(history, indent=2))
        baseline_rows.append(record)
        print(f'{scene.scene_id} {name}: RMSE={record["rmse_db"]:.3f}, MAE={record["mae_db"]:.3f} dB', flush=True)
    return rows, baseline_rows


def validate_config(config):
    """Check experiment choices once, before loading scenes or training."""
    data = config["data"]
    require_wesn_prediction_mode(config["model"].get("prediction_mode", "fspl_residual"))
    if data.get("val_fraction", 0) or data.get("support_fraction", 0):
        raise ValueError("Use training/test splits without validation or support partitions")
    if min(data["source_frequency_ghz"], data["target_frequency_ghz"]) <= 0:
        raise ValueError("Frequencies must be positive GHz values")


def write_results(out, rows, baselines, inventory, selected, methods, baseline_names, config, provenance):
    for filename, values in (("main_per_scene.csv", rows), ("baseline_per_scene.csv", baselines)):
        if values:
            with (out / filename).open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    expected = len(selected) * len(methods)
    report = dict(main_experiments=aggregate_results(rows),
                  baselines=aggregate_results(baselines), selected_scenes=selected,
                  methods=list(methods), inventory=inventory, completed_runs=len(rows),
                  expected_runs=expected,
                  include_baselines=bool(baseline_names), baseline_names=list(baseline_names),
                  wesn_prediction_mode=(config["model"].get("prediction_mode", "fspl_residual") if methods else None),
                  seed=config["train"]["seed"], train_fraction=config["data"]["train_fraction"],
                  completed_baseline_runs=len(baselines),
                  expected_baseline_runs=len(selected) * len(baseline_names),
                  complete=len(rows) == expected and len(baselines) == len(selected) * len(baseline_names),
                  checkpoint_selection="final_step", evaluation_limited=bool(config["train"]["max_eval_points"]),
                  source_sha256=provenance['source_sha256'], dataset_sha256=provenance['dataset_sha256'],
                  reproducibility=provenance,
                  units="path_loss_db", time_unit="seconds")
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def run(config, scenes=None, methods=METHODS, include_baselines=False, inspect_only=False, baselines=None):
    validate_config(config)
    methods = list(methods)
    baseline_names = list(BASELINES if include_baselines else ()) if baselines is None else list(baselines)
    if len(set(baseline_names)) != len(baseline_names) or any(b not in BASELINES for b in baseline_names):
        raise ValueError(f"Choose distinct baselines from {BASELINES}")
    if ((not methods and not baseline_names) or len(set(methods)) != len(methods)
            or any(m not in METHODS for m in methods)):
        raise ValueError(f"Choose distinct methods from {METHODS}")
    dataset_root = Path(config["data"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    available = discover_scenes(dataset_root)
    selected = list(available if scenes is None else scenes)
    if not selected or len(set(selected)) != len(selected) or any(s not in available for s in selected):
        raise ValueError("Choose distinct scene IDs from the dataset")
    torch.set_num_threads(config["train"]["num_threads"])
    device = choose_device(config["train"]["device"])
    out = Path(config["output_dir"])
    provenance = None
    if not inspect_only:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output directory is nonempty; choose a new output_dir: {out}")
        out.mkdir(parents=True, exist_ok=True)
        provenance = reproducibility_metadata(PROJECT_ROOT, dataset_root, selected,
                                               config['data']['source_band'],
                                               config['data']['target_band'], device)
        (out / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        (out / "runtime.json").write_text(json.dumps(provenance, indent=2))
        print(f'Runtime: {provenance["torch_version"]}, device={device}, '
              f'GPU={provenance["cuda_device_name"]}, code={provenance["source_sha256"][:12]}', flush=True)
    rows, baseline_rows, inventory = [], [], []
    for scene_id in selected:
        scene = scene_from_config(config, scene_id)
        seed = scene_seed(config["train"]["seed"], scene_id)
        split = make_train_test_split(scene.valid_mask, config["data"]["train_fraction"], seed)
        inventory.append(dict(scene_id=scene_id, building_id=scene.building_id,
                              n_pixels=scene.height*scene.width,
                              image_width_px=scene.width, image_height_px=scene.height,
                              coordinate_frame=scene.coordinate_frame,
                              building_bounds_m=scene.building_bounds_m,
                              raster_coverage_m=scene.raster_coverage_m,
                              grid_bounds_m=scene.grid_bounds_m,
                              tx_pixel_index=list(scene.tx_pixel_index),
                              n_coincident_receivers=int((scene.distance_m == 0).sum()),
                              n_valid=int(scene.valid_mask.sum()), n_train=len(split["train"]),
                              n_test=len(split["test"])))
        if not inspect_only:
            current, baseline = run_scene(scene, split, methods, config, out, device, seed, baseline_names, provenance)
            rows.extend(current)
            baseline_rows.extend(baseline)
            report = write_results(out, rows, baseline_rows, inventory, selected, methods,
                                   baseline_names, config, provenance)
    if inspect_only:
        report = dict(n_scenes=len(inventory), n_buildings=len({s["building_id"] for s in inventory}),
                      wesn_prediction_mode=config["model"].get("prediction_mode", "fspl_residual"),
                      inventory=inventory, source_frequency_ghz=config["data"]["source_frequency_ghz"],
                      target_frequency_ghz=config["data"]["target_frequency_ghz"])
        print(json.dumps(report, indent=2))
    return report


def predict_checkpoint(checkpoint_path, output_path, device="auto"):
    output = Path(output_path)
    if output.suffix != ".npz" or output.exists():
        raise ValueError("Prediction output must be a new .npz file")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model_type = checkpoint["model_type"]
    prediction_mode = checkpoint["model_params"].get("prediction_mode", "fspl_residual")
    torch.set_num_threads(config["train"]["num_threads"])
    scene = scene_from_config(config, checkpoint["scene_id"])
    model = new_model(checkpoint["model_params"], config["train"]["seed"], choose_device(device),
                       model_type)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    indices = np.flatnonzero(scene.valid_mask)
    prediction = np.full(scene.height * scene.width, np.nan, dtype=np.float32)
    prediction[indices] = predict_field(model, scene.rx_m[indices], scene.tx_m,
                                       config["data"]["target_frequency_ghz"],
                                       config["train"]["eval_batch_size"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, prediction_db=prediction.reshape(scene.height, scene.width),
                        wesn_prediction_mode=prediction_mode if model_type == "radio_map" else "",
                        scene_id=scene.scene_id, method=checkpoint["method"],
                        target_frequency_ghz=config["data"]["target_frequency_ghz"])
    print(f"Saved dense predictions to {output}; use held-out metrics for evaluation.")
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/icassp.yml"))
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", help="Run only the selected baselines")
    parser.add_argument("--baselines", nargs="+", choices=BASELINES)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--pretrain-steps", type=int)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--max-eval-points", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--wesn-mode", "--prediction-mode", dest="wesn_mode", choices=WESN_PREDICTION_MODES,
                        help="WESN output mode; all three main experiments use the same selection")
    parser.add_argument("--predict-checkpoint")
    parser.add_argument("--prediction-output")
    args = parser.parse_args()
    if args.predict_checkpoint:
        if args.wesn_mode is not None:
            parser.error("Checkpoint inference uses its saved WESN mode; --wesn-mode is a training option")
        if not args.prediction_output:
            parser.error("--predict-checkpoint requires --prediction-output")
        predict_checkpoint(args.predict_checkpoint, args.prediction_output, args.device or "auto")
        return
    with open(args.config) as file:
        config = yaml.safe_load(file)
    for name in ("pretrain_steps", "target_steps", "max_eval_points", "device", "seed"):
        value = getattr(args, name)
        if value is not None:
            config["train"][name] = value
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.train_fraction is not None:
        config["data"]["train_fraction"] = args.train_fraction
    if args.wesn_mode is not None:
        config["model"]["prediction_mode"] = args.wesn_mode
        if args.output_dir is None:
            percentage = f'{config["data"]["train_fraction"] * 100:g}'
            config["output_dir"] = f'runs/wesn-{args.wesn_mode.replace("_", "-")}-{percentage}pct'
    run(config, args.scenes, [] if args.baseline_only else args.methods,
        args.include_baselines or args.baseline_only, args.inspect, baselines=args.baselines)


if __name__ == "__main__":
    main()
