"""Sequential f2->f3 transfer, per scene, with matched 20%/80% splits.

E2: target scratch. E3: source pretrain then target fine-tune.
E4: the very same source checkpoint, evaluated at target frequency before
fine-tuning. E1 FSPL+IDW is an optional, separately reported baseline.
No query PL measurements or observation context enter neural predictions.
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
from baselines.fspl_idw import FSPLIDW
from datasets.data_loader import discover_scenes, load_scene
from experiment_utils import aggregate_results, as_geometry, choose_device, metrics, scene_seed, seed_everything


METHODS = ("E2_scratch", "E3_transfer", "E4_zero_shot")
BASELINE = "E1_fspl_idw"


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


def new_model(params, seed, device):
    seed_everything(seed)
    return RadioMapField(params).to(device)


def fit_field(model, rx_m, tx_m, geometry, frequency_ghz, labels_db,
              training, steps, lr, batch_seed):
    """Fit from explicitly sliced training arrays; no held-out labels accepted.

    A new Adam optimizer is created for every stage, including fine-tuning.
    E2/E3 use the same batch seed, target step count and default learning rate.
    All trainable weights adapt; WESN's fixed reservoir is never optimized.
    """
    if len(rx_m) != len(labels_db) or not len(rx_m):
        raise ValueError("Training coordinates and labels must have equal nonzero length")
    device = next(model.parameters()).device
    rx = torch.as_tensor(rx_m, dtype=torch.float32, device=device)
    tx = torch.as_tensor(tx_m, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_db, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("Nonfinite training labels")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(lr), weight_decay=float(training["weight_decay"]))
    rng = np.random.default_rng(batch_seed)
    history = []
    start = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        batch = rng.choice(len(rx), min(training["batch_size"], len(rx)), replace=False)
        index = torch.as_tensor(batch, device=device)
        prediction = model(rx[index], tx.expand(len(index), -1), frequency_ghz, geometry)
        loss = ((prediction - labels[index]) / 40.0).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % training["log_every"] == 0 or step == steps:
            history.append({"step": step, "train_mse_db2": float(loss.item() * 1600)})
    return history, time.monotonic() - start


def predict_field(model, rx_m, tx_m, frequency_ghz, geometry, batch_size):
    """Pure prediction; neither source nor target path-loss labels are inputs."""
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rx_m), batch_size):
            rx = torch.as_tensor(rx_m[start:start + batch_size], dtype=torch.float32, device=device)
            tx = torch.as_tensor(tx_m, dtype=torch.float32, device=device).expand(len(rx), -1)
            result.append(model(rx, tx, frequency_ghz, geometry).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


def scene_from_config(config, scene_id):
    data = config["data"]
    return load_scene(data["root"], scene_id, data["source_band"], data["target_band"],
                      band_frequencies_ghz={data["source_band"]: data["source_frequency_ghz"],
                                            data["target_band"]: data["target_frequency_ghz"]})


def save_checkpoint(path, model, params, config, scene_id, method, initial_digest, record):
    state = copy_state(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": "sequential_transfer", "state_dict": state,
                "state_sha256": state_digest(state), "initial_state_sha256": initial_digest,
                "model_params": params, "config": config, "scene_id": scene_id,
                "method": method, "metrics": record}, path)


def record_prediction(dest, record, indices, prediction, target):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    np.savez_compressed(dest / "test_predictions.npz", indices=indices,
                        prediction_db=prediction, target_db=target)


def run_scene(scene, split, methods, config, out, device, seed, include_baselines):
    data, training = config["data"], config["train"]
    train_indices, test_indices = split["train"], split["test"]
    limit = training["max_eval_points"]
    if limit and len(test_indices) > limit:
        test_indices = np.sort(np.random.default_rng(seed + 1).choice(test_indices, limit, replace=False))
    scene_dir = out / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    # Identical source/target training coordinates; there is no support subset.
    np.savez_compressed(scene_dir / "split.npz", **split,
                        source_train=train_indices, target_train=train_indices,
                        test_evaluation=test_indices)
    params = dict(config["model"], width_m=scene.width_m, height_m=scene.height_m)
    geometries = {"source": as_geometry(scene.source_geometry, device),
                  "target": as_geometry(scene.geometry, device)}
    source_state, source_hash, source_initial = None, None, None
    source_history, source_seconds = [], 0.0
    source_needed = "E3_transfer" in methods or "E4_zero_shot" in methods
    if source_needed:
        source = new_model(params, seed, device)
        source_initial = state_digest(source.state_dict())
        source_history, source_seconds = fit_field(
            source, scene.rx_m[train_indices], scene.tx_m, geometries["source"],
            data["source_frequency_ghz"], scene.source_pl_db[train_indices], training,
            training["pretrain_steps"], training["pretrain_lr"], seed + 17)
        source_state = copy_state(source)
        source_hash = state_digest(source_state)
        source_record = dict(f2_label_count=len(train_indices), f3_label_count=0,
                             source_steps=training["pretrain_steps"], target_steps=0,
                             checkpoint_selection="final_step", elapsed_s=source_seconds)
        save_checkpoint(scene_dir / "source_pretrain" / "final.pt", source, params, config,
                        scene.scene_id, "source_pretrain", source_initial, source_record)
        (scene_dir / "source_pretrain" / "history.json").write_text(json.dumps(source_history, indent=2))
        del source
    rows = []
    # Source checkpoint is immutable; E3 fine-tunes a separate model instance.
    for method in methods:
        start = time.monotonic()
        model = new_model(params, seed, device)
        source_steps = training["pretrain_steps"] if method != "E2_scratch" else 0
        target_steps = 0 if method == "E4_zero_shot" else training["target_steps"]
        if method != "E2_scratch":
            model.load_state_dict(source_state, strict=True)
        initial_digest = state_digest(model.state_dict())
        history = []
        if target_steps:
            lr = training["target_lr"] if method == "E2_scratch" else training["finetune_lr"]
            history, _ = fit_field(
                model, scene.rx_m[train_indices], scene.tx_m, geometries["target"],
                data["target_frequency_ghz"], scene.target_pl_db[train_indices], training,
                target_steps, lr, seed + 17)
        prediction = predict_field(model, scene.rx_m[test_indices], scene.tx_m,
                                   data["target_frequency_ghz"], geometries["target"],
                                   training["eval_batch_size"])
        record = dict(scene_id=scene.scene_id, building_id=scene.building_id,
                      approach=method, seed=seed, train_fraction=data["train_fraction"],
                      n_train=len(train_indices), n_test_total=len(split["test"]),
                      f2_label_count=len(train_indices) if source_steps else 0,
                      f3_label_count=len(train_indices) if target_steps else 0,
                      source_steps=source_steps, target_steps=target_steps,
                      checkpoint_selection="final_step", source_state_sha256=source_hash if source_steps else None,
                      source_frequency_ghz=data["source_frequency_ghz"],
                      target_frequency_ghz=data["target_frequency_ghz"],
                      elapsed_s=time.monotonic() - start,
                      shared_pretrain_elapsed_s=source_seconds if source_steps else 0.0,
                      evaluation_limited=bool(limit),
                      **metrics(prediction, scene.target_pl_db[test_indices]))
        dest = scene_dir / method
        save_checkpoint(dest / "final.pt", model, params, config, scene.scene_id,
                        method, initial_digest, record)
        (dest / "history.json").write_text(json.dumps(history, indent=2))
        record_prediction(dest, record, test_indices, prediction, scene.target_pl_db[test_indices])
        rows.append(record)
        print(f'{scene.scene_id} {method}: test RMSE={record["rmse_db"]:.3f} dB', flush=True)
        del model
    # Guard against accidental in-place fine-tuning of the shared source weights.
    if source_state is not None and state_digest(source_state) != source_hash:
        raise RuntimeError("Fine-tuning modified the source checkpoint")
    baseline_rows = []
    if include_baselines:
        start = time.monotonic()
        baseline = FSPLIDW(**config["baseline"]).fit(
            scene.rx_m[train_indices], scene.target_pl_db[train_indices],
            scene.tx_m, data["target_frequency_ghz"])
        prediction = baseline.predict(scene.rx_m[test_indices])
        record = dict(scene_id=scene.scene_id, building_id=scene.building_id,
                      approach=BASELINE, seed=seed, f2_label_count=0,
                      f3_label_count=len(train_indices), n_train=len(train_indices),
                      n_test_total=len(split["test"]), elapsed_s=time.monotonic() - start,
                      evaluation_limited=bool(limit),
                      **metrics(prediction, scene.target_pl_db[test_indices]))
        dest = scene_dir / "baselines" / BASELINE
        record_prediction(dest, record, test_indices, prediction, scene.target_pl_db[test_indices])
        np.savez_compressed(dest / "observations.npz", indices=train_indices,
                            rx_m=scene.rx_m[train_indices], path_loss_db=scene.target_pl_db[train_indices],
                            tx_m=scene.tx_m, frequency_ghz=data["target_frequency_ghz"])
        baseline_rows.append(record)
        print(f'{scene.scene_id} baseline {BASELINE}: test RMSE={record["rmse_db"]:.3f} dB', flush=True)
    return rows, baseline_rows


def validate_config(config):
    if config.get("protocol") != "sequential_transfer":
        raise ValueError("Use a sequential_transfer configuration, e.g. configs/icassp-transfer.yml")
    data, train = config["data"], config["train"]
    if data["source_band"] == data["target_band"]:
        raise ValueError("Source and target bands must differ")
    for key in ("source_frequency_ghz", "target_frequency_ghz"):
        if not math.isfinite(float(data[key])) or data[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 < data["train_fraction"] < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    for key in ("pretrain_steps", "target_steps", "batch_size", "eval_batch_size", "log_every", "num_threads"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("pretrain_lr", "target_lr", "finetune_lr"):
        if not math.isfinite(float(train[key])) or train[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(float(train["weight_decay"])) or train["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if not isinstance(train["max_eval_points"], int) or train["max_eval_points"] < 0:
        raise ValueError("max_eval_points must be a nonnegative integer")


def write_results(out, rows, baselines, inventory, selected, methods, include_baselines, config):
    for filename, values in (("main_per_scene.csv", rows), ("baseline_per_scene.csv", baselines)):
        if values:
            with (out / filename).open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    paired = []
    for scene_id in selected:
        scores = {r["approach"]: r["rmse_db"] for r in rows if r["scene_id"] == scene_id}
        if "E2_scratch" in scores and "E3_transfer" in scores:
            paired.append({"scene_id": scene_id,
                           "scratch_minus_transfer_rmse_db": scores["E2_scratch"] - scores["E3_transfer"]})
    expected = len(selected) * len(methods)
    report = dict(protocol="sequential_transfer", main_experiments=aggregate_results(rows),
                  baselines=aggregate_results(baselines), selected_scenes=selected,
                  methods=list(methods), inventory=inventory, completed_runs=len(rows),
                  expected_runs=expected,
                  include_baselines=include_baselines,
                  completed_baseline_runs=len(baselines),
                  expected_baseline_runs=len(selected) if include_baselines else 0,
                  complete=len(rows) == expected and (not include_baselines or len(baselines) == len(selected)),
                  checkpoint_selection="final_step", evaluation_limited=bool(config["train"]["max_eval_points"]),
                  paired_transfer_gain=paired,
                  mean_transfer_gain_db=(float(np.mean([p["scratch_minus_transfer_rmse_db"] for p in paired]))
                                         if paired else None), units="path_loss_db")
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def run(config, scenes=None, methods=METHODS, include_baselines=False, inspect_only=False):
    validate_config(config)
    methods = list(methods)
    if ((not methods and not include_baselines) or len(set(methods)) != len(methods)
            or any(m not in METHODS for m in methods)):
        raise ValueError(f"Choose distinct methods from {METHODS}")
    available = discover_scenes(config["data"]["root"])
    selected = list(available if scenes is None else scenes)
    if not selected or len(set(selected)) != len(selected) or any(s not in available for s in selected):
        raise ValueError("Choose distinct scene IDs from the dataset")
    torch.set_num_threads(config["train"]["num_threads"])
    device = choose_device(config["train"]["device"])
    out = Path(config["output_dir"])
    if not inspect_only:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output directory is nonempty; choose a new output_dir: {out}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    rows, baseline_rows, inventory = [], [], []
    for scene_id in selected:
        scene = scene_from_config(config, scene_id)
        seed = scene_seed(config["train"]["seed"], scene_id)
        split = make_train_test_split(scene.valid_mask, config["data"]["train_fraction"], seed)
        inventory.append(dict(scene_id=scene_id, building_id=scene.building_id,
                              n_valid=int(scene.valid_mask.sum()), n_train=len(split["train"]),
                              n_test=len(split["test"])))
        if not inspect_only:
            current, baseline = run_scene(scene, split, methods, config, out, device, seed, include_baselines)
            rows.extend(current)
            baseline_rows.extend(baseline)
            report = write_results(out, rows, baseline_rows, inventory, selected, methods,
                                   include_baselines, config)
    if inspect_only:
        report = dict(n_scenes=len(inventory), n_buildings=len({s["building_id"] for s in inventory}),
                      inventory=inventory, source_frequency_ghz=config["data"]["source_frequency_ghz"],
                      target_frequency_ghz=config["data"]["target_frequency_ghz"])
        print(json.dumps(report, indent=2))
    return report


def predict_checkpoint(checkpoint_path, output_path, device="auto"):
    output = Path(output_path)
    if output.suffix != ".npz" or output.exists():
        raise ValueError("Prediction output must be a new .npz file")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol") != "sequential_transfer" or checkpoint["method"] not in METHODS:
        raise ValueError("Use an E2/E3/E4 checkpoint from the sequential protocol")
    config = checkpoint["config"]
    torch.set_num_threads(config["train"]["num_threads"])
    scene = scene_from_config(config, checkpoint["scene_id"])
    model = new_model(checkpoint["model_params"], config["train"]["seed"], choose_device(device))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if state_digest(model.state_dict()) != checkpoint["state_sha256"]:
        raise ValueError("Checkpoint state fingerprint mismatch")
    geometry = as_geometry(scene.geometry, next(model.parameters()).device)
    prediction = predict_field(model, scene.rx_m, scene.tx_m, config["data"]["target_frequency_ghz"],
                               geometry, config["train"]["eval_batch_size"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, prediction_db=prediction.reshape(scene.height, scene.width),
                        scene_id=scene.scene_id, method=checkpoint["method"],
                        target_frequency_ghz=config["data"]["target_frequency_ghz"])
    print(f"Saved dense predictions to {output}; use held-out metrics for evaluation.")
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icassp.yml")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", help="Run E1 only, without neural training")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--pretrain-steps", type=int)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--max-eval-points", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--predict-checkpoint")
    parser.add_argument("--prediction-output")
    args = parser.parse_args()
    if args.predict_checkpoint:
        if not args.prediction_output:
            parser.error("--predict-checkpoint requires --prediction-output")
        predict_checkpoint(args.predict_checkpoint, args.prediction_output, args.device or "auto")
        return
    with open(args.config) as file:
        config = yaml.safe_load(file)
    for name in ("pretrain_steps", "target_steps", "max_eval_points", "device"):
        value = getattr(args, name)
        if value is not None:
            config["train"][name] = value
    if args.output_dir:
        config["output_dir"] = args.output_dir
    run(config, args.scenes, [] if args.baseline_only else args.methods,
        args.include_baselines or args.baseline_only, args.inspect)


if __name__ == "__main__":
    main()
