"""Step 1 for NeRF-WESN: export paired ICASSP Task 2 links and 20/80 splits.

Run with --list-pairs to inspect CSV coordinate matches, or --analyze-only
to validate the PNGs and write a report of each pair and its tensor shapes.
Use --building B1 to limit the analysis or export to the first building.
The two bands share Rx coordinates and split indices. Ray directions point
from Tx to Rx. Coordinates are local to each building's image, in metres.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


TASK = "Task_2_ICASSP"
FREQUENCY_HZ = {"f1": 868_000_000, "f2": 1_800_000_000, "f3": 3_500_000_000}
SPACING_M = 0.25
HEIGHT_M = 1.5


@dataclass(frozen=True)
class PairedSite:
    building: str
    tx_rc: tuple[int, int]
    bands: tuple[str, str]
    sample_ids: tuple[int, int]

    @property
    def stems(self) -> tuple[str, str]:
        a, b = self.bands
        i, j = self.sample_ids
        return f"{self.building}_Ant1_{a}_S{i}", f"{self.building}_Ant1_{b}_S{j}"


# 1. Match source and target maps by building and actual Tx pixel.
def read_positions(path: Path) -> dict[tuple[int, int], int]:
    """Return (row, col) -> sample index, keeping the first duplicate."""
    positions = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"X", "Y"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected columns X and Y in {path}")
        for sample_id, row in enumerate(reader):
            if row.get("") not in (None, "", str(sample_id)):
                raise ValueError(f"Unexpected sample index on row {sample_id} in {path}")
            pixel = (int(row["X"]), int(row["Y"]))
            positions.setdefault(pixel, sample_id)
    return positions


def find_pairs(root: Path, source: str, target: str, building: str | None) -> list[PairedSite]:
    if source == target:
        raise ValueError("Choose two distinct frequency bands")
    positions_dir = root / "Positions"
    if not positions_dir.is_dir():
        raise FileNotFoundError(f"Missing Positions directory: {positions_dir}")
    pattern = re.compile(r"Positions_(B\d+)_Ant1_f[123]\.csv")
    buildings = {
        match[1] for path in positions_dir.iterdir() if (match := pattern.fullmatch(path.name))
    }
    if building is not None:
        buildings = {building}
    pairs = []
    for name in sorted(buildings, key=lambda value: int(value[1:])):
        a = read_positions(positions_dir / f"Positions_{name}_Ant1_{source}.csv")
        b = read_positions(positions_dir / f"Positions_{name}_Ant1_{target}.csv")
        for pixel in sorted(a.keys() & b.keys()):
            pairs.append(PairedSite(name, pixel, (source, target), (a[pixel], b[pixel])))
    return pairs


# 2. Read the original PNG grid and check the CSV/image coordinate convention.
def read_png(folder: Path, stem: str, mode: str) -> np.ndarray:
    path = folder / f"{stem}.png"
    if not path.exists():
        path = path.with_suffix(".PNG")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != mode:
            raise ValueError(f"Expected a {mode} PNG: {path}")
        return np.array(image)


def read_site(root: Path, site: PairedSite) -> tuple[np.ndarray, np.ndarray]:
    """Return material maps [2,H,W,2] and path-loss maps [2,H,W]."""
    materials, losses = [], []
    row, col = site.tx_rc
    for stem in site.stems:
        rgb = read_png(root / "Inputs" / TASK, stem, "RGB")
        pl = read_png(root / "Outputs" / TASK, stem, "L")
        height, width = pl.shape
        if rgb.shape[:2] != pl.shape or not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"Image shape or Tx pixel mismatch for {stem}")
        rr, cc = np.ogrid[:height, :width]
        distance_m = SPACING_M * np.hypot(rr - row, cc - col)
        # Blue stores distance rounded down to integer metres in this release.
        if not np.array_equal(rgb[..., 2], np.floor(distance_m)):
            raise ValueError(f"Tx row/column or distance-channel mismatch for {stem}")
        if pl.min() < 13 or pl.max() > 160:
            raise ValueError(f"Unexpected path-loss PNG values in {stem}: {pl.min()}..{pl.max()}")
        materials.append(rgb[..., :2])
        losses.append(pl)
    return np.stack(materials), np.stack(losses)


# 3. Select exactly the same Rx pixels for both bands, then construct rays.
def split_rx_pixels(
    shape: tuple[int, int], site: PairedSite, seed: int, mask_dir: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(shape, dtype=bool)
    if mask_dir is not None:
        path = mask_dir / f"{site.building}.npy"
        mask = np.load(path, allow_pickle=False)
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError(f"Expected a boolean mask of shape {shape} in {path}")
    mask[site.tx_rc] = False  # A zero-length Tx-to-Tx link has no unit direction.
    pixels = np.argwhere(mask).astype(np.int32)
    n_train = len(pixels) // 5  # 20%, rounded down to a whole pixel.
    if n_train < 1:
        raise ValueError(f"Fewer than five eligible Rx pixels in {site.building}")
    rng = np.random.default_rng([seed, int(site.building[1:]), *site.tx_rc])
    order = rng.permutation(len(pixels))
    return pixels[order[:n_train]], pixels[order[n_train:]], mask


def pixels_to_xyz_m(pixels_rc: np.ndarray) -> np.ndarray:
    """Convert row/column indices to local Cartesian x=col, y=row, z=1.5 m."""
    xyz = np.full((len(pixels_rc), 3), HEIGHT_M, dtype=np.float32)
    xyz[:, :2] = pixels_rc[:, ::-1] * SPACING_M
    return xyz


def build_split(pixels: np.ndarray, tx_xyz_m: np.ndarray, path_loss: np.ndarray) -> dict[str, np.ndarray]:
    rx = pixels_to_xyz_m(pixels)
    delta = rx - tx_xyz_m
    length = np.linalg.norm(delta, axis=-1)
    row, col = pixels.T
    return {
        "rx_rc": pixels,
        "rx_xyz_m": rx,
        "ray_o_m": np.broadcast_to(tx_xyz_m, rx.shape),
        "ray_d": delta / length[:, None],
        "ray_length_m": length,
        "path_loss_db": path_loss[:, row, col, None].astype(np.float32),
    }


# 4. Sample only the rays in the current batch when connecting the model.
def sample_ray_batch(
    ray_o_m: np.ndarray,
    ray_d: np.ndarray,
    ray_length_m: np.ndarray,
    material_map: np.ndarray,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Return pts/view/tx [B,K,3], distance [B,K], and materials [B,K,2].

    Pass one band of geometry.npz's materials_db. Use a fixed n_steps for
    both bands, at least as large as the WESN window. This utility samples
    the direct segment and looks up the nearest original geometry pixel.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    fraction = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    distance = ray_length_m[:, None] * fraction
    pts = ray_o_m[:, None, :] + distance[..., None] * ray_d[:, None, :]
    rc = np.rint(pts[..., [1, 0]] / SPACING_M).astype(np.intp)
    return {
        "pts": pts,
        "view": np.broadcast_to(ray_d[:, None, :], pts.shape).copy(),
        "tx": np.broadcast_to(ray_o_m[:, None, :], pts.shape).copy(),
        "distance_along_ray_m": distance,
        "materials_db": material_map[rc[..., 0], rc[..., 1]].astype(np.float32),
    }


def prepare_site(
    root: Path, site: PairedSite, seed: int, mask_dir: Path | None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    materials, losses = read_site(root, site)
    train, test, mask = split_rx_pixels(losses.shape[1:], site, seed, mask_dir)
    tx = pixels_to_xyz_m(np.array([site.tx_rc]))[0]
    geometry = {
        "materials_db": materials,
        "rx_mask": mask,
        "tx_rc": np.array(site.tx_rc, dtype=np.int32),
        "tx_xyz_m": tx,
        "frequency_hz": np.array([FREQUENCY_HZ[band] for band in site.bands]),
        "band": np.array(site.bands),
        "map_stem": np.array(site.stems),
        "scene_id": np.array(site.building),
        "spacing_m": np.array(SPACING_M, dtype=np.float32),
        "seed": np.array(seed, dtype=np.int64),
    }
    return geometry, build_split(train, tx, losses), build_split(test, tx, losses)


def write_site(output_dir: Path, site: PairedSite, seed: int, geometry: dict, train: dict, test: dict) -> Path:
    a, b = site.stems
    destination = output_dir / f"{a}__{b}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Publish a complete site in one rename. Keep earlier exports available.
    with TemporaryDirectory(dir=output_dir) as tmp:
        staging = Path(tmp) / "site"
        staging.mkdir()
        np.savez_compressed(staging / "geometry.npz", **geometry)
        np.savez_compressed(staging / "train.npz", **train)
        np.savez_compressed(staging / "test.npz", **test)
        try:
            staging.rename(destination)
        except OSError as exc:
            if destination.exists():
                raise FileExistsError(f"Export already exists: {destination}. Use another --output-dir to keep both.") from exc
            raise
    return destination


# 5. Print coordinate matches and report shapes from the arrays actually built.
def describe_pair(site: PairedSite) -> str:
    source_hz, target_hz = (FREQUENCY_HZ[band] for band in site.bands)
    a, b = site.stems
    return (
        f"{a} ({source_hz / 1e9:g} GHz) / {b} ({target_hz / 1e9:g} GHz), "
        f"Tx (row, col) = {site.tx_rc}"
    )


def analyze_site(site: PairedSite, geometry: dict, train: dict, test: dict) -> dict:
    height, width = geometry["rx_mask"].shape
    x, y, z = geometry["tx_xyz_m"]
    n_train, n_test = len(train["rx_rc"]), len(test["rx_rc"])
    train_pl, test_pl = train["path_loss_db"], test["path_loss_db"]
    pl_min = np.minimum(train_pl.min(axis=(1, 2)), test_pl.min(axis=(1, 2)))
    pl_max = np.maximum(train_pl.max(axis=(1, 2)), test_pl.max(axis=(1, 2)))
    print(describe_pair(site))
    print(f"  Tx (x, y, z) = ({x:g}, {y:g}, {z:g}) m")
    print(f"  Original PNG per band: RGB {(height, width, 3)}, PL {(height, width)}")
    print(f"  Rx points shared by both bands: train={n_train:,}, test={n_test:,}")
    for filename, fields in (("geometry.npz", geometry), ("train.npz", train), ("test.npz", test)):
        shapes = ", ".join(f"{name}={value.shape} [{value.dtype}]" for name, value in fields.items())
        print(f"  {filename}: {shapes}")
    for i, band in enumerate(site.bands):
        print(f"  {band} PL on eligible Rx grid: {pl_min[i]:g} to {pl_max[i]:g} dB")

    return {
        "building": site.building,
        "source_band": site.bands[0],
        "source_frequency_hz": int(geometry["frequency_hz"][0]),
        "source_sample": f"S{site.sample_ids[0]}",
        "source_map": site.stems[0],
        "target_band": site.bands[1],
        "target_frequency_hz": int(geometry["frequency_hz"][1]),
        "target_sample": f"S{site.sample_ids[1]}",
        "target_map": site.stems[1],
        "tx_row": site.tx_rc[0],
        "tx_col": site.tx_rc[1],
        "tx_x_m": float(x),
        "tx_y_m": float(y),
        "tx_z_m": float(z),
        "height": height,
        "width": width,
        "eligible_rx": n_train + n_test,
        "train_rx": n_train,
        "test_rx": n_test,
        "materials_shape": geometry["materials_db"].shape,
        "train_rx_xyz_shape": train["rx_xyz_m"].shape,
        "test_rx_xyz_shape": test["rx_xyz_m"].shape,
        "train_path_loss_shape": train_pl.shape,
        "test_path_loss_shape": test_pl.shape,
        "source_pl_min_db": float(pl_min[0]),
        "source_pl_max_db": float(pl_max[0]),
        "target_pl_min_db": float(pl_min[1]),
        "target_pl_max_db": float(pl_max[1]),
        "seed": int(geometry["seed"]),
    }


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        staging = Path(tmp) / path.name
        with staging.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        staging.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing Inputs, Outputs, and Positions")
    parser.add_argument("--source", choices=FREQUENCY_HZ, default="f2")
    parser.add_argument("--target", choices=FREQUENCY_HZ, default="f3")
    parser.add_argument("--building", help="Limit processing to one building, e.g. B1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("processed_task2"))
    parser.add_argument("--mask-dir", type=Path, help="Optional building masks, e.g. B1.npy, boolean [H,W]")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-pairs", action="store_true", help="List Tx coordinate matches using the CSV files only")
    mode.add_argument("--analyze-only", action="store_true", help="Validate PNGs and write the pair and shape report without NPZ export")
    args = parser.parse_args()
    try:
        pairs = find_pairs(args.data_root, args.source, args.target, args.building)
        print(f"{len(pairs)} matched Tx sites across {len({site.building for site in pairs})} buildings")
        rows = []
        for site in pairs:
            if args.list_pairs:
                print(describe_pair(site))
            else:
                arrays = prepare_site(args.data_root, site, args.seed, args.mask_dir)
                rows.append(analyze_site(site, *arrays))
                if not args.analyze_only:
                    destination = write_site(args.output_dir, site, args.seed, *arrays)
                    print(f"  Export: {destination}")
        if rows:
            scope = args.building or "all"
            report = args.output_dir / f"pair_report_{args.source}_{args.target}_{scope}_seed{args.seed}.csv"
            write_report(report, rows)
            print(f"Pair report: {report}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
