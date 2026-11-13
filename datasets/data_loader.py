"""Read the preprocessed ICASSP paired path-loss maps without altering them.

Coordinates are local image coordinates: x = column * 0.25 m and
y = row * 0.25 m (no image flip or building-world-coordinate offset).
Documentation.pdf defines f2 as 1.8 GHz and f3 as 3.5 GHz. The PNG values
are retained as quantized dB values, not divided by 255. No official valid
receiver mask is supplied, so all pixels are valid, including PL=160.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Mapping

import numpy as np
from PIL import Image


BAND_FREQUENCIES_GHZ = {"f2": 1.8, "f3": 3.5}
PIXEL_SIZE_M = 0.25


@dataclass(frozen=True)
class SceneData:
    scene_id: str
    building_id: str
    tx_m: np.ndarray
    rx_m: np.ndarray
    geometry: np.ndarray  # target-band H x W x 2 reflectance/transmittance dB
    source_geometry: np.ndarray
    distance_m: np.ndarray  # exact coordinate-derived distance, not quantized PNG
    source_pl_db: np.ndarray
    target_pl_db: np.ndarray
    valid_mask: np.ndarray
    height: int
    width: int
    source_band: str
    target_band: str
    source_frequency_ghz: float
    target_frequency_ghz: float
    pixel_size_m: float = PIXEL_SIZE_M

    @property
    def width_m(self) -> float:
        """Distance between first/last pixel centers for align_corners=True.

        Building metadata bounds describe the original geometry and can differ
        slightly from the raster extent; sampling must use this raster grid.
        """
        return (self.width - 1) * self.pixel_size_m

    @property
    def height_m(self) -> float:
        """Distance between first/last row centers for align_corners=True."""
        return (self.height - 1) * self.pixel_size_m


def _rows_by_scene(path: Path) -> dict[str, dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or "scene_id" not in rows[0]:
        raise ValueError(f"Missing scene_id metadata in {path}")
    result = {}
    for row in rows:
        scene = row["scene_id"]
        if not scene or scene in result:
            raise ValueError(f"Duplicate/empty scene_id in {path}: {scene!r}")
        result[scene] = row
    return result


def discover_scenes(root: str | Path) -> list[str]:
    """Return metadata-declared scenes, requiring corresponding directories."""
    root = Path(root)
    positions = _rows_by_scene(root / "positions.csv")
    buildings = _rows_by_scene(root / "building_details.csv")
    if set(positions) != set(buildings):
        raise ValueError("Scene sets differ between positions and building metadata")
    for scene in positions:
        if Path(scene).name != scene or not (root / scene).is_dir():
            raise ValueError(f"Missing or invalid scene directory: {scene!r}")
    return sorted(positions)


def _read_input(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.shape != (*expected_shape, 3) or values.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image {expected_shape} in {path}, got {values.shape}/{values.dtype}")
    return values.astype(np.float32)


def _read_pl(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.shape != expected_shape or values.dtype != np.uint8:
        raise ValueError(f"Expected uint8 grayscale image {expected_shape} in {path}, got {values.shape}/{values.dtype}")
    return values.astype(np.float32).reshape(-1)


def load_scene(
    root: str | Path,
    scene_id: str,
    source_band: str = "f2",
    target_band: str = "f3",
    *,
    band_frequencies_ghz: Mapping[str, float] | None = None,
) -> SceneData:
    """Load a scene;

    ``valid_mask`` is independent of labels. The returned PL maps are dense
    ground truth; callers must partition observation/train/test indices before
    reading labels for a model, normalization, or checkpoint selection.
    """
    root = Path(root)
    if Path(scene_id).name != scene_id or scene_id in {".", ".."}:
        raise ValueError(f"Invalid scene_id {scene_id!r}")
    frequencies = dict(BAND_FREQUENCIES_GHZ)
    frequencies.update(band_frequencies_ghz or {})
    for band in (source_band, target_band):
        if band not in frequencies or Path(band).name != band:
            raise ValueError(f"Unknown frequency band: {band!r}")
        if not np.isfinite(frequencies[band]) or frequencies[band] <= 0:
            raise ValueError(f"Frequency for {band} must be positive")
    positions = _rows_by_scene(root / "positions.csv")
    buildings = _rows_by_scene(root / "building_details.csv")
    if scene_id not in positions or scene_id not in buildings:
        raise ValueError(f"Missing scene metadata: {scene_id}")
    meta = buildings[scene_id]
    height, width = int(meta["H"]), int(meta["W"])
    # Tx pos
    tx = np.array([float(positions[scene_id][f"tx_{axis}_m"]) for axis in "xyz"], dtype=np.float32)

    # Rx pos
    rows, cols = np.indices((height, width), dtype=np.float32)
    rx = np.stack((cols * PIXEL_SIZE_M, rows * PIXEL_SIZE_M, np.full_like(rows, tx[2])), axis=-1).reshape(-1, 3)
    # Use unquantized coordinates for ray length; PNG's third channel is floored m.
    distance = np.linalg.norm(rx - tx[None], axis=-1)
    source_input = _read_input(root / scene_id / f"{source_band}_input.png", (height, width))
    target_input = _read_input(root / scene_id / f"{target_band}_input.png", (height, width))
    if not np.array_equal(source_input[..., 2], target_input[..., 2]):
        raise ValueError(f"Paired distance maps differ for {scene_id}; transmitter alignment is invalid")
    if not np.allclose(source_input[..., 2].reshape(-1), np.floor(distance), atol=1):
        raise ValueError(f"Distance image is inconsistent with transmitter/grid metadata for {scene_id}")
    return SceneData(
        scene_id=scene_id, building_id=meta["building_id"], tx_m=tx, rx_m=rx,
        geometry=target_input[..., :2], source_geometry=source_input[..., :2],
        distance_m=distance, source_pl_db=_read_pl(root / scene_id / f"{source_band}_path_loss.png", (height, width)),
        target_pl_db=_read_pl(root / scene_id / f"{target_band}_path_loss.png", (height, width)),
        valid_mask=np.ones(height * width, dtype=bool), height=height, width=width,
        source_band=source_band, target_band=target_band,
        source_frequency_ghz=float(frequencies[source_band]), target_frequency_ghz=float(frequencies[target_band]),
    )
