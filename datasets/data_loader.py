"""Read the preprocessed ICASSP paired path-loss maps without altering them.

Coordinates are local image coordinates: x = column * 0.25 m and
y = row * 0.25 m (no image flip or building-world-coordinate offset).
Documentation.pdf defines f2 as 1.8 GHz and f3 as 3.5 GHz. The PNG values
are retained as quantized dB values, not divided by 255. The loader reads only
the paired path-loss maps and coordinate metadata, never material/input PNGs.
All raster locations are eligible except exactly coincident Tx/Rx locations,
where FSPL is undefined. Receiver selection depends only on coordinates and
positive distances are not clipped.
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
    building_bounds_m: Mapping[str, float] | None = None

    @property
    def coordinate_frame(self) -> str:
        """Tx and Rx both use the already-cropped image's local raster frame."""
        return "local_raster"

    @property
    def raster_coverage_m(self) -> dict[str, float]:
        """W/H pixel footprints; differs by one step from sample-point span."""
        return {"width": self.width * self.pixel_size_m,
                "height": self.height * self.pixel_size_m}

    @property
    def grid_bounds_m(self) -> dict[str, float]:
        """Bounds of local sample coordinates, not the raw world geometry box.

        W/H are authoritative for the already-cropped PNGs. Computing the last
        sample as an integer number of steps avoids fractional cells and the
        extra endpoint introduced by arange over rounded world bounds.
        """
        return {"x_min": 0.0, "x_max": self.width_m,
                "y_min": 0.0, "y_max": self.height_m}

    @property
    def tx_pixel_index(self) -> tuple[int, int]:
        """Validated Tx index as (row, column) in the paired raster."""
        col, row = np.rint(self.tx_m[:2] / self.pixel_size_m).astype(int)
        return int(row), int(col)

    @property
    def width_m(self) -> float:
        """Distance between first/last pixel centers for coordinate scaling.

        Building metadata bounds describe the original geometry and can differ
        slightly from the raster extent; sampling must use this raster grid.
        """
        return (self.width - 1) * self.pixel_size_m

    @property
    def height_m(self) -> float:
        """Distance between first/last row centers for coordinate scaling."""
        return (self.height - 1) * self.pixel_size_m


def _rows_by_scene(path: Path) -> dict[str, dict]:
    with path.open(newline="") as file:
        return {row["scene_id"]: row for row in csv.DictReader(file)}


def discover_scenes(root: str | Path) -> list[str]:
    """Return the preprocessed scenes listed in positions.csv."""
    return sorted(_rows_by_scene(Path(root) / "positions.csv"))


def _read_pl(path: Path) -> np.ndarray:
    """Read the dense PL raster without consulting any material image."""
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float32)


def compute_pl_minmax(root: str | Path, band_frequencies_ghz: Mapping[str, float]) -> dict:
    """Collect known PL ranges over all scenes and pixels, separately per band.

    The complete dataset supplies these calibration statistics, including
    eventual test locations. The raw images and labels are never rewritten.
    """
    root = Path(root)
    scenes = discover_scenes(root)
    bands = {}
    for band, frequency in band_frequencies_ghz.items():
        minimum, maximum = float('inf'), float('-inf')
        for scene in scenes:
            values = _read_pl(root / scene / f'{band}_path_loss.png')
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
        bands[band] = dict(frequency_ghz=float(frequency), min_db=minimum, max_db=maximum)
    return dict(method='minmax', scope='full_dataset_per_band', includes_test_labels=True,
                n_scenes=len(scenes), bands=bands)


def build_local_receiver_grid(width: int, height: int, tx_m: np.ndarray) -> np.ndarray:
    """Construct the local quarter-metre raster at the supplied Tx height."""
    rows, cols = np.indices((height, width), dtype=np.float32)
    return np.stack((cols * PIXEL_SIZE_M, rows * PIXEL_SIZE_M,
                     np.full_like(rows, tx_m[2])), axis=-1).reshape(-1, 3)


def load_scene(
    root: str | Path,
    scene_id: str,
    source_band: str = "f2",
    target_band: str = "f3",
    *,
    band_frequencies_ghz: Mapping[str, float] | None = None,
) -> SceneData:
    """Load a scene;

    ``valid_mask`` is independent of labels. The returned PL maps stay in dB;
    model fitting uses the training slice. Dataset-wide band ranges are
    collected separately by ``compute_pl_minmax`` for known calibration.
    """
    root = Path(root)
    frequencies = dict(BAND_FREQUENCIES_GHZ)
    frequencies.update(band_frequencies_ghz or {})
    positions = _rows_by_scene(root / "positions.csv")
    buildings = _rows_by_scene(root / "building_details.csv")
    meta = buildings[scene_id]
    source_pl = _read_pl(root / scene_id / f"{source_band}_path_loss.png")
    target_pl = _read_pl(root / scene_id / f"{target_band}_path_loss.png")
    if source_pl.shape != target_pl.shape:
        raise ValueError(f"Paired path-loss map dimensions differ for {scene_id}")
    height, width = source_pl.shape
    tx = np.array([float(positions[scene_id][f"tx_{axis}_m"]) for axis in "xyz"], dtype=np.float64)
    # Check the coordinate convention once at the loading boundary. Never move
    # Tx or reinterpret world bounds to make an inconsistent position fit.
    col, row = tx[:2] / PIXEL_SIZE_M
    if (not np.array_equal(tx[:2], np.rint(tx[:2] / PIXEL_SIZE_M) * PIXEL_SIZE_M)
            or not (0 <= col < width and 0 <= row < height)):
        raise ValueError(f"Tx must lie on the local 0.25 m raster for {scene_id}")

    # The PNGs are already cropped. Local bounds are 0..(W-1)*0.25 and
    # 0..(H-1)*0.25; rounding raw world bounds would change the raster size.
    rx = build_local_receiver_grid(width, height, tx)
    tx = tx.astype(np.float32)
    # Derive exact distances only from the Tx/Rx coordinates.
    distance = np.linalg.norm(rx - tx[None], axis=-1)
    return SceneData(
        scene_id=scene_id, building_id=meta["building_id"], tx_m=tx, rx_m=rx,
        distance_m=distance, source_pl_db=source_pl.reshape(-1),
        target_pl_db=target_pl.reshape(-1), valid_mask=distance > 0,
        height=height, width=width,
        source_band=source_band, target_band=target_band,
        source_frequency_ghz=float(frequencies[source_band]), target_frequency_ghz=float(frequencies[target_band]),
        building_bounds_m={key.lower(): float(meta[key])
                           for key in ("X_min", "X_max", "Y_min", "Y_max")},
    )
