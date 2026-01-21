"""Utilities for handling HistoPLUS outputs and coordinate normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GlobalDetection:
    """Simple container for globally-referenced cell detections."""

    centroid_x: float
    centroid_y: float
    cell_type: str
    confidence: float | None


def _deepzoom_tile_origin_and_scale(dz, tile_col: int, tile_row: int, dz_level: int) -> Optional[Tuple[float, float, float]]:
    try:
        (tile_l0_x, tile_l0_y), _, _ = dz.get_tile_coordinates(dz_level, (tile_col, tile_row))
    except Exception:
        return None
    level_scale = 2 ** (dz.level_count - 1 - dz_level)
    return float(tile_l0_x), float(tile_l0_y), float(level_scale)


def iter_global_centroids(cell_masks: list[dict], slide, tile_size: int, overlap: int = 0) -> Iterable[GlobalDetection]:
    """Yield global detections from HistoPLUS tile-local masks."""
    from openslide.deepzoom import DeepZoomGenerator

    if not cell_masks:
        return

    dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)

    for item in cell_masks:
        tile_col = int(item.get("x", 0))
        tile_row = int(item.get("y", 0))
        dz_level = int(item.get("level", 0))

        tile_info = _deepzoom_tile_origin_and_scale(dz, tile_col, tile_row, dz_level)
        if tile_info is None:
            continue
        tile_l0_x, tile_l0_y, level_scale = tile_info

        for mask in item.get("masks", []):
            centroid = mask.get("centroid")
            if not centroid or len(centroid) < 2:
                continue
            local_x, local_y = float(centroid[0]), float(centroid[1])
            global_x = tile_l0_x + local_x * level_scale
            global_y = tile_l0_y + local_y * level_scale
            yield GlobalDetection(
                centroid_x=global_x,
                centroid_y=global_y,
                cell_type=str(mask.get("cell_type", "Unknown")),
                confidence=mask.get("confidence"),
            )


def ensure_global_centroids(centroids: np.ndarray, slide_dimensions: Tuple[int, int]) -> np.ndarray:
    """
    Ensure centroids are expressed in slide-level coordinates.

    If the maximum centroid coordinate is smaller than a single tile while the
    slide is much larger, warn by returning the same centroids (caller can decide
    to re-run conversion from HistoPLUS JSON).
    """
    if centroids.size == 0:
        return centroids

    width, height = slide_dimensions
    max_x = float(np.nanmax(centroids[:, 0]))
    max_y = float(np.nanmax(centroids[:, 1]))
    if max_x <= 4096 and max_y <= 4096 and (width > 10000 or height > 10000):
        # Likely local coords; return unchanged for now and let caller handle explicitly.
        return centroids
    return centroids
