"""Feature extraction for nuclei and ROI-based spatial metrics."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.prepared import prep

from skimage.color import rgb2hsv
from skimage.draw import polygon as draw_polygon
from skimage.feature import graycomatrix, graycoprops

from eyemelanoma.config import FeatureConfig


def _normalize_mpp(value: object) -> Optional[float]:
    """Normalize an mpp value to a float, handling tuples or arrays."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return float(np.nanmean(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lookup_slide_property(properties: object, key: str) -> Optional[str]:
    """Safely retrieve a slide property from dict-like or attribute-based stores."""
    if properties is None:
        return None
    if hasattr(properties, "get"):
        try:
            return properties.get(key)
        except Exception:
            return None
    try:
        return properties[key]
    except Exception:
        pass
    candidate_keys = {
        key.replace(".", "_").replace("-", "_"),
        key.replace("openslide.", "").replace("-", "_"),
    }
    for candidate in candidate_keys:
        try:
            return getattr(properties, candidate)
        except Exception:
            continue
    return None


def _resolve_base_mpp(wsi) -> float:
    """
    Resolve the base microns-per-pixel (mpp) for a WSI object.

    Falls back to the OpenSlide property if available, otherwise defaults to 0.25.
    """
    mpp = _normalize_mpp(getattr(wsi, "mpp", None))
    if mpp is not None and not np.isnan(mpp):
        return float(mpp)
    reader = getattr(wsi, "reader", None)
    properties = getattr(reader, "properties", None)
    prop_val = _lookup_slide_property(properties, "openslide.mpp-x")
    mpp = _normalize_mpp(prop_val)
    return float(mpp) if mpp is not None else 0.25


def polygon_to_mask(poly: Polygon, height: int, width: int) -> np.ndarray:
    """Rasterize a polygon into a boolean mask."""
    x, y = poly.exterior.xy
    rr, cc = draw_polygon(np.array(y), np.array(x), (height, width))
    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True
    return mask


def hsv_melanin_fraction(rgb_patch: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate melanin fraction using HSV heuristics.

    Brownish pixels (15°–45° hue) or very dark pixels are counted as melanin.
    """
    if rgb_patch.dtype != np.float32 and rgb_patch.dtype != np.float64:
        arr = rgb_patch.astype(np.float32) / 255.0
    else:
        arr = rgb_patch
    hsv = rgb2hsv(arr)
    hue = hsv[..., 0] * 360.0
    sat = hsv[..., 1]
    val = hsv[..., 2]
    brown = (sat > 0.5) & (hue >= 15.0) & (hue <= 45.0)
    dark = val < 0.35
    mel_mask = (brown | dark) & mask
    total = mask.sum()
    return float(mel_mask.sum()) / float(total) if total > 0 else 0.0


def texture_features(gray_u8: np.ndarray, mask: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
    """Compute GLCM texture features inside a masked patch."""
    arr = gray_u8.copy()
    arr[~mask] = 0
    glcm = graycomatrix(
        arr,
        distances=config.glcm_distances,
        angles=config.glcm_angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
        val = graycoprops(glcm, prop)
        features[prop] = float(np.nanmean(val))
    return features


def geometry_features(poly: Polygon, mpp: float) -> Dict[str, float]:
    """Compute basic morphology features in microns."""
    area_px = poly.area
    perim_px = poly.length
    area_um2 = area_px * (mpp**2)
    perim_um = perim_px * mpp
    circularity = 4 * math.pi * area_px / (perim_px**2) if perim_px > 0 else 0.0
    convex_area = poly.convex_hull.area if poly.convex_hull.area > 0 else np.nan
    solidity = float(area_px / convex_area) if convex_area and convex_area > 0 else 0.0
    return {
        "area_um2": float(area_um2),
        "perimeter_um": float(perim_um),
        "circularity": float(circularity),
        "solidity": float(solidity),
    }


def filter_cells_in_roi(cell_gdf, roi_polygon: Polygon) -> pd.DataFrame:
    """Filter cells with centroids inside the ROI polygon."""
    roi_prepared = prep(roi_polygon)
    mask = cell_gdf.geometry.centroid.apply(lambda pt: roi_prepared.contains(pt))
    return cell_gdf.loc[mask].copy()


def read_cell_patch_rgba(wsi, bbox_xyxy: Tuple[int, int, int, int], level: Optional[int], color_mpp: float) -> Tuple[np.ndarray, float]:
    """Read an RGB patch from the slide and return patch + effective mpp."""
    reader = getattr(wsi, "reader", None)
    if reader is None:
        raise RuntimeError("WSI reader not found on wsi object.")

    downs = getattr(reader, "level_downsamples", [1.0])
    base_mpp = _resolve_base_mpp(wsi)

    if level is None:
        target_down = max(color_mpp / float(base_mpp), 1.0)
        level = int(np.argmin([abs(d - target_down) for d in downs]))
    level = max(0, min(int(level), len(downs) - 1))
    effective_mpp = float(base_mpp) * float(downs[level])

    x0, y0, x1, y1 = bbox_xyxy
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    scale = float(downs[level])
    width_level = max(1, int(math.ceil(width / scale)))
    height_level = max(1, int(math.ceil(height / scale)))

    tile = None
    if hasattr(wsi, "read_region"):
        tile = wsi.read_region(int(x0), int(y0), int(width_level), int(height_level), level=int(level))
    elif hasattr(reader, "get_region"):
        tile = reader.get_region(int(x0), int(y0), int(width_level), int(height_level), level=int(level))
    elif hasattr(reader, "read_region"):
        tile = reader.read_region((int(x0), int(y0)), int(level), (int(width_level), int(height_level)))

    if tile is None:
        raise AttributeError("Could not read region from WSI.")

    from PIL import Image as PILImage

    if isinstance(tile, PILImage.Image):
        arr = np.array(tile.convert("RGB"))
    else:
        arr = np.asarray(tile)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr, effective_mpp


def extract_nuclei_features(wsi, cell_gdf, config: FeatureConfig) -> pd.DataFrame:
    """Compute per-nucleus morphology, color, and texture features."""
    rows = []
    base_mpp = _resolve_base_mpp(wsi)

    for idx, row in cell_gdf.iterrows():
        poly: Polygon = row.geometry
        if poly is None or poly.is_empty:
            continue

        geom = geometry_features(poly, base_mpp)
        record = {
            "cell_id": idx,
            "cell_type": row.get("class", row.get("cell_type", "Unknown")),
            "mpp_eff": float(base_mpp),
            **geom,
        }

        minx, miny, maxx, maxy = poly.bounds
        minx = int(math.floor(minx)) - config.patch_expansion_px
        miny = int(math.floor(miny)) - config.patch_expansion_px
        maxx = int(math.ceil(maxx)) + config.patch_expansion_px
        maxy = int(math.ceil(maxy)) + config.patch_expansion_px

        local_poly = translate(poly, xoff=-minx, yoff=-miny)
        rgb_patch, eff_mpp = read_cell_patch_rgba(
            wsi, (minx, miny, maxx, maxy), config.reader_level_for_color, config.color_mpp
        )
        height, width, _ = rgb_patch.shape
        mask = polygon_to_mask(local_poly, height, width)

        melanin = hsv_melanin_fraction(rgb_patch, mask)
        masked = rgb_patch.copy()
        masked[~mask] = 0
        mean_rgb = masked.sum(axis=(0, 1)) / (mask.sum() + 1e-6)

        gray = (0.299 * rgb_patch[..., 0] + 0.587 * rgb_patch[..., 1] + 0.114 * rgb_patch[..., 2]).astype(np.uint8)
        tex = texture_features(gray, mask, config)

        record.update(
            {
                "mpp_eff": float(eff_mpp),
                "mean_R": float(mean_rgb[0]),
                "mean_G": float(mean_rgb[1]),
                "mean_B": float(mean_rgb[2]),
                "melanin_fraction": float(melanin),
                **{f"tex_{k}": v for k, v in tex.items()},
            }
        )
        rows.append(record)

    return pd.DataFrame(rows)


def add_spatial_features(df: pd.DataFrame, roi_polygon: Polygon, k_neighbors: int = 5) -> pd.DataFrame:
    """
    Add spatial features based on centroid density and neighborhood composition.
    """
    if df.empty:
        return df

    if "centroid_x" not in df.columns or "centroid_y" not in df.columns:
        raise ValueError("Spatial features require centroid_x and centroid_y columns.")

    coords = df[["centroid_x", "centroid_y"]].to_numpy()
    roi_area = roi_polygon.area if roi_polygon and not roi_polygon.is_empty else np.nan

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("scikit-learn is required for spatial features.") from exc

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(coords)))
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # exclude self distance at index 0
    k_dist = distances[:, -1]
    local_density = np.where(k_dist > 0, k_neighbors / (math.pi * (k_dist ** 2)), 0.0)
    df["local_density"] = local_density
    df["roi_density"] = len(df) / roi_area if roi_area and roi_area > 0 else np.nan

    if "cell_type" in df.columns:
        cell_types = df["cell_type"].astype(str).tolist()
        neighbor_types = []
        for row_indices in indices:
            # Skip self (first index)
            neighbors = [cell_types[idx] for idx in row_indices[1:]]
            neighbor_types.append(neighbors)
        # Example: fraction of same-type neighbors
        same_type_frac = []
        for current, neigh in zip(cell_types, neighbor_types):
            if not neigh:
                same_type_frac.append(np.nan)
            else:
                same_type_frac.append(sum(n == current for n in neigh) / len(neigh))
        df["same_type_neighbor_fraction"] = same_type_frac

    return df
