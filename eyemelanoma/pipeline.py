"""End-to-end MRXS pipeline using LazySlide and HistoPLUS."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import geopandas as gpd
import lazyslide as zs
from wsidata import open_wsi
from wsidata.io._elems import subset_tiles

from eyemelanoma.config import PipelineConfig
from eyemelanoma.embeddings import run_embeddings_and_clustering
from eyemelanoma.features import (
    add_spatial_features,
    extract_nuclei_features,
    filter_cells_in_roi,
    stream_nuclei_features,
)
from eyemelanoma.roi import load_roi_from_mrxs_dir
from eyemelanoma.slide_vectors import (
    build_composition_matrix_from_paths,
    build_means_matrix_from_paths,
    celltype_distribution,
    celltype_feature_means,
)
from eyemelanoma.profiling import profile_resource_usage


def _parse_positive_int(value: str) -> Optional[int]:
    """Parse a positive integer from a string, returning None on failure."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_histoplus_batch(config: PipelineConfig) -> int:
    """
    Resolve the HistoPLUS batch size used for cell segmentation/classification.

    The environment variables EYEMELANOMA_HISTOPLUS_SEG_BATCH and
    EYEMELANOMA_HISTOPLUS_CLS_BATCH override defaults. When both are set, the
    smaller value is used to reduce peak memory usage on shared systems.
    """
    seg_env_val = os.getenv("EYEMELANOMA_HISTOPLUS_SEG_BATCH")
    cls_env_val = os.getenv("EYEMELANOMA_HISTOPLUS_CLS_BATCH")
    seg_env_batch = _parse_positive_int(seg_env_val) if seg_env_val is not None else None
    cls_env_batch = _parse_positive_int(cls_env_val) if cls_env_val is not None else None

    cfg_seg_batch = max(1, int(config.segmentation.seg_batch))
    cfg_cls_batch = max(1, int(config.segmentation.cls_batch))
    configured_batch = min(cfg_seg_batch, cfg_cls_batch)

    env_batches = [batch for batch in (seg_env_batch, cls_env_batch) if batch is not None]
    return min(env_batches) if env_batches else configured_batch


def _slide_data_dir(slide_path: Path) -> Optional[Path]:
    candidate = slide_path.parent / slide_path.stem
    if candidate.exists():
        return candidate
    candidate = slide_path.parent / slide_path.name
    return candidate if candidate.exists() else None


def list_slides(input_dir: Path, suffixes: tuple[str, ...]) -> List[Path]:
    slides = []
    for suffix in suffixes:
        slides.extend(sorted(input_dir.glob(f"**/*{suffix}")))
    return slides


def _ensure_cache_dirs(output_dir: Path) -> None:
    os.environ["HF_HOME"] = str(output_dir / "huggingface_cache")
    os.environ["XDG_CACHE_DIR"] = str(output_dir / "cache")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


def _roi_to_geodataframe(roi_polygon) -> gpd.GeoDataFrame:
    """
    Wrap an ROI polygon in a GeoDataFrame for wsidata compatibility.

    Parameters
    ----------
    roi_polygon
        Shapely polygon in slide-level coordinates.
    """
    return gpd.GeoDataFrame({"roi_id": [0]}, geometry=[roi_polygon])


def _select_cell_type_column(cell_gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Select the column containing cell-type labels."""
    if "cell_type" in cell_gdf.columns:
        return "cell_type"
    if "class" in cell_gdf.columns:
        return "class"
    return None


def _shrink_cell_gdf(cell_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Retain only geometry and cell-type columns to minimize memory usage.

    Parameters
    ----------
    cell_gdf
        GeoDataFrame with cell polygons.
    """
    assert "geometry" in cell_gdf.columns, "cell_gdf must include geometry column."
    cell_type_col = _select_cell_type_column(cell_gdf)
    keep_cols = ["geometry"] + ([cell_type_col] if cell_type_col else [])
    trimmed = cell_gdf[keep_cols].copy()
    if cell_type_col and cell_type_col != "cell_type":
        trimmed = trimmed.rename(columns={cell_type_col: "cell_type"})
    if "cell_type" not in trimmed.columns:
        trimmed["cell_type"] = "Unknown"
    return trimmed


def _drop_shape_keys(wsi, keys: List[str]) -> None:
    """Drop large shape entries from the WSI to free memory."""
    shapes = getattr(wsi, "shapes", None)
    if shapes is None:
        return
    for key in keys:
        if key in shapes:
            del shapes[key]


def _maybe_log_resource_usage(stage: str) -> None:
    """Optionally log resource usage to help diagnose memory hotspots."""
    if os.getenv("EYEMELANOMA_PROFILE_RESOURCES", "0") != "1":
        return
    profile = profile_resource_usage(sample_size_mb=1)
    print(
        f"[RESOURCE] {stage}: max_rss_mb={profile.max_rss_mb:.1f}, "
        f"io_write_mb_s={profile.io_write_mb_s:.1f}, io_read_mb_s={profile.io_read_mb_s:.1f}, "
        f"gpu_mem_mb={profile.gpu_memory_mb if profile.gpu_memory_mb is not None else 'n/a'}"
    )


def _subset_segmentation_tiles(wsi, source_key: str, indices: List[int], new_key: str) -> str:
    """
    Persist a subset of tiles for downstream segmentation steps.

    Parameters
    ----------
    wsi
        WSI object containing wsidata shapes and attrs.
    source_key
        Key in ``wsi.shapes`` containing the full tile set.
    indices
        Indices to keep in the tile dataframe.
    new_key
        Name of the new tile subset entry.
    """
    subset_tiles(wsi, source_key, indices, new_key=new_key)
    return new_key


def _extract_cells_with_histoplus(wsi, config: PipelineConfig, roi_polygon=None) -> None:
    zs.pp.find_tissues(wsi)
    if roi_polygon is not None:
        wsi.shapes["rois"] = _roi_to_geodataframe(roi_polygon)
        zs.pp.tile_tissues(
            wsi,
            config.tile.tile_px,
            overlap=config.tile.overlap,
            background_fraction=config.tile.background_fraction,
            mpp=config.tile.mpp_tiles,
            tissue_key="rois",
        )
        tiles = wsi.shapes.get("tiles")
        if tiles is None:
            raise ValueError("ROI tiling failed to create tiles for segmentation.")
        tile_key = _subset_segmentation_tiles(
            wsi,
            "tiles",
            list(range(len(tiles))),
            new_key="cell_segmentation_tiles",
        )
    else:
        zs.pp.tile_tissues(
            wsi,
            config.tile.tile_px,
            overlap=config.tile.overlap,
            background_fraction=config.tile.background_fraction,
            mpp=config.tile.mpp_tiles,
        )
        tile_key = "tiles"
    zs.seg.cell_types(
        wsi,
        model=config.segmentation.model,
        tile_key=tile_key,
        batch_size=_resolve_histoplus_batch(config),
        num_workers=0,
        amp=True,
        size_filter=False,
        nucleus_size=(20, 1000),
        pbar=True,
        key_added="cell_types",
    )
    _drop_shape_keys(wsi, ["tiles", "cell_segmentation_tiles"])


def process_slide(slide_path: Path, output_dir: Path, config: PipelineConfig) -> Dict:
    slide_out = output_dir / slide_path.stem
    slide_out.mkdir(parents=True, exist_ok=True)

    data_dir = _slide_data_dir(slide_path)
    roi_polygon = None
    if data_dir:
        try:
            roi_polygon = load_roi_from_mrxs_dir(data_dir)
        except Exception as exc:
            print(f"[WARN] Failed to parse ROI from {data_dir}: {exc}")

    wsi = open_wsi(str(slide_path))
    try:
        if "cell_types" not in wsi.shapes:
            _extract_cells_with_histoplus(wsi, config, roi_polygon=roi_polygon)
            try:
                wsi.write()
            except Exception:
                pass

        cell_gdf = _shrink_cell_gdf(wsi.shapes["cell_types"])
        try:
            slide_dims = getattr(wsi, "dimensions", None) or getattr(wsi.reader, "dimensions", None)
            if slide_dims:
                centroids = np.column_stack([cell_gdf.geometry.centroid.x, cell_gdf.geometry.centroid.y])
                assert centroids.ndim == 2 and centroids.shape[1] == 2, "Expected centroids shape (N, 2)."
                max_x = float(np.nanmax(centroids[:, 0])) if len(centroids) else 0.0
                max_y = float(np.nanmax(centroids[:, 1])) if len(centroids) else 0.0
                if max_x <= config.tile.tile_px and max_y <= config.tile.tile_px and (
                    slide_dims[0] > config.tile.tile_px * 2 or slide_dims[1] > config.tile.tile_px * 2
                ):
                    print(
                        "[WARN] HistoPLUS centroids appear to be tile-local. "
                        "Use histoplus_to_global.py to convert JSON outputs if needed."
                    )
        except Exception:
            pass
        if roi_polygon is not None:
            cell_gdf = filter_cells_in_roi(cell_gdf, roi_polygon)

        features_path = slide_out / "cells_features.csv"
        n_cells = int(len(cell_gdf))
        enable_spatial = config.features.enable_spatial_features
        use_spatial = enable_spatial and n_cells <= config.features.max_cells_for_spatial
        if enable_spatial and not use_spatial:
            print(
                "[WARN] Skipping spatial features to reduce memory usage. "
                f"Cell count {n_cells} exceeds max_cells_for_spatial={config.features.max_cells_for_spatial}."
            )

        if use_spatial:
            df_cells = extract_nuclei_features(wsi, cell_gdf, config.features)
            if roi_polygon is not None:
                df_cells = add_spatial_features(df_cells, roi_polygon)
            df_cells.to_csv(features_path, index=False)
            n_cells = int(len(df_cells))
            comp = celltype_distribution(df_cells)
            means = celltype_feature_means(df_cells)
            del df_cells
        else:
            summary = stream_nuclei_features(wsi, cell_gdf, config.features, features_path)
            n_cells = summary.total_cells
            comp = summary.to_distribution()
            means = summary.to_means_frame()
        _maybe_log_resource_usage(f"post_features:{slide_path.stem}")
    finally:
        close_fn = getattr(wsi, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        del wsi
        gc.collect()

    comp_path = slide_out / "celltype_distribution.csv"
    means_path = slide_out / "celltype_feature_means.csv"
    comp.to_csv(comp_path, header=["fraction"])
    means.to_csv(means_path, index=False)
    gc.collect()

    meta = {"slide": slide_path.name, "n_cells": n_cells, "roi_applied": bool(roi_polygon is not None)}
    with open(slide_out / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)

    return {
        "slide": slide_path.stem,
        "features_path": str(features_path),
        "comp_path": str(comp_path),
        "means_path": str(means_path),
    }


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    suffixes: tuple[str, ...] = (".mrxs",),
    config: PipelineConfig | None = None,
) -> None:
    """Run the end-to-end pipeline across all slides in a directory."""
    config = config or PipelineConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_cache_dirs(output_dir)

    slides = list_slides(input_dir, suffixes)
    if not slides:
        raise FileNotFoundError(f"No slides found under {input_dir} for suffixes: {suffixes}")

    manifest = []
    comp_paths: Dict[str, Path] = {}
    means_paths: Dict[str, Path] = {}

    for slide_path in slides:
        info = process_slide(slide_path, output_dir, config)
        manifest.append(info)

        slide_id = info["slide"]
        comp_paths[slide_id] = Path(info["comp_path"])
        means_paths[slide_id] = Path(info["means_path"])

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    comp_mat = build_composition_matrix_from_paths(comp_paths)
    means_mat = build_means_matrix_from_paths(means_paths)
    comp_mat.to_csv(output_dir / "matrix_celltype_distribution.csv")
    means_mat.to_csv(output_dir / "matrix_celltype_feature_means.csv")

    combined = pd.concat([comp_mat, means_mat], axis=1).fillna(0.0)
    combined.to_csv(output_dir / "matrix_combined.csv")

    embeddings_dir = output_dir / "embeddings"
    run_embeddings_and_clustering(comp_mat.values, "composition", embeddings_dir, config.clustering)
    run_embeddings_and_clustering(means_mat.values, "nuclei_means", embeddings_dir, config.clustering)
    run_embeddings_and_clustering(combined.values, "combined", embeddings_dir, config.clustering)
