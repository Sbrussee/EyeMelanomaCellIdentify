"""End-to-end MRXS pipeline using LazySlide and HistoPLUS."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
    summarize_feature_csv,
)
from eyemelanoma.roi import load_roi_from_mrxs_dir
from eyemelanoma.slide_vectors import (
    build_composition_matrix_from_paths,
    build_means_matrix_from_paths,
    celltype_distribution,
    celltype_feature_means,
)
from eyemelanoma.profiling import ResourceLogger
from eyemelanoma.progress import progress_iter


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


def _maybe_log_resource_usage(stage: str, resource_logger: ResourceLogger | None) -> None:
    """Optionally log resource usage to help diagnose memory hotspots."""
    if resource_logger is None:
        return
    resource_logger.log(stage)


def _should_use_cached_features(
    features_path: Path,
    meta_path: Path,
    roi_present: bool,
    config: PipelineConfig,
) -> bool:
    """Decide whether cached features can be reused for the current run."""
    if not config.features.use_cached_features:
        return False
    if not features_path.exists() or features_path.stat().st_size == 0:
        return False
    if meta_path.exists():
        try:
            with open(meta_path) as handle:
                meta = json.load(handle)
            if "roi_applied" in meta and bool(meta["roi_applied"]) != roi_present:
                return False
        except Exception:
            return False
    elif roi_present:
        return False
    return True


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


def _iter_tile_indices(total_tiles: int, chunk_size: int) -> Iterable[List[int]]:
    """
    Yield tile indices in chunks to limit peak memory usage.

    Parameters
    ----------
    total_tiles
        Total number of tiles available for segmentation.
    chunk_size
        Maximum number of tiles per chunk.
    """
    assert total_tiles >= 0, "total_tiles must be non-negative."
    assert chunk_size > 0, "chunk_size must be positive."
    for start in range(0, total_tiles, chunk_size):
        end = min(start + chunk_size, total_tiles)
        yield list(range(start, end))


def _collect_cell_type_chunks(
    cell_chunks: Sequence[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """
    Combine cell-type GeoDataFrames into a single GeoDataFrame.

    Parameters
    ----------
    cell_chunks
        Sequence of per-chunk GeoDataFrames with geometry and cell_type columns.
    """
    if not cell_chunks:
        return gpd.GeoDataFrame({"cell_type": []}, geometry=[], crs=None)
    combined = pd.concat(cell_chunks, ignore_index=True)
    crs = getattr(cell_chunks[0], "crs", None)
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
    assert "geometry" in combined_gdf.columns, "Combined GeoDataFrame missing geometry column."
    return combined_gdf


def _extract_cells_with_histoplus(
    wsi,
    config: PipelineConfig,
    *,
    roi_polygon=None,
    resource_logger: ResourceLogger | None = None,
) -> None:
    if roi_polygon is not None:
        _drop_shape_keys(wsi, ["tissues"])
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
        zs.pp.find_tissues(wsi)
        _maybe_log_resource_usage("post_find_tissues", resource_logger)
        zs.pp.tile_tissues(
            wsi,
            config.tile.tile_px,
            overlap=config.tile.overlap,
            background_fraction=config.tile.background_fraction,
            mpp=config.tile.mpp_tiles,
        )
        tile_key = "tiles"
    _maybe_log_resource_usage("post_tiling", resource_logger)

    tiles = wsi.shapes.get(tile_key)
    if tiles is None:
        raise ValueError(f"Tile key {tile_key} not found after tiling.")
    total_tiles = int(len(tiles))
    chunk_size = config.segmentation.tile_chunk_size
    if chunk_size is not None and chunk_size > 0 and total_tiles > chunk_size:
        cell_chunks: List[gpd.GeoDataFrame] = []
        chunk_indices = list(_iter_tile_indices(total_tiles, chunk_size))
        for chunk_id, indices in progress_iter(
            list(enumerate(chunk_indices)),
            total=len(chunk_indices),
            desc="Cell types (chunks)",
        ):
            chunk_key = _subset_segmentation_tiles(
                wsi,
                tile_key,
                indices,
                new_key=f"cell_segmentation_tiles_{chunk_id}",
            )
            zs.seg.cell_types(
                wsi,
                model=config.segmentation.model,
                tile_key=chunk_key,
                batch_size=_resolve_histoplus_batch(config),
                num_workers=0,
                amp=True,
                size_filter=False,
                nucleus_size=(20, 1000),
                pbar=True,
                key_added="cell_types_chunk",
            )
            cell_chunk = _shrink_cell_gdf(wsi.shapes["cell_types_chunk"])
            cell_chunks.append(cell_chunk)
            _drop_shape_keys(wsi, [chunk_key, "cell_types_chunk"])
            gc.collect()
        wsi.shapes["cell_types"] = _collect_cell_type_chunks(cell_chunks)
    else:
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
    _maybe_log_resource_usage("post_cell_types", resource_logger)
    _drop_shape_keys(wsi, ["tiles", "cell_segmentation_tiles"])


def process_slide(
    slide_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    *,
    resource_logger: ResourceLogger | None = None,
) -> Dict:
    slide_out = output_dir / slide_path.stem
    slide_out.mkdir(parents=True, exist_ok=True)
    features_path = slide_out / "cells_features.csv"
    comp_path = slide_out / "celltype_distribution.csv"
    means_path = slide_out / "celltype_feature_means.csv"
    meta_path = slide_out / "meta.json"

    data_dir = _slide_data_dir(slide_path)
    roi_polygon = None
    if data_dir:
        try:
            roi_polygon = load_roi_from_mrxs_dir(data_dir)
        except Exception as exc:
            print(f"[WARN] Failed to parse ROI from {data_dir}: {exc}")

    _maybe_log_resource_usage(f"slide_start:{slide_path.stem}", resource_logger)
    if _should_use_cached_features(features_path, meta_path, roi_polygon is not None, config):
        summary = summarize_feature_csv(features_path, chunk_size=config.features.cache_chunk_size)
        n_cells = summary.total_cells
        comp = summary.to_distribution()
        means = summary.to_means_frame()
        _maybe_log_resource_usage(f"cached_features:{slide_path.stem}", resource_logger)
    else:
        wsi = open_wsi(str(slide_path))
        try:
            if "cell_types" not in wsi.shapes:
                _extract_cells_with_histoplus(
                    wsi,
                    config,
                    roi_polygon=roi_polygon,
                    resource_logger=resource_logger,
                )
                try:
                    wsi.write()
                except Exception:
                    pass

            cell_gdf = _shrink_cell_gdf(wsi.shapes["cell_types"])
            _drop_shape_keys(wsi, ["cell_types", "rois"])
            _maybe_log_resource_usage("post_cell_types_copy", resource_logger)
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

            n_cells = int(len(cell_gdf))
            enable_spatial = config.features.enable_spatial_features
            use_spatial = enable_spatial and n_cells <= config.features.max_cells_for_spatial
            if enable_spatial and not use_spatial:
                print(
                    "[WARN] Skipping spatial features to reduce memory usage. "
                    f"Cell count {n_cells} exceeds max_cells_for_spatial={config.features.max_cells_for_spatial}."
                )

            if use_spatial:
                df_cells = extract_nuclei_features(
                    wsi,
                    cell_gdf,
                    config.features,
                    show_progress=True,
                    progress_desc=f"Features {slide_path.stem}",
                    resource_logger=resource_logger,
                )
                if roi_polygon is not None:
                    df_cells = add_spatial_features(df_cells, roi_polygon)
                df_cells.to_csv(features_path, index=False)
                n_cells = int(len(df_cells))
                comp = celltype_distribution(df_cells)
                means = celltype_feature_means(df_cells)
                del df_cells
            else:
                summary = stream_nuclei_features(
                    wsi,
                    cell_gdf,
                    config.features,
                    features_path,
                    show_progress=True,
                    progress_desc=f"Features {slide_path.stem}",
                    resource_logger=resource_logger,
                )
                n_cells = summary.total_cells
                comp = summary.to_distribution()
                means = summary.to_means_frame()
            _maybe_log_resource_usage(f"post_features:{slide_path.stem}", resource_logger)
        finally:
            close_fn = getattr(wsi, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            del wsi
            gc.collect()
    comp.to_csv(comp_path, header=["fraction"])
    means.to_csv(means_path, index=False)
    gc.collect()
    _maybe_log_resource_usage(f"post_slide:{slide_path.stem}", resource_logger)

    meta = {"slide": slide_path.name, "n_cells": n_cells, "roi_applied": bool(roi_polygon is not None)}
    with open(meta_path, "w") as handle:
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

    resource_logger = ResourceLogger.from_env()
    _maybe_log_resource_usage("pipeline_start", resource_logger)

    manifest = []
    comp_paths: Dict[str, Path] = {}
    means_paths: Dict[str, Path] = {}

    for slide_path in progress_iter(slides, total=len(slides), desc="Slides"):
        info = process_slide(slide_path, output_dir, config, resource_logger=resource_logger)
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
    embedding_tasks = [
        (comp_mat.values, "composition"),
        (means_mat.values, "nuclei_means"),
        (combined.values, "combined"),
    ]
    for values, label in progress_iter(embedding_tasks, total=len(embedding_tasks), desc="Embeddings"):
        run_embeddings_and_clustering(values, label, embeddings_dir, config.clustering)
    del comp_mat, means_mat, combined
    gc.collect()
    _maybe_log_resource_usage("pipeline_end", resource_logger)
