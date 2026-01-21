"""End-to-end MRXS pipeline using LazySlide and HistoPLUS."""

from __future__ import annotations

import json
import os
import gc
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
from eyemelanoma.features import add_spatial_features, extract_nuclei_features, filter_cells_in_roi
from eyemelanoma.roi import load_roi_from_mrxs_dir
from eyemelanoma.slide_vectors import (
    build_composition_matrix,
    build_means_matrix,
    celltype_distribution,
    celltype_feature_means,
)


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


def _attach_centroids(df: pd.DataFrame, cell_gdf) -> pd.DataFrame:
    centroids = cell_gdf.geometry.centroid
    df["centroid_x"] = centroids.x.values
    df["centroid_y"] = centroids.y.values
    return df


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

        cell_gdf = wsi.shapes["cell_types"]
        try:
            slide_dims = getattr(wsi, "dimensions", None) or getattr(wsi.reader, "dimensions", None)
            if slide_dims:
                centroids = np.column_stack([cell_gdf.geometry.centroid.x, cell_gdf.geometry.centroid.y])
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

        df_cells = extract_nuclei_features(wsi, cell_gdf, config.features)
        df_cells = _attach_centroids(df_cells, cell_gdf)
        if roi_polygon is not None:
            df_cells = add_spatial_features(df_cells, roi_polygon)
    finally:
        close_fn = getattr(wsi, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        del wsi
        gc.collect()

    features_path = slide_out / "cells_features.csv"
    df_cells.to_csv(features_path, index=False)

    comp = celltype_distribution(df_cells)
    means = celltype_feature_means(df_cells)
    comp_path = slide_out / "celltype_distribution.csv"
    means_path = slide_out / "celltype_feature_means.csv"
    comp.to_csv(comp_path, header=["fraction"])
    means.to_csv(means_path, index=False)

    meta = {
        "slide": slide_path.name,
        "n_cells": int(len(df_cells)),
        "roi_applied": bool(roi_polygon is not None),
    }
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
    per_slide_comp: Dict[str, pd.Series] = {}
    per_slide_means: Dict[str, pd.DataFrame] = {}

    for slide_path in slides:
        info = process_slide(slide_path, output_dir, config)
        manifest.append(info)

        slide_id = info["slide"]
        comp = pd.read_csv(info["comp_path"], index_col=0)["fraction"]
        means = pd.read_csv(info["means_path"])
        per_slide_comp[slide_id] = comp
        per_slide_means[slide_id] = means

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    comp_mat = build_composition_matrix(per_slide_comp)
    means_mat = build_means_matrix(per_slide_means)
    comp_mat.to_csv(output_dir / "matrix_celltype_distribution.csv")
    means_mat.to_csv(output_dir / "matrix_celltype_feature_means.csv")

    combined = pd.concat([comp_mat, means_mat], axis=1).fillna(0.0)
    combined.to_csv(output_dir / "matrix_combined.csv")

    embeddings_dir = output_dir / "embeddings"
    run_embeddings_and_clustering(comp_mat.values, "composition", embeddings_dir, config.clustering)
    run_embeddings_and_clustering(means_mat.values, "nuclei_means", embeddings_dir, config.clustering)
    run_embeddings_and_clustering(combined.values, "combined", embeddings_dir, config.clustering)
