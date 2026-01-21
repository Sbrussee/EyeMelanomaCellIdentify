"""Configuration dataclasses for the MRXS processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class TileConfig:
    """Configuration for tiling and segmentation resolution."""

    tile_px: int = 512
    overlap: float = 0.20
    mpp_tiles: float = 0.5
    background_fraction: float = 0.95


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for segmentation and classification models."""

    seg_batch: int = 16
    cls_batch: int = 8
    model: str = "histoplus"


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature extraction."""

    patch_expansion_px: int = 6
    reader_level_for_color: int | None = None
    color_mpp: float = 0.5
    glcm_distances: Tuple[int, ...] = (1, 2, 4)
    glcm_angles: Tuple[float, ...] = (0.0, 0.78539816339, 1.57079632679, 2.35619449019)


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for slide-level clustering and embeddings."""

    k_min: int = 4
    k_max: int = 8
    random_state: int = 17


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the pipeline."""

    tile: TileConfig = TileConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    features: FeatureConfig = FeatureConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    output_dir: Path = Path("outputs")
