from eyemelanoma.config import PipelineConfig
import pytest

gpd = pytest.importorskip("geopandas")
Polygon = pytest.importorskip("shapely.geometry").Polygon

from eyemelanoma.pipeline import _resolve_histoplus_batch, _shrink_cell_gdf


def test_resolve_histoplus_batch_default() -> None:
    config = PipelineConfig()
    assert _resolve_histoplus_batch(config) == min(
        config.segmentation.seg_batch,
        config.segmentation.cls_batch,
    )


def test_resolve_histoplus_batch_env_override(monkeypatch) -> None:
    config = PipelineConfig()
    monkeypatch.setenv("EYEMELANOMA_HISTOPLUS_CLS_BATCH", "2")
    assert _resolve_histoplus_batch(config) == 2


def test_resolve_histoplus_batch_env_invalid(monkeypatch) -> None:
    config = PipelineConfig()
    monkeypatch.setenv("EYEMELANOMA_HISTOPLUS_CLS_BATCH", "-1")
    assert _resolve_histoplus_batch(config) == min(
        config.segmentation.seg_batch,
        config.segmentation.cls_batch,
    )


def test_resolve_histoplus_batch_prefers_smallest_env(monkeypatch) -> None:
    config = PipelineConfig()
    monkeypatch.setenv("EYEMELANOMA_HISTOPLUS_SEG_BATCH", "4")
    monkeypatch.setenv("EYEMELANOMA_HISTOPLUS_CLS_BATCH", "3")
    assert _resolve_histoplus_batch(config) == 3


def test_shrink_cell_gdf_keeps_geometry_and_cell_type() -> None:
    poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({"class": ["Melanoma"], "extra": [1]}, geometry=[poly])

    trimmed = _shrink_cell_gdf(gdf)

    assert list(trimmed.columns) == ["geometry", "cell_type"]
    assert trimmed.loc[0, "cell_type"] == "Melanoma"
