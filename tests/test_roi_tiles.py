import pytest


class DummyWSI:
    TILE_SPEC_KEY = "tile_spec"

    def __init__(self, pd_module) -> None:
        self.shapes = {"tiles": pd_module.DataFrame({"x": [0, 1], "y": [0, 1]})}
        self.attrs = {"tile_spec": {"tiles": {"tile_px": 256}}}


def test_roi_to_geodataframe_wraps_polygon() -> None:
    pytest.importorskip("geopandas")
    pytest.importorskip("pandas")
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    from eyemelanoma.pipeline import _roi_to_geodataframe

    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    gdf = _roi_to_geodataframe(polygon)

    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].equals(polygon)
    assert gdf.loc[0, "roi_id"] == 0


def test_subset_segmentation_tiles_creates_new_key() -> None:
    pytest.importorskip("geopandas")
    pd = pytest.importorskip("pandas")

    from eyemelanoma.pipeline import _subset_segmentation_tiles

    wsi = DummyWSI(pd)

    new_key = _subset_segmentation_tiles(wsi, "tiles", [1], new_key="cell_segmentation_tiles")

    assert new_key == "cell_segmentation_tiles"
    assert new_key in wsi.shapes
    assert len(wsi.shapes[new_key]) == 1
    assert wsi.attrs["tile_spec"][new_key] == wsi.attrs["tile_spec"]["tiles"]
