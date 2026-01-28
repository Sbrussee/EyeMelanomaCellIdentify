import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("shapely")
pytest.importorskip("skimage")
pytest.importorskip("sklearn")

from eyemelanoma.config import FeatureConfig
from eyemelanoma.features import (
    _resolve_base_mpp,
    add_spatial_features,
    FeatureSummary,
    hsv_melanin_fraction,
    polygon_to_mask,
    summarize_feature_csv,
    stream_nuclei_features_in_chunks,
    texture_features,
)


def test_polygon_to_mask() -> None:
    np = pytest.importorskip("numpy")
    Polygon = pytest.importorskip("shapely.geometry").Polygon
    poly = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
    mask = polygon_to_mask(poly, 6, 6)
    assert mask.sum() > 0


def test_hsv_melanin_fraction_detects_dark() -> None:
    np = pytest.importorskip("numpy")
    patch = np.zeros((4, 4, 3), dtype=np.uint8)
    patch[:] = [20, 20, 20]
    mask = np.ones((4, 4), dtype=bool)
    frac = hsv_melanin_fraction(patch, mask)
    assert 0.9 <= frac <= 1.0


def test_texture_features_returns_keys() -> None:
    np = pytest.importorskip("numpy")
    patch = np.ones((8, 8), dtype=np.uint8) * 80
    mask = np.ones((8, 8), dtype=bool)
    features = texture_features(patch, mask, FeatureConfig())
    assert "contrast" in features
    assert "homogeneity" in features


def test_add_spatial_features() -> None:
    pd = pytest.importorskip("pandas")
    Polygon = pytest.importorskip("shapely.geometry").Polygon
    df = pd.DataFrame(
        {
            "centroid_x": [0.0, 1.0, 2.0],
            "centroid_y": [0.0, 1.0, 2.0],
            "cell_type": ["A", "A", "B"],
        }
    )
    roi = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    out = add_spatial_features(df.copy(), roi, k_neighbors=1)
    assert "local_density" in out.columns
    assert "same_type_neighbor_fraction" in out.columns


def test_resolve_base_mpp_handles_properties_without_get() -> None:
    class DummyProperties:
        def __init__(self, value: float) -> None:
            self._value = value

        def __getitem__(self, key: str) -> float:
            if key == "openslide.mpp-x":
                return self._value
            raise KeyError(key)

    class DummyReader:
        def __init__(self, value: float) -> None:
            self.properties = DummyProperties(value)

    class DummyWSI:
        def __init__(self, value: float) -> None:
            self.reader = DummyReader(value)

    wsi = DummyWSI(0.5)
    assert _resolve_base_mpp(wsi) == pytest.approx(0.5)


def test_feature_summary_distribution_and_means() -> None:
    summary = FeatureSummary()
    summary.update({"cell_id": 1, "cell_type": "A", "area_um2": 4.0, "mean_R": 10.0})
    summary.update({"cell_id": 2, "cell_type": "A", "area_um2": 6.0, "mean_R": 14.0})
    summary.update({"cell_id": 3, "cell_type": "B", "area_um2": 9.0, "mean_R": 20.0})

    dist = summary.to_distribution()
    assert dist["A"] == pytest.approx(2 / 3)
    assert dist["B"] == pytest.approx(1 / 3)

    means = summary.to_means_frame()
    means = means.set_index("cell_type")
    assert means.loc["A", "area_um2"] == pytest.approx(5.0)
    assert means.loc["B", "mean_R"] == pytest.approx(20.0)


def test_summarize_feature_csv(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    path = tmp_path / "features.csv"
    df = pd.DataFrame(
        [
            {"cell_type": "A", "area_um2": 4.0, "mean_R": 10.0},
            {"cell_type": "A", "area_um2": 6.0, "mean_R": 14.0},
            {"cell_type": "B", "area_um2": 9.0, "mean_R": 20.0},
        ]
    )
    df.to_csv(path, index=False)

    summary = summarize_feature_csv(path, chunk_size=2)
    dist = summary.to_distribution()
    means = summary.to_means_frame().set_index("cell_type")

    assert dist["A"] == pytest.approx(2 / 3)
    assert means.loc["A", "area_um2"] == pytest.approx(5.0)


def test_stream_nuclei_features_in_chunks(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    Polygon = pytest.importorskip("shapely.geometry").Polygon

    class DummyWSI:
        mpp = 0.5

    def fake_read_cell_patch_rgba(wsi, bbox_xyxy, level, color_mpp):
        patch = np.ones((4, 4, 3), dtype=np.uint8) * 10
        return patch, 0.5

    monkeypatch.setattr("eyemelanoma.features.read_cell_patch_rgba", fake_read_cell_patch_rgba)

    poly = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
    chunk_a = pd.DataFrame({"geometry": [poly], "cell_type": ["A"]})
    chunk_b = pd.DataFrame({"geometry": [poly], "cell_type": ["B"]})

    output_path = tmp_path / "features.csv"
    summary = stream_nuclei_features_in_chunks(
        DummyWSI(),
        [chunk_a, chunk_b],
        FeatureConfig(),
        output_path,
        show_progress=False,
    )

    assert summary.total_cells == 2
    assert output_path.exists()
