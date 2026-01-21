import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("shapely")
pytest.importorskip("skimage")
pytest.importorskip("sklearn")

from eyemelanoma.config import FeatureConfig
from eyemelanoma.features import add_spatial_features, hsv_melanin_fraction, polygon_to_mask, texture_features


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
