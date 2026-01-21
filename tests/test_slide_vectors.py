import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from eyemelanoma.slide_vectors import (
    build_composition_matrix,
    build_means_matrix,
    celltype_distribution,
    celltype_feature_means,
)


def test_celltype_distribution() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"cell_type": ["A", "A", "B"]})
    dist = celltype_distribution(df)
    assert dist["A"] == 2 / 3
    assert dist["B"] == 1 / 3


def test_celltype_feature_means() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"cell_type": ["A", "A", "B"], "area_um2": [1.0, 3.0, 5.0]})
    means = celltype_feature_means(df)
    assert set(means["cell_type"]) == {"A", "B"}
    assert means.loc[means["cell_type"] == "A", "area_um2"].iloc[0] == 2.0


def test_build_matrices() -> None:
    pd = pytest.importorskip("pandas")
    dist = {
        "slide1": pd.Series({"A": 0.5, "B": 0.5}),
        "slide2": pd.Series({"A": 1.0}),
    }
    comp = build_composition_matrix(dist)
    assert "A" in comp.columns
    assert "B" in comp.columns

    means = {
        "slide1": pd.DataFrame({"cell_type": ["A"], "area_um2": [1.0]}),
        "slide2": pd.DataFrame({"cell_type": ["B"], "area_um2": [2.0]}),
    }
    means_mat = build_means_matrix(means)
    assert "A__area_um2" in means_mat.columns
    assert "B__area_um2" in means_mat.columns
