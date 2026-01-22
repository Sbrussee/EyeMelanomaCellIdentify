from pathlib import Path

import pandas as pd

from eyemelanoma.slide_vectors import (
    build_composition_matrix_from_paths,
    build_means_matrix_from_paths,
)


def _write_comp(path: Path, data: dict[str, float]) -> None:
    series = pd.Series(data, name="fraction")
    series.to_csv(path, header=["fraction"])


def _write_means(path: Path, rows: list[dict[str, float | str]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_build_composition_matrix_from_paths(tmp_path: Path) -> None:
    slide_a = tmp_path / "slide_a.csv"
    slide_b = tmp_path / "slide_b.csv"
    _write_comp(slide_a, {"tumor": 0.7, "stromal": 0.3})
    _write_comp(slide_b, {"immune": 1.0})

    matrix = build_composition_matrix_from_paths({"a": slide_a, "b": slide_b})

    assert list(matrix.index) == ["a", "b"]
    assert matrix.loc["a", "tumor"] == 0.7
    assert matrix.loc["a", "stromal"] == 0.3
    assert matrix.loc["a", "immune"] == 0.0
    assert matrix.loc["b", "immune"] == 1.0


def test_build_means_matrix_from_paths(tmp_path: Path) -> None:
    slide_a = tmp_path / "slide_a.csv"
    slide_b = tmp_path / "slide_b.csv"
    _write_means(
        slide_a,
        [
            {"cell_type": "tumor", "area": 10.0, "perimeter": 3.0, "note": "ignore"},
            {"cell_type": "immune", "area": 5.0, "perimeter": 2.0, "note": "ignore"},
        ],
    )
    _write_means(slide_b, [{"cell_type": "tumor", "area": 8.0, "perimeter": 4.0}])

    matrix = build_means_matrix_from_paths({"a": slide_a, "b": slide_b})

    assert list(matrix.index) == ["a", "b"]
    assert matrix.loc["a", "tumor__area"] == 10.0
    assert matrix.loc["a", "immune__perimeter"] == 2.0
    assert matrix.loc["b", "tumor__area"] == 8.0
    assert matrix.loc["b", "immune__area"] == 0.0
