"""Utilities for generating slide-level vectors from per-cell features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def celltype_distribution(df_cells: pd.DataFrame) -> pd.Series:
    """Compute normalized cell-type distribution."""
    if "cell_type" not in df_cells.columns or df_cells.empty:
        return pd.Series(dtype=float)
    counts = df_cells["cell_type"].value_counts(dropna=True)
    return counts / counts.sum() if counts.sum() > 0 else counts.astype(float)


def celltype_feature_means(df_cells: pd.DataFrame) -> pd.DataFrame:
    """Compute per-celltype mean of numeric features."""
    if df_cells.empty or "cell_type" not in df_cells.columns:
        return pd.DataFrame()
    feature_cols = [
        c
        for c in df_cells.columns
        if c not in {"cell_id", "cell_type"} and pd.api.types.is_numeric_dtype(df_cells[c])
    ]
    return df_cells.groupby("cell_type")[feature_cols].mean(numeric_only=True).reset_index()


def build_composition_matrix(per_slide: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build a slide-by-celltype composition matrix."""
    all_types = sorted({t for s in per_slide.values() for t in s.index})
    rows = []
    for slide, series in per_slide.items():
        row = {t: 0.0 for t in all_types}
        row.update(series.to_dict())
        row["slide"] = slide
        rows.append(row)
    return pd.DataFrame(rows).set_index("slide").sort_index()


def build_means_matrix(per_slide: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a slide-by-(celltype, feature) matrix."""
    all_types = sorted({t for df in per_slide.values() for t in df["cell_type"].astype(str)})
    all_features = sorted(
        {c for df in per_slide.values() for c in df.columns if c != "cell_type" and pd.api.types.is_numeric_dtype(df[c])}
    )
    rows = []
    for slide, df in per_slide.items():
        row = {f"{t}__{f}": 0.0 for t in all_types for f in all_features}
        for _, r in df.iterrows():
            t = str(r["cell_type"])
            for f in all_features:
                if f in r and pd.notna(r[f]):
                    row[f"{t}__{f}"] = float(r[f])
        row["slide"] = slide
        rows.append(row)
    return pd.DataFrame(rows).set_index("slide").sort_index()


def build_composition_matrix_from_paths(slide_paths: Dict[str, Path]) -> pd.DataFrame:
    """
    Build a slide-by-celltype composition matrix from CSV paths.

    This helper avoids keeping all per-slide series in memory at once by
    scanning the CSVs twice: first to collect cell types, then to populate rows.
    """
    all_types: set[str] = set()
    for path in slide_paths.values():
        series = pd.read_csv(path, index_col=0)["fraction"]
        all_types.update(series.index.astype(str))

    ordered_types = sorted(all_types)
    rows = []
    for slide, path in slide_paths.items():
        series = pd.read_csv(path, index_col=0)["fraction"]
        row = {t: 0.0 for t in ordered_types}
        row.update(series.to_dict())
        row["slide"] = slide
        rows.append(row)
    return pd.DataFrame(rows).set_index("slide").sort_index()


def build_means_matrix_from_paths(slide_paths: Dict[str, Path]) -> pd.DataFrame:
    """
    Build a slide-by-(celltype, feature) matrix from CSV paths.

    This helper reads each CSV on demand to keep peak memory usage low while
    still producing a dense slide-level feature matrix.
    """
    all_types: set[str] = set()
    all_features: set[str] = set()
    for path in slide_paths.values():
        df = pd.read_csv(path)
        if df.empty:
            continue
        all_types.update(df["cell_type"].astype(str))
        for col in df.columns:
            if col == "cell_type":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                all_features.add(col)

    ordered_types = sorted(all_types)
    ordered_features = sorted(all_features)
    rows = []
    for slide, path in slide_paths.items():
        df = pd.read_csv(path)
        row = {f"{t}__{f}": 0.0 for t in ordered_types for f in ordered_features}
        if not df.empty:
            for _, record in df.iterrows():
                cell_type = str(record["cell_type"])
                for feature in ordered_features:
                    if feature in record and pd.notna(record[feature]):
                        row[f"{cell_type}__{feature}"] = float(record[feature])
        row["slide"] = slide
        rows.append(row)
    return pd.DataFrame(rows).set_index("slide").sort_index()
