from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

import eyemelanoma.pipeline as pipeline
from eyemelanoma.config import PipelineConfig


def test_run_pipeline_integration(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "slides"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    slide_paths = [input_dir / "slide_a.mrxs", input_dir / "slide_b.mrxs"]
    for path in slide_paths:
        path.write_text("dummy")

    def fake_process_slide(slide_path: Path, out_dir: Path, config: PipelineConfig) -> dict:
        slide_out = out_dir / slide_path.stem
        slide_out.mkdir(parents=True, exist_ok=True)
        features_path = slide_out / "cells_features.csv"
        features_path.write_text("cell_type,area_um2\n")

        comp_path = slide_out / "celltype_distribution.csv"
        pd.Series({"Tumor": 1.0}, name="fraction").to_csv(comp_path, header=True)

        means_path = slide_out / "celltype_feature_means.csv"
        pd.DataFrame([{"cell_type": "Tumor", "area_um2": 10.0}]).to_csv(means_path, index=False)

        return {
            "slide": slide_path.stem,
            "features_path": str(features_path),
            "comp_path": str(comp_path),
            "means_path": str(means_path),
        }

    calls: list[tuple[str, str]] = []

    def fake_embeddings(values, label, emb_dir, config) -> None:
        calls.append((label, str(emb_dir)))

    monkeypatch.setattr(pipeline, "process_slide", fake_process_slide)
    monkeypatch.setattr(pipeline, "run_embeddings_and_clustering", fake_embeddings)

    pipeline.run_pipeline(input_dir, output_dir, suffixes=(".mrxs",))

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "matrix_celltype_distribution.csv").exists()
    assert (output_dir / "matrix_celltype_feature_means.csv").exists()
    assert (output_dir / "matrix_combined.csv").exists()
    assert len(calls) == 3
