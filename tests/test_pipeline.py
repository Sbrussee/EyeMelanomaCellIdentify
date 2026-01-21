from eyemelanoma.config import PipelineConfig
from eyemelanoma.pipeline import _resolve_histoplus_batch


def test_resolve_histoplus_batch_default() -> None:
    config = PipelineConfig()
    assert _resolve_histoplus_batch(config) == config.segmentation.cls_batch


def test_resolve_histoplus_batch_env_override(monkeypatch) -> None:
    config = PipelineConfig()
    monkeypatch.setenv("EYEMELANOMA_HISTOPLUS_CLS_BATCH", "2")
    assert _resolve_histoplus_batch(config) == 2


def test_resolve_histoplus_batch_env_invalid(monkeypatch) -> None:
    config = PipelineConfig()
    monkeypatch.setenv("EYEMELANOMA_HISTOPLUS_CLS_BATCH", "-1")
    assert _resolve_histoplus_batch(config) == config.segmentation.cls_batch
