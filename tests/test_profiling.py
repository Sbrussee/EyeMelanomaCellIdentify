from eyemelanoma.profiling import ResourceLogger, profile_resource_usage


def test_profile_resource_usage_reports_values() -> None:
    profile = profile_resource_usage(sample_size_mb=1)
    assert profile.max_rss_mb >= 0.0
    assert profile.io_write_mb_s > 0.0
    assert profile.io_read_mb_s > 0.0


def test_resource_logger_from_env(monkeypatch) -> None:
    monkeypatch.delenv("EYEMELANOMA_PROFILE_RESOURCES", raising=False)
    logger = ResourceLogger.from_env()
    assert logger.enabled is False

    monkeypatch.setenv("EYEMELANOMA_PROFILE_RESOURCES", "1")
    monkeypatch.setenv("EYEMELANOMA_PROFILE_EVERY", "5")
    monkeypatch.setenv("EYEMELANOMA_PROFILE_IO_MB", "2")
    logger = ResourceLogger.from_env()
    assert logger.enabled is True
    assert logger.log_every == 5
    assert logger.sample_size_mb == 2

    monkeypatch.setenv("EYEMELANOMA_PROFILE_EVERY", "bad")
    monkeypatch.setenv("EYEMELANOMA_PROFILE_IO_MB", "bad")
    logger = ResourceLogger.from_env()
    assert logger.log_every == 2000
    assert logger.sample_size_mb == 1
