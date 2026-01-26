from eyemelanoma.profiling import profile_resource_usage


def test_profile_resource_usage_reports_values() -> None:
    profile = profile_resource_usage(sample_size_mb=1)
    assert profile.max_rss_mb >= 0.0
    assert profile.io_write_mb_s > 0.0
    assert profile.io_read_mb_s > 0.0
