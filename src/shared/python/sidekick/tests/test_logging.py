def test_logging_imports() -> None:
    from upstream_drift_tools.utils.logging import (
        DEFAULT_FORMAT,
        SIMPLE_FORMAT,
        get_logger,
    )

    assert DEFAULT_FORMAT is not None
    assert SIMPLE_FORMAT is not None
    assert get_logger is not None
    assert callable(get_logger)
