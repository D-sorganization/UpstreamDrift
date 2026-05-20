import pytest


def test_headless_plotting_import() -> None:
    """Test that the plotting package can be imported without PyQt6.

    Validates that the core plotting infrastructure is importable in
    headless environments where no display server is available.
    """
    import importlib.util

    spec = importlib.util.find_spec("src.shared.python.plotting")
    assert (
        spec is not None
    ), "src.shared.python.plotting should be importable in headless mode"
