"""Smoke tests for top-level shared module imports.

Addresses issue #5478: ensures no top-level ImportError on shared modules.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path",
    [
        "src.shared.python.config",
        "src.shared.python.data_io",
        "src.api",
    ],
)
def test_all_public_modules_import_cleanly(module_path: str) -> None:
    """Ensure no top-level ImportError on any shared module."""
    importlib.import_module(module_path)  # must not raise
