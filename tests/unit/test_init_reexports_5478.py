"""Smoke tests for __init__.py re-exports -- issue #5478.

Verifies that the config and data_io packages can be imported at module level
without raising ImportError (previously broken by missing symbols that
__init__.py still re-exported after prior refactors).
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_config_imports_cleanly() -> None:
    """src.shared.python.config must import without raising ImportError."""
    importlib.import_module("src.shared.python.config")


@pytest.mark.unit
def test_data_io_imports_cleanly() -> None:
    """src.shared.python.data_io must import without raising ImportError."""
    importlib.import_module("src.shared.python.data_io")


@pytest.mark.unit
def test_config_exposes_get_setting() -> None:
    """get_setting must be importable from src.shared.python.config (issue #5478).

    This function was missing from settings.py while config/__init__.py still
    listed it as a re-export, causing ImportError at collection time.
    """
    import src.shared.python.config as cfg

    assert hasattr(
        cfg, "get_setting"
    ), "src.shared.python.config is missing get_setting (issue #5478)"
    assert callable(cfg.get_setting), "get_setting must be callable"


@pytest.mark.unit
def test_config_exposes_load_save_settings() -> None:
    """load_settings and save_settings must be importable (issue #5478)."""
    import src.shared.python.config as cfg

    for name in ("load_settings", "save_settings"):
        assert hasattr(
            cfg, name
        ), f"src.shared.python.config is missing {name!r} (issue #5478)"
        assert callable(getattr(cfg, name)), f"{name} must be callable"


@pytest.mark.unit
def test_data_io_exposes_provenance_symbols() -> None:
    """Provenance symbols must be importable from src.shared.python.data_io (issue #5478).

    add_provenance_header_file existed; add_provenance_header was missing while
    __init__.py re-exported it, causing ImportError.
    """
    import src.shared.python.data_io as dio

    for name in ("ProvenanceInfo", "add_provenance_header_file"):
        assert hasattr(
            dio, name
        ), f"src.shared.python.data_io is missing {name!r} (issue #5478)"
