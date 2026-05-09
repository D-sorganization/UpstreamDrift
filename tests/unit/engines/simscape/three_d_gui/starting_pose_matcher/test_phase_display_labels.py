"""Unit tests for the Simscape Starting-Pose Matcher.

Two layers:

1. **Pure-data tests** (`starting_pose_core`) — RigidTransform math,
   shaft-snap solver, xlsx loaders, skeleton loader, MocapEvents.  These
   have no Qt dependency and run in any environment with numpy + pandas.
2. **UI smoke tests** (`starting_pose_matcher`) — instantiate the Qt
   window with offscreen platform.  Skipped if PyQt6 fails to load.

The module files live under a directory with spaces (``Motion Capture
Plotter``), so we load them by absolute path with importlib.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Locate the matcher modules                                                  #
# --------------------------------------------------------------------------- #

_REPO = Path(__file__).resolve().parents[5]
# New canonical location (per #4376).
_PACKAGE_DIR = _REPO / "src" / "tools" / "starting_pose_matcher"
_CORE_PY = _PACKAGE_DIR / "core.py"
_MATCHER_PY = _PACKAGE_DIR / "gui.py"
# Wiffle xlsx is still in the legacy MATLAB tree (it's a subject-specific
# motion-capture asset, not part of the matcher's own package).
_WIFFLE_XLSX = (
    _REPO
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui"
    / "Motion Capture Plotter"
    / "Wiffle_ProV1_club_3D_data.xlsx"
)


def _load_module_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        pytest.skip(f"could not build module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_core():
    if not _CORE_PY.exists():
        pytest.skip(f"core module not found: {_CORE_PY}")
    # Repo root must be on sys.path so ``from src.shared...`` imports resolve.
    repo_root_str = str(_REPO)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    # Load via the package import so the FK shared-infra import resolves.
    import importlib

    return importlib.import_module("src.tools.starting_pose_matcher.core")


def _load_matcher():
    """Load the Qt UI module; skip if Qt cannot import in this env."""
    try:
        import PyQt6.QtCore  # noqa: F401
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable in this env: {exc}")
    if not _MATCHER_PY.exists():
        pytest.skip(f"matcher not found: {_MATCHER_PY}")
    repo_root_str = str(_REPO)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    try:
        import importlib

        return importlib.import_module("src.tools.starting_pose_matcher.gui")
    except (ImportError, OSError) as exc:
        pytest.skip(f"matcher module failed to load: {exc}")


@pytest.fixture(scope="module")
def core():
    return _load_core()


# --------------------------------------------------------------------------- #
# 1. Pure-data tests (no Qt)                                                  #
# --------------------------------------------------------------------------- #


class TestPhaseDisplayLabels:
    """Make sure phase labels in the UI are spelled out, not abbreviated."""

    @pytest.fixture
    def labels(self, core):
        return dict(core.EVENT_LABEL_PRESETS["Wiffle (A/T/I/F)"])

    def test_backswing_uses_full_words(self, core, labels):
        s = core.phase_display_label("backswing", labels)
        assert "Address" in s and "Top of Backswing" in s
        # No abbreviated arrows in the spelled-out form
        assert "(A → T)" not in s and "A→T" not in s

    def test_downswing_uses_full_words(self, core, labels):
        s = core.phase_display_label("downswing", labels)
        assert "Top of Backswing" in s and "Impact" in s
        assert "(T → I)" not in s

    def test_follow_through_uses_full_words(self, core, labels):
        s = core.phase_display_label("follow_through", labels)
        assert "Impact" in s and "Finish" in s
        assert "(I → F)" not in s

    def test_full_swing_uses_full_words(self, core, labels):
        s = core.phase_display_label("full_swing", labels)
        assert "Address" in s and "Finish" in s
        assert "(A → F)" not in s

    def test_custom_labels_propagate(self, core):
        custom = {"A": "MySetup", "T": "MyTop", "I": "MyStrike", "F": "MyEnd"}
        s = core.phase_display_label("downswing", custom)
        assert "MyTop" in s and "MyStrike" in s

    def test_legacy_label_lookup_returns_key(self, core):
        # Old session JSON might persist "Backswing (A → T)" — accept it.
        assert core.phase_key_from_label("Backswing (A → T)") == "backswing"
        assert core.phase_key_from_label("Full swing (A → F)") == "full_swing"

    def test_logical_key_passthrough(self, core):
        # Already a key — still resolves.
        assert core.phase_key_from_label("backswing") == "backswing"

    def test_unknown_label_returns_none(self, core):
        assert core.phase_key_from_label("Not A Real Phase") is None


# --------------------------------------------------------------------------- #
# 2. xlsx loaders (require the Wiffle fixture)                                #
# --------------------------------------------------------------------------- #


def _require_xlsx() -> None:
    if not _WIFFLE_XLSX.exists():
        pytest.skip(f"Wiffle xlsx fixture not available: {_WIFFLE_XLSX}")


# --------------------------------------------------------------------------- #
# 3. UI smoke tests                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def spm():
    """Loaded matcher module; skipped if Qt unavailable."""
    return _load_matcher()


@pytest.fixture(scope="module")
def qapp(spm):
    """QApplication using offscreen platform for headless tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def matcher(qapp, spm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    win = spm.StartingPoseMatcher()
    yield win
    win.close()
