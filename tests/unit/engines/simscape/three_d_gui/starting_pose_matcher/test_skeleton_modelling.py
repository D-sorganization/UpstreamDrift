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


class TestSkeletonModelling:
    """Verify the fallback skeletons reflect the actual model geometry."""

    def test_torso_present_in_both_fallbacks(self, core, tmp_path):
        impact = core.load_skeleton(tmp_path / "x.json", "Impact")
        top = core.load_skeleton(tmp_path / "x.json", "TopofBackswing")
        assert "torso" in impact.joints, "Impact pose missing torso joint"
        assert "torso" in top.joints, "TopofBackswing pose missing torso joint"

    def test_torso_between_spine_and_hub(self, core, tmp_path):
        """Torso must lie on the line between spine and hub (along the
        body's central column)."""
        for pose in ("Impact", "TopofBackswing"):
            s = core.load_skeleton(tmp_path / "x.json", pose)
            spine = s.joints["spine"]
            torso = s.joints["torso"]
            hub = s.joints["hub"]
            # Z-coordinate strictly between spine and hub
            assert spine[2] < torso[2] < hub[2], (
                f"{pose}: torso Z {torso[2]} not between spine {spine[2]} "
                f"and hub {hub[2]}"
            )

    def test_segments_include_full_torso_chain(self, core, tmp_path):
        s = core.load_skeleton(tmp_path / "x.json", "Impact")
        seg = s.segments
        assert ("hip", "spine") in seg
        assert ("spine", "torso") in seg, "missing spine→torso segment"
        assert ("torso", "hub") in seg, "missing torso→hub segment"
        # The old hip→hub-direct shortcut MUST be gone.
        assert ("spine", "hub") not in seg
        assert ("hip", "hub") not in seg

    def test_top_of_backswing_shows_torso_twist(self, core, tmp_path):
        """The two fallback poses must have visibly different shoulder lines
        — that's how the torso revolute (twist) is shown to the user.
        At Impact: shoulders aligned with X axis (target line).
        At Top of Backswing: shoulders rotated ~90° about body Z so the
        line lies more along Y (ball direction) than X.
        """
        impact = core.load_skeleton(tmp_path / "x.json", "Impact")
        top = core.load_skeleton(tmp_path / "x.json", "TopofBackswing")

        sl_imp = impact.joints["rs"] - impact.joints["ls"]
        sl_top = top.joints["rs"] - top.joints["ls"]
        # Project to XY (the twist plane).  Compute angle of each.
        ang_imp = float(np.arctan2(sl_imp[1], sl_imp[0]))
        ang_top = float(np.arctan2(sl_top[1], sl_top[0]))
        # The two shoulder-line angles should differ by at least 60°
        # (90° expected; allow generous tolerance for our hand-tuned values).
        diff_deg = abs(np.degrees(ang_top - ang_imp))
        diff_deg = min(diff_deg, 360.0 - diff_deg)
        assert diff_deg > 60.0, (
            f"shoulder line should rotate visibly between poses; "
            f"got only {diff_deg:.1f}° change"
        )


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
