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


class TestSkeletonTrajectory:
    def test_empty_trajectory_default(self, core):
        t = core.SkeletonTrajectory()
        assert len(t) == 0

    def test_frame_at_time_clamps_to_range(self, core):
        def sk(v):
            return core.Skeleton(joints={"mp": np.array([v, 0.0, 1.0])})

        t = core.SkeletonTrajectory(
            times=np.array([0.0, 0.1, 0.2, 0.3]),
            frames=[sk(0.0), sk(0.1), sk(0.2), sk(0.3)],
        )
        assert t.frame_at_time(-1.0) == 0
        assert t.frame_at_time(0.0) == 0
        assert t.frame_at_time(0.11) == 1
        assert t.frame_at_time(0.30) == 3
        assert t.frame_at_time(99.0) == 3

    def test_load_trajectory_csv_short_columns(self, core, tmp_path):
        # Build a minimal CSV with the short-form columns
        import pandas as pd

        df = pd.DataFrame(
            {
                "time": np.linspace(0.0, 0.1, 11),
                "club_head_X": np.linspace(0.0, 1.0, 11),
                "club_head_Y": np.zeros(11),
                "club_head_Z": np.linspace(0.5, 0.0, 11),
                "left_hand_X": np.zeros(11),
                "left_hand_Y": np.linspace(0.0, 0.2, 11),
                "left_hand_Z": np.full(11, 0.8),
                "right_hand_X": np.full(11, 0.05),
                "right_hand_Y": np.linspace(0.0, 0.2, 11),
                "right_hand_Z": np.full(11, 0.8),
            }
        )
        path = tmp_path / "traj.csv"
        df.to_csv(path, index=False)
        traj = core.load_simscape_trajectory_csv(path)
        assert len(traj) == 11
        # First frame's joints
        f0 = traj.frames[0]
        assert "ch" in f0.joints
        assert "lw" in f0.joints
        assert "rw" in f0.joints
        # mp is synthesized as the midpoint of lw and rw
        assert "mp" in f0.joints
        np.testing.assert_allclose(
            f0.joints["mp"], (f0.joints["lw"] + f0.joints["rw"]) / 2.0, atol=1e-9
        )
        # times preserved
        np.testing.assert_allclose(traj.times, df["time"].to_numpy())

    def test_load_trajectory_csv_long_columns(self, core, tmp_path):
        # The raw Simscape bus convention.
        import pandas as pd

        df = pd.DataFrame(
            {
                "time": np.linspace(0.0, 0.05, 6),
                "ClubLogs_CHGlobalPosition_1": np.zeros(6),
                "ClubLogs_CHGlobalPosition_2": np.linspace(0, 1, 6),
                "ClubLogs_CHGlobalPosition_3": np.zeros(6),
            }
        )
        path = tmp_path / "long.csv"
        df.to_csv(path, index=False)
        traj = core.load_simscape_trajectory_csv(path)
        assert len(traj) == 6
        np.testing.assert_allclose(traj.frames[0].joints["ch"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(traj.frames[-1].joints["ch"], [0.0, 1.0, 0.0])

    def test_load_trajectory_csv_missing_time_raises(self, core, tmp_path):
        import pandas as pd

        path = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="time"):
            core.load_simscape_trajectory_csv(path)

    def test_load_trajectory_csv_no_recognised_joints_raises(self, core, tmp_path):
        import pandas as pd

        path = tmp_path / "junk.csv"
        pd.DataFrame(
            {"time": [0, 0.01], "foo_X": [1, 2], "foo_Y": [0, 0], "foo_Z": [0, 0]}
        ).to_csv(path, index=False)
        with pytest.raises(ValueError, match="recognised joint"):
            core.load_simscape_trajectory_csv(path)

    def test_torso_synthesised_when_missing(self, core, tmp_path):
        """When the CSV has spine + hub but no torso column, the loader
        synthesises torso at 20% of the way from spine to hub (matching
        the UpperTorsoBase = 0.2 * UpperTorsoLength split in the .mdl)."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "time": [0.0, 0.01],
                "spine_X": [0.0, 0.0],
                "spine_Y": [-0.30, -0.30],
                "spine_Z": [1.20, 1.20],
                "hub_X": [0.0, 0.0],
                "hub_Y": [-0.30, -0.30],
                "hub_Z": [1.45, 1.45],
                "left_hand_X": [-0.05, -0.05],
                "left_hand_Y": [-0.10, -0.10],
                "left_hand_Z": [0.85, 0.85],
                "right_hand_X": [0.05, 0.05],
                "right_hand_Y": [-0.10, -0.10],
                "right_hand_Z": [0.85, 0.85],
            }
        )
        path = tmp_path / "no_torso.csv"
        df.to_csv(path, index=False)
        traj = core.load_simscape_trajectory_csv(path)
        f0 = traj.frames[0]
        assert "torso" in f0.joints, "loader should synthesise torso"
        # 20% from spine -> hub: (1.20 + 0.2 * (1.45 - 1.20)) = 1.25
        np.testing.assert_allclose(f0.joints["torso"], [0.0, -0.30, 1.25], atol=1e-9)


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
