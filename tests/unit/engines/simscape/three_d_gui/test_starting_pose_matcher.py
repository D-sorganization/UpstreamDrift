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


class TestRigidTransform:
    def test_identity_leaves_points_unchanged(self, core):
        T = core.RigidTransform()
        pts = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 0.7]])
        np.testing.assert_allclose(T.apply(pts), pts, atol=1e-12)

    def test_translation_only(self, core):
        T = core.RigidTransform(tx=0.1, ty=-0.2, tz=0.3)
        pts = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(T.apply(pts), [[0.1, -0.2, 0.3]], atol=1e-12)

    def test_rz_90_about_origin(self, core):
        # +X axis rotates to +Y under Rz(90°)
        T = core.RigidTransform(rz=90.0)
        out = T.apply(np.array([[1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-9)

    def test_rz_about_pivot_keeps_pivot_fixed(self, core):
        pivot = (0.5, 0.5, 1.0)
        T = core.RigidTransform(rz=37.5, pivot=pivot)
        out = T.apply(np.array([list(pivot)]))
        np.testing.assert_allclose(out[0], list(pivot), atol=1e-12)

    def test_scale_isotropic(self, core):
        T = core.RigidTransform(scale=2.0, pivot=(0.0, 0.0, 0.0))
        pts = np.array([[1.0, -2.0, 3.0]])
        np.testing.assert_allclose(T.apply(pts), 2.0 * pts, atol=1e-12)

    def test_translation_after_rotation(self, core):
        # Rotate +X 90° about origin, then translate by (1, 0, 0).
        T = core.RigidTransform(tx=1.0, rz=90.0)
        out = T.apply(np.array([[1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(out, [[1.0, 1.0, 0.0]], atol=1e-9)

    def test_inverse_under_negation(self, core):
        # Translation+rotation around origin should be undone by the
        # inverse transform applied in reverse order.
        Tf = core.RigidTransform(tx=0.3, ty=-0.1, rz=37.5)
        pts = np.array([[1.0, 0.0, 0.0]])
        forward = Tf.apply(pts)
        # Reverse: subtract translation, then rotate by -rz around origin.
        reverse_translation = forward - np.array([0.3, -0.1, 0.0])
        Tinv = core.RigidTransform(rz=-37.5)
        recovered = Tinv.apply(reverse_translation)
        np.testing.assert_allclose(recovered, pts, atol=1e-9)


class TestShaftSnap:
    def test_aligned_shafts_return_zero(self, core):
        rz = core.solve_shaft_rz_deg(
            np.array([0.0, 0.0, 1.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert abs(rz) < 1e-6

    def test_perpendicular_shafts_return_90(self, core):
        rz = core.solve_shaft_rz_deg(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert abs(rz - 90.0) < 1e-6

    def test_apply_solved_rz_aligns_xy_shaft(self, core):
        mp_skel = np.array([0.0, 0.0, 1.0])
        ch_skel = np.array([1.0, 0.0, 0.5])
        mp_target = np.array([0.5, -0.2, 1.3])
        ch_target = np.array([0.5, 0.8, 0.7])

        rz = core.solve_shaft_rz_deg(mp_target, ch_target, mp_skel, ch_skel)
        T = core.RigidTransform(rz=rz, pivot=tuple(mp_skel))
        rotated_mp = T.apply(mp_skel[None, :])[0]
        T = core.RigidTransform(
            tx=mp_target[0] - rotated_mp[0],
            ty=mp_target[1] - rotated_mp[1],
            tz=mp_target[2] - rotated_mp[2],
            rz=rz,
            pivot=tuple(mp_skel),
        )
        np.testing.assert_allclose(T.apply(mp_skel[None, :])[0], mp_target, atol=1e-9)
        out_dir = (T.apply(ch_skel[None, :])[0] - mp_target)[:2]
        out_dir /= np.linalg.norm(out_dir)
        tgt_dir = (ch_target - mp_target)[:2]
        tgt_dir /= np.linalg.norm(tgt_dir)
        np.testing.assert_allclose(out_dir, tgt_dir, atol=1e-9)

    def test_degenerate_vertical_shaft_returns_zero(self, core):
        # When the XY projection has no length, Rz is undefined; the
        # solver returns 0.0 rather than raising.
        rz = core.solve_shaft_rz_deg(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),  # vertical -> XY projection = 0
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert rz == 0.0


class TestSkeleton:
    def test_load_skeleton_returns_fallback(self, core, tmp_path):
        s = core.load_skeleton(tmp_path / "missing.json", "Impact")
        assert isinstance(s, core.Skeleton)
        assert "mp" in s.joints and "ch" in s.joints
        assert len(s.segments) > 0
        assert all(s.joints[k].shape == (3,) for k in s.joints)

    def test_fallback_top_pose_higher_than_impact(self, core, tmp_path):
        impact = core.load_skeleton(tmp_path / "x.json", "Impact")
        top = core.load_skeleton(tmp_path / "x.json", "TopofBackswing")
        # At top of backswing, mid-hands must be higher than at impact.
        assert top.joints["mp"][2] > impact.joints["mp"][2]

    def test_load_skeleton_from_json(self, core, tmp_path):
        path = tmp_path / "skel.json"
        path.write_text(
            json.dumps(
                {
                    "pose": "Test",
                    "joints": {"mp": [0.0, 0.0, 1.0], "ch": [1.0, 0.0, 0.0]},
                    "segments": [["mp", "ch"]],
                }
            )
        )
        s = core.load_skeleton(path, "Test")
        assert s.name == "Test"
        np.testing.assert_allclose(s.joints["mp"], [0, 0, 1])
        np.testing.assert_allclose(s.joints["ch"], [1, 0, 0])
        assert s.segments == [("mp", "ch")]


class TestMocapEvents:
    def test_default_events_all_nan(self, core):
        e = core.MocapEvents()
        for k in ("A", "T", "I", "F"):
            assert getattr(e, f"{k}_sample") != getattr(e, f"{k}_sample")

    def test_frame_for_nan_returns_none(self, core):
        e = core.MocapEvents()
        for k in ("A", "T", "I", "F"):
            assert e.frame_for(k) is None

    def test_frame_for_subtracts_one(self, core):
        e = core.MocapEvents(A_sample=240, T_sample=418, I_sample=525, F_sample=725)
        assert e.frame_for("A") == 239
        assert e.frame_for("T") == 417
        assert e.frame_for("I") == 524
        assert e.frame_for("F") == 724


class TestEventLabelPresets:
    def test_wiffle_default_labels_are_canonical(self, core):
        labels = core.EVENT_LABEL_PRESETS["Wiffle (A/T/I/F)"]
        assert labels["A"] == "Address"
        assert labels["T"] == "Top of Backswing"
        assert labels["I"] == "Impact"
        assert labels["F"] == "Finish"

    def test_all_presets_cover_all_keys(self, core):
        for name, mapping in core.EVENT_LABEL_PRESETS.items():
            assert set(mapping.keys()) == set(
                core.EVENT_KEYS
            ), f"Preset {name!r} missing keys; got {sorted(mapping)}"

    def test_default_preset_exists(self, core):
        assert core.DEFAULT_EVENT_PRESET in core.EVENT_LABEL_PRESETS


class TestPhaseWindows:
    def test_required_phase_keys_exist(self, core):
        for key in (
            "none",
            "backswing",
            "downswing",
            "follow_through",
            "full_swing",
            "manual",
        ):
            assert key in core.PHASE_KEYS, f"Phase key {key!r} missing from PHASE_KEYS"
            assert (
                key in core.PHASE_BOUNDS
            ), f"Phase key {key!r} missing from PHASE_BOUNDS"

    def test_default_phase_key_exists(self, core):
        assert core.DEFAULT_PHASE in core.PHASE_KEYS
        assert core.DEFAULT_PHASE in core.PHASE_BOUNDS

    def test_phase_event_endpoints_are_valid(self, core):
        for key, (a, b) in core.PHASE_BOUNDS.items():
            for end in (a, b):
                if end is None or end == "manual":
                    continue
                assert (
                    end in core.EVENT_KEYS
                ), f"Phase {key!r} references unknown event {end!r}"


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


class TestXlsxLoaders:
    def test_load_mocap_xlsx_columns(self, core):
        _require_xlsx()
        df = core.load_mocap_xlsx(str(_WIFFLE_XLSX), "TW_ProV1")
        assert len(df) > 100
        assert {
            "time",
            "mid_X",
            "mid_Y",
            "mid_Z",
            "club_X",
            "club_Y",
            "club_Z",
        }.issubset(df.columns)

    def test_load_mocap_xlsx_units_are_metres(self, core):
        """Sanity-check: median shaft length is plausible (0.7-1.4 m).
        Catches the cm/inches mix-up bug."""
        _require_xlsx()
        df = core.load_mocap_xlsx(str(_WIFFLE_XLSX), "TW_ProV1")
        shaft = np.linalg.norm(
            df[["club_X", "club_Y", "club_Z"]].values
            - df[["mid_X", "mid_Y", "mid_Z"]].values,
            axis=1,
        )
        finite = np.isfinite(shaft) & (shaft > 1e-3)
        median_shaft = float(np.median(shaft[finite]))
        assert 0.7 < median_shaft < 1.4, (
            f"Median shaft length {median_shaft:.3f} m suggests wrong units "
            "(expected cm->m factor 0.01)."
        )

    def test_event_header_for_prov1(self, core):
        _require_xlsx()
        ev = core.read_event_header(str(_WIFFLE_XLSX), "TW_ProV1")
        for k in ("A", "T", "I", "F"):
            v = getattr(ev, f"{k}_sample")
            assert v == v and v > 0, f"Missing {k}_sample"
        assert 60.0 < ev.CHS_mph < 200.0


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
