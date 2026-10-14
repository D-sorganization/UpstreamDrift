"""Swing tracking of the two-hand club golf humanoid (OSV-9, #11756)."""

from __future__ import annotations

import numpy as np
import pytest
from tests.helpers.opensim_support import require_opensim

from src.engines.physics_engines.opensim.python import msk_club as mc

pytestmark = pytest.mark.unit

SWING = mc.REPO_ROOT / "tests" / "fixtures" / "club_face" / "swing_q_driver.npz"
GRIP_TOL_M = 0.005  # welded lead grip frame to the captured club grip
TRAIL_TOL_M = 0.02  # trail hand to its grip (the weld closes the rest)


@pytest.fixture(scope="module")
def tracked():  # noqa: ANN201
    require_opensim()
    pytest.importorskip("scipy")
    from src.engines.physics_engines.opensim.python import msk_club_tracking as mt

    rows = np.load(SWING)["q"][[0, 10]]
    return mt.track_swing(mc.MODELS_DIR / "golf_humanoid.osim", rows, club="driver")


def test_tracking_follows_the_generated_club(tracked) -> None:  # noqa: ANN001
    assert len(tracked) == 2
    for frame in tracked:
        assert frame.lead_grip_error_m <= GRIP_TOL_M
        assert frame.trail_grip_gap_m <= TRAIL_TOL_M
        assert np.isfinite(list(frame.q.values())).all()


def test_tracking_starts_at_the_calibrated_address(tracked) -> None:  # noqa: ANN001
    calibration = mc.load_calibration("golf_humanoid")
    first = tracked[0].q
    for name in ("pro_sup_l", "wrist_dev_l", "elbow_flex_l"):
        assert first[name] == pytest.approx(calibration.address_q[name], abs=0.05)


def test_track_swing_contracts(tmp_path) -> None:  # noqa: ANN001
    from src.engines.physics_engines.opensim.python import msk_club_tracking as mt

    with pytest.raises(ValueError, match="2-D"):
        mt.track_swing(mc.MODELS_DIR / "golf_humanoid.osim", np.zeros(3))
    with pytest.raises(FileNotFoundError):
        mt.track_swing(tmp_path / "missing.osim", np.zeros((1, 44)))


def _foot_track(rows: int, heel_lift_m: list[float]):  # noqa: ANN202
    """Static address feet (square stance) with a per-row right-heel lift."""
    from src.engines.physics_engines.opensim.python import msk_turn_targets as tt

    ankle = {"l": np.array([0.0, -0.2, 0.08]), "r": np.array([0.0, 0.2, 0.08])}
    axis = {s: np.array([-1.0, 0.0, 0.0]) for s in "lr"}
    lateral = {"l": np.array([0.0, -1.0, 0.0]), "r": np.array([0.0, 1.0, 0.0])}
    feet = tt.PlantedFeet(ankle, axis, lateral, 1)
    tile = lambda v: np.tile(v, (rows, 1))  # noqa: E731
    zero = np.zeros(rows)
    return tt.FootTrack(
        address=feet,
        ankle={s: tile(ankle[s]) for s in "lr"},
        axis={s: tile(axis[s]) for s in "lr"},
        lateral={s: tile(lateral[s]) for s in "lr"},
        heel_dz_m={"l": zero, "r": np.array(heel_lift_m)},
        toe_dz_m={"l": zero, "r": zero},
    )


def test_track_swing_rejects_misaligned_foot_track() -> None:
    from src.engines.physics_engines.opensim.python import msk_club_tracking as mt

    with pytest.raises(ValueError, match="feet has 3 entries for 2 rows"):
        mt.track_swing(
            mc.MODELS_DIR / "golf_humanoid.osim",
            np.zeros((2, 44)),
            feet=_foot_track(3, [0.0, 0.0, 0.0]),
        )


def test_foot_track_lifts_the_tracked_trail_heel() -> None:
    """A per-row heel lift moves the tracked trail calcaneus; static feet do not."""
    require_opensim()
    pytest.importorskip("scipy")
    from src.engines.physics_engines.opensim.python import msk_club_calibration as cal
    from src.engines.physics_engines.opensim.python import msk_club_tracking as mt

    model = mc.MODELS_DIR / "golf_humanoid.osim"
    rows = np.load(SWING)["q"][[0, 0]]
    lift = 0.08
    moving = mt.track_swing(model, rows, feet=_foot_track(2, [0.0, lift]))
    flat = mt.track_swing(model, rows, feet=_foot_track(2, [0.0, 0.0]))

    def calcn_y(frames) -> list[float]:  # noqa: ANN001
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            probe = cal.PoseProbe(cal.skeleton_without_club(model, Path(tmp)))
            heights = []
            for frame in frames:
                probe.set(frame.q)
                heights.append(float(probe.body("calcn_r")[1, 3]))
        return heights

    rise = np.diff(calcn_y(moving))[0]
    assert rise > 0.5 * lift
    assert abs(np.diff(calcn_y(flat))[0]) < 0.01
