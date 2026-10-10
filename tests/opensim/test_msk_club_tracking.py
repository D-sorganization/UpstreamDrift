"""Swing tracking of the two-hand club golf humanoid (OSV-9, #11756)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import msk_club as mc

pytestmark = pytest.mark.unit

SWING = mc.REPO_ROOT / "tests" / "fixtures" / "club_face" / "swing_q_driver.npz"
GRIP_TOL_M = 0.005  # welded lead grip frame to the captured club grip
TRAIL_TOL_M = 0.02  # trail hand to its grip (the weld closes the rest)


@pytest.fixture(scope="module")
def tracked():  # noqa: ANN201
    pytest.importorskip("opensim")
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
