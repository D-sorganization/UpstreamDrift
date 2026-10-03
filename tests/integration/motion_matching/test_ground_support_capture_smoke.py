"""MuJoCo native ground-support match smoke test across captures (#11166).

Runs ``src.shared.python.motion_matching.pipeline.cli.run_pipeline`` -- the
MuJoCo native match entry point -- at its smallest existing scale: default
settings, mujoco engine/backend, no shooting-fit/zmp-filter/trajectory
optimiser passes. Parameterized over the public "driver" capture (capture-A)
and the private "owner" capture (capture-O); "owner" skips cleanly when the
private dataset is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.motion_capture.capture_registry import require_capture

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Owner's real-world anthropometry (height, mass), scaled onto the de Leva
# candidate for the owner capture only; the driver capture uses the qualified
# spec's own default geometry (no scaling) as every other committed evidence
# receipt does.
OWNER_STATURE_M = 1.956
OWNER_MASS_KG = 104.3

# Marker RMS bounds (m).  Measured 2026-09-30: IK 0.052 (driver) and 0.083
# (owner); dynamics 0.089 (driver) and 0.567 (owner, #11166 open: the
# driver-tuned rig does not yet track the scaled owner).
IK_RMS_BOUND_M = 0.10
DYNAMICS_RMS_BOUND_M = 0.15


@pytest.mark.parametrize("capture_name", ["driver", "owner"])
def test_mujoco_native_match_smoke(
    tmp_path: Path,
    capture_name: str,
    caplog: pytest.LogCaptureFixture,
    request: pytest.FixtureRequest,
) -> None:
    """The MuJoCo native match runs end to end and tracks the capture."""
    if capture_name == "owner":
        require_capture("capture-O")
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="#11166: owner dynamics marker RMS 0.567 m on the driver-tuned rig",
            )
        )

    from src.shared.python.motion_matching.pipeline.cli import (
        build_parser,
        run_pipeline,
    )

    argv = ["--out", str(tmp_path), "--capture", capture_name]
    if capture_name == "owner":
        argv += [
            "--anthropometric",
            str(OWNER_STATURE_M),
            str(OWNER_MASS_KG),
        ]
    args = build_parser().parse_args(argv)

    with caplog.at_level("INFO"):
        receipt = run_pipeline(args)

    assert receipt["capture"] == capture_name
    assert receipt["ik"]["frames"] > 0
    assert math.isfinite(receipt["ik"]["marker_rms_m"])
    assert math.isfinite(receipt["dynamics"]["marker_rms_m"])
    assert (tmp_path / "receipt.json").is_file()
    assert receipt["ik"]["marker_rms_m"] < IK_RMS_BOUND_M
    assert receipt["dynamics"]["marker_rms_m"] < DYNAMICS_RMS_BOUND_M
