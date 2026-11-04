"""Tests for address grip roll metric (SHOW-A1, #14550).

Verifies:
1. `address_grip_roll_deg(model_state)` enforces SO(3) orthonormality via `require`.
2. Synthetic hand at 0°, 15°, 45° strong and weak recovers within 0.5°.
3. Documented mapping to HX alpha/gamma decomposition (#13757).
4. Flagship driver model evaluation reports current best-fit value (~90°-105° strong).
5. Face-on overlay rendering generates valid PNG artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ContractViolationError
from src.shared.python.motion_matching.address_grip import (
    AddressGripRoll,
    AddressGripState,
    address_grip_roll_deg,
    flagship_driver_grip_roll,
    render_address_grip_overlay_png,
)
from src.shared.python.motion_matching.grip_contract import (
    HandAngles,
    grip_matrix,
    hand_from_euler,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DRIVER_SPEC_PATH = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def _make_rot(axis: str, deg: float) -> np.ndarray:
    return Rotation.from_euler(axis, deg, degrees=True).as_matrix()


def test_orthonormality_enforced_on_frames() -> None:
    """Precondition: frames must be proper SO(3) rotations, else ContractViolationError."""
    # Non-orthogonal matrix
    bad_rot = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    eye = np.eye(3)
    state = AddressGripState(
        lead_hand_frame=bad_rot,
        trail_hand_frame=eye,
        shaft_frame=eye,
    )
    with pytest.raises((ContractViolationError, ValueError)):
        address_grip_roll_deg(state)

    # Improper rotation (det = -1)
    reflection = np.diag([-1.0, 1.0, 1.0])
    bad_state = AddressGripState(
        lead_hand_frame=reflection,
        trail_hand_frame=eye,
        shaft_frame=eye,
    )
    with pytest.raises((ContractViolationError, ValueError)):
        address_grip_roll_deg(bad_state)


@pytest.mark.parametrize("nominal_roll_deg", [0.0, 15.0, 45.0, -15.0, -45.0])
def test_synthetic_lead_hand_roll_recovered(nominal_roll_deg: float) -> None:
    """Synthetic lead hand at 0°, 15°, 45° strong/weak recovers value within 0.5°."""
    # Shaft axis along +y in club frame
    shaft_rot = np.eye(3)
    # Roll about the shaft (+y): positive = clockwise looking butt-to-head = strong
    # In club frame: top of grip is +x, trail is -z (or +z depending on convention)
    # R_y(nominal_roll_deg) rotates the hand about the shaft
    lead_hand_rot = _make_rot("y", nominal_roll_deg)
    trail_hand_rot = np.eye(3)

    state = AddressGripState(
        lead_hand_frame=lead_hand_rot,
        trail_hand_frame=trail_hand_rot,
        shaft_frame=shaft_rot,
    )
    result = address_grip_roll_deg(state)
    assert isinstance(result, (tuple, AddressGripRoll))
    lead_roll = result[0] if isinstance(result, tuple) else result.lead_deg
    assert lead_roll == pytest.approx(nominal_roll_deg, abs=0.5)


@pytest.mark.parametrize("nominal_roll_deg", [0.0, 15.0, 45.0, -15.0, -45.0])
def test_synthetic_trail_hand_roll_recovered(nominal_roll_deg: float) -> None:
    """Synthetic trail hand at 0°, 15°, 45° strong/weak recovers value within 0.5°."""
    shaft_rot = np.eye(3)
    lead_hand_rot = np.eye(3)
    trail_hand_rot = _make_rot("y", nominal_roll_deg)

    state = AddressGripState(
        lead_hand_frame=lead_hand_rot,
        trail_hand_frame=trail_hand_rot,
        shaft_frame=shaft_rot,
    )
    result = address_grip_roll_deg(state)
    trail_roll = result[1] if isinstance(result, tuple) else result.trail_deg
    assert trail_roll == pytest.approx(nominal_roll_deg, abs=0.5)


def test_hx_alpha_gamma_decomposition_mapping() -> None:
    """Documented mapping: roll tracks HX alpha across varied psi and gamma."""
    # M = R_y(alpha) R_x(psi - 90) R_z(gamma) (#13757)
    for alpha in [-45.0, -15.0, 0.0, 15.0, 45.0]:
        for psi in [25.0, 40.0, 65.0]:
            for gamma in [-20.0, 0.0, 20.0]:
                m = grip_matrix(alpha, psi, gamma)
                state = AddressGripState(
                    lead_hand_frame=m,
                    trail_hand_frame=np.eye(3),
                    shaft_frame=np.eye(3),
                )
                res = address_grip_roll_deg(state)
                lead_roll = res[0] if isinstance(res, tuple) else res.lead_deg
                assert lead_roll == pytest.approx(alpha, abs=0.5)


def test_flagship_driver_address_grip_roll_reports_current_value() -> None:
    """Flagship driver model currently sits in ~90°-105° strong grip (#14549)."""
    assert DRIVER_SPEC_PATH.exists(), f"Driver spec not found at {DRIVER_SPEC_PATH}"
    spec = json.loads(DRIVER_SPEC_PATH.read_text())
    rolls = flagship_driver_grip_roll(spec)
    lead, trail = rolls.lead_deg, rolls.trail_deg
    # Verified in analysis: lead ~94.1°, trail ~104.3°
    assert 90.0 <= lead <= 100.0, f"Expected lead roll ~94°, got {lead}"
    assert 100.0 <= trail <= 110.0, f"Expected trail roll ~104°, got {trail}"


def test_render_address_grip_overlay_png(tmp_path: Path) -> None:
    """Render face-on overlay PNG and verify file existence and non-zero size."""
    spec = json.loads(DRIVER_SPEC_PATH.read_text())
    output_png = tmp_path / "test_overlay.png"
    render_address_grip_overlay_png(spec, output_png)
    assert output_png.exists()
    assert output_png.stat().st_size > 1000
