"""GM-1b (#14423): X-factor variants mapped to literature families."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.x_factor_literature import (
    X_FACTOR_REGISTRY_VERSION,
    x_factor_publish_payload,
    x_factor_variant_meta,
)
from src.shared.python.spine.golf_terms import evaluate
from src.shared.python.swing_comparison.events import SwingEvents

pytestmark = pytest.mark.unit

EPS_DEG = 1e-9
_ROT = "pelvis_to_thorax_rotation"


def _ry(deg: float) -> np.ndarray:
    return Rotation.from_euler("y", deg, degrees=True).as_matrix()


def test_proxy_flagged_non_literature_with_version_and_cites() -> None:
    meta = x_factor_variant_meta("x_factor_position_proxy")
    assert meta.literature_family == "non_literature_legacy"
    assert meta.literature_aligned is False
    assert meta.non_literature_note
    payload = x_factor_publish_payload("x_factor_position_proxy", [1.0, 2.0])
    assert payload["registry_version"] == X_FACTOR_REGISTRY_VERSION
    assert payload["literature_aligned"] is False
    assert "non_literature_note" in payload


def test_jcs_axial_matches_synthetic_relative_rotation() -> None:
    shoulder_deg, hip_deg = 35.0, 12.0
    sep = shoulder_deg - hip_deg
    # Same pelvis→thorax convention as ``test_pure_axial_45_right_handed``: -AR backswing.
    want = float(
        evaluate("x_factor_jcs_axial", {_ROT: _ry(-sep)}, handedness="right").reshape(
            -1
        )[0]
    )
    meta = x_factor_variant_meta("x_factor_jcs_axial")
    assert meta.literature_family == "brown_kwon_joint_angle"
    np.testing.assert_allclose(want, sep, rtol=0, atol=EPS_DEG)


def test_euler_yaw_matches_synthetic_thorax_minus_pelvis_yaw() -> None:
    yaw_s, yaw_h = 1.1, -0.4
    thorax = np.array([[0.0, 0.0, yaw_s]])
    pelvis = np.array([[0.0, 0.0, yaw_h]])
    got = evaluate(
        "x_factor_euler_yaw_legacy_v1",
        {"thorax_euler": thorax, "pelvis_euler": pelvis},
        handedness="right",
    )
    np.testing.assert_allclose(
        got,
        np.degrees(yaw_s - yaw_h),
        rtol=0,
        atol=EPS_DEG,
    )
    assert (
        x_factor_variant_meta("x_factor_euler_yaw_legacy_v1").literature_family
        == "brown_kwon_joint_angle"
    )


def _events(n: int = 10) -> SwingEvents:
    return SwingEvents(
        address_idx=0,
        address_time=0.0,
        top_idx=n // 2,
        top_time=(n // 2) * 0.01,
        impact_idx=n - 2,
        impact_time=(n - 2) * 0.01,
        finish_idx=n - 1,
        finish_time=(n - 1) * 0.01,
    )


def _marker_set(upper_deg: np.ndarray, pelvis_deg: np.ndarray) -> dict[str, np.ndarray]:
    """Minimal Back*/Waist* geometry (see ``tests/shared/python/spine/test_golf_terms``)."""

    def line(angle_deg: np.ndarray, centre: tuple[float, float, float]):
        phi = -np.deg2rad(angle_deg) + 0.3
        half = 0.2 * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
        c = np.asarray(centre)
        return c + half, c - half

    out: dict[str, np.ndarray] = {}
    for (left, right), (ang, c) in {
        ("BackLeft", "BackRight"): (upper_deg, (0.0, 0.0, 1.3)),
        ("WaistLeft", "WaistRight"): (pelvis_deg, (0.0, 0.0, 1.0)),
        ("LShoulderBack", "RShoulderBack"): (upper_deg, (0.0, 0.0, 1.35)),
    }.items():
        lp, rp = line(ang, c)
        out[left], out[right] = lp, rp
    return out


def test_line_turn_matches_mclean_style_horizontal_separation() -> None:
    n = 10
    t = np.arange(n) * 0.01
    upper = np.linspace(0.0, 90.0, n)
    pelvis = np.linspace(0.0, 40.0, n)
    inputs = {
        "points": _marker_set(upper, pelvis),
        "t": t,
        "events": _events(n),
        "up": np.array([0.0, 0.0, 1.0]),
    }
    out = evaluate("x_factor_line_turn", inputs, handedness="right")
    assert out[-1] == pytest.approx(50.0)
    assert (
        x_factor_variant_meta("x_factor_line_turn").literature_family
        == "mclean_line_projection"
    )


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown X-factor"):
        x_factor_variant_meta("x_factor_nope")
