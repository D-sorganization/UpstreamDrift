"""Tests for the golf-term registry (SPN-4a, ADR-0055 items 4 and 8)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.analysis.swing_metrics import SwingMetricsMixin
from src.shared.python.biomechanics.joint_conventions import anatomical_joint_angles
from src.shared.python.injury.spinal_load_analysis import SpinalLoadAnalyzer
from src.shared.python.spine import golf_terms
from src.shared.python.spine.golf_terms import (
    GolfTermDefinition,
    definition,
    evaluate,
    registered_terms,
    segmental_contributions,
    contribution_remainder,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import LevelTransform, SegmentalPose
from src.shared.python.swing_comparison import turn as turn_mod
from src.shared.python.swing_comparison.events import SwingEvents

pytestmark = pytest.mark.unit

EXPECTED_TERMS = {
    "x_factor_jcs_axial",
    "x_factor_line_turn",
    "x_factor_position_proxy",
    "x_factor_euler_yaw_legacy_v1",
    "trunk_extension_jcs",
    "trunk_lateral_flexion_jcs",
    "side_bend_tilt",
    "crunch_factor_legacy_v1",
    "crunch_factor_rate",
    "early_extension",
    "reverse_spine_angle",
}


def _ry(deg: float) -> np.ndarray:
    return Rotation.from_euler("y", deg, degrees=True).as_matrix()


def _rx(deg: float) -> np.ndarray:
    return Rotation.from_euler("x", deg, degrees=True).as_matrix()


def _rz(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


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


# -- registry -----------------------------------------------------------------


def test_registered_terms_are_the_adr_table() -> None:
    assert set(registered_terms()) == EXPECTED_TERMS


@pytest.mark.parametrize("term_id", sorted(EXPECTED_TERMS))
def test_definitions_are_complete(term_id: str) -> None:
    d = definition(term_id)
    assert isinstance(d, GolfTermDefinition)
    assert d.term_id == term_id
    for field in (d.version, d.description, d.frame, d.sign_convention, d.units):
        assert field.strip()
    assert d.reference.strip()
    assert d.required_inputs


def test_crunch_rate_reference_is_sugaya_1999() -> None:
    assert "Sugaya" in definition("crunch_factor_rate").reference


def test_unknown_term_raises() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        definition("nonsense")
    with pytest.raises(ValueError, match="nonsense"):
        evaluate("nonsense", {}, handedness="right")


def test_bad_handedness_raises() -> None:
    with pytest.raises(ValueError, match="handedness"):
        evaluate(
            "x_factor_jcs_axial",
            {"pelvis_to_thorax_rotation": np.eye(3)},
            handedness="both",  # type: ignore[arg-type]
        )


def test_missing_inputs_are_named() -> None:
    with pytest.raises(ValueError, match="pelvis_to_thorax_rotation"):
        evaluate("x_factor_jcs_axial", {}, handedness="right")
    with pytest.raises(ValueError) as exc:
        evaluate("early_extension", {"up": [0, 0, 1]}, handedness="right")
    for name in ("pelvis_position", "target", "events", "ball_position"):
        assert name in str(exc.value)


# -- JCS terms ----------------------------------------------------------------


def test_pure_axial_45_right_handed() -> None:
    # +AR is LEFT axial rotation; a right-hander's backswing is -AR.
    inputs = {"pelvis_to_thorax_rotation": _ry(-45.0)}
    assert evaluate("x_factor_jcs_axial", inputs, handedness="right") == (
        pytest.approx([45.0])
    )
    assert evaluate(
        "trunk_lateral_flexion_jcs", inputs, handedness="right"
    ) == pytest.approx([0.0], abs=1e-9)
    assert evaluate("trunk_extension_jcs", inputs, handedness="right") == (
        pytest.approx([0.0], abs=1e-9)
    )


def test_axial_handedness_flip() -> None:
    inputs = {"pelvis_to_thorax_rotation": _ry(-45.0)}
    right = evaluate("x_factor_jcs_axial", inputs, handedness="right")
    left = evaluate("x_factor_jcs_axial", inputs, handedness="left")
    assert left == pytest.approx(-right)


def test_axial_sign_matches_isb_thorax_left_axial_rotation() -> None:
    rot = _ry(30.0)
    thorax = anatomical_joint_angles(rot, "isb_thorax", degrees=True).values
    spine = anatomical_joint_angles(rot, "isb_spine", degrees=True).values
    assert thorax[2] == pytest.approx(spine[2])  # same sense, both left-positive
    inputs = {"pelvis_to_thorax_rotation": rot}
    assert evaluate("x_factor_jcs_axial", inputs, handedness="right")[0] == (
        pytest.approx(-thorax[2])
    )
    assert evaluate("x_factor_jcs_axial", inputs, handedness="left")[0] == (
        pytest.approx(thorax[2])
    )


def test_extension_positive_and_not_handed() -> None:
    # Positive FE is a +Z rotation (superior Y tilts posteriorly).
    inputs = {"pelvis_to_thorax_rotation": _rz(20.0)}
    ext_r = evaluate("trunk_extension_jcs", inputs, handedness="right")
    ext_l = evaluate("trunk_extension_jcs", inputs, handedness="left")
    assert ext_r == pytest.approx([20.0])
    assert ext_l == pytest.approx(ext_r)
    sup_y = _rz(20.0) @ np.array([0.0, 1.0, 0.0])
    assert sup_y[0] < 0  # tilts posteriorly (X is anterior)


def test_lateral_flexion_is_lead_side_positive() -> None:
    # +LB is right lateral bending; right-hander lead side is left: -LB.
    inputs = {"pelvis_to_thorax_rotation": _rx(12.0)}
    assert evaluate("trunk_lateral_flexion_jcs", inputs, handedness="right") == (
        pytest.approx([-12.0])
    )
    assert evaluate("trunk_lateral_flexion_jcs", inputs, handedness="left") == (
        pytest.approx([12.0])
    )


def test_jcs_series_input() -> None:
    series = np.stack([_ry(-a) for a in (0.0, 10.0, 20.0)])
    out = evaluate(
        "x_factor_jcs_axial",
        {"pelvis_to_thorax_rotation": series},
        handedness="right",
    )
    assert out == pytest.approx([0.0, 10.0, 20.0])


def test_jcs_rejects_non_rotation() -> None:
    with pytest.raises(ValueError, match="rotation"):
        evaluate(
            "x_factor_jcs_axial",
            {"pelvis_to_thorax_rotation": np.eye(3) * 2.0},
            handedness="right",
        )
    with pytest.raises(ValueError, match="shape"):
        evaluate(
            "x_factor_jcs_axial",
            {"pelvis_to_thorax_rotation": np.eye(4)},
            handedness="right",
        )


def test_jcs_singular_orientation_is_nan_not_zero() -> None:
    rot = _rx(90.0)  # gimbal lock of ZXY: |LB| = 90 deg
    out = evaluate(
        "x_factor_jcs_axial",
        {"pelvis_to_thorax_rotation": rot},
        handedness="right",
    )
    assert np.isnan(out).all()


# -- segmental contributions --------------------------------------------------


def _pose(*angles_deg: tuple[float, float, float]) -> SegmentalPose:
    levels = list(SpinalLevel)[: len(angles_deg)]
    return SegmentalPose(
        tuple(
            LevelTransform.from_isb_angles(lv, np.deg2rad(a))
            for lv, a in zip(levels, angles_deg, strict=True)
        )
    )


def test_contributions_pure_axial_are_additive() -> None:
    pose = _pose((0, 0, -10.0), (0, 0, -15.0), (0, 0, -20.0))
    contrib = segmental_contributions("x_factor_jcs_axial", pose, handedness="right")
    assert list(contrib) == list(pose.levels)
    assert list(contrib.values()) == pytest.approx([10.0, 15.0, 20.0])
    assert contribution_remainder(
        "x_factor_jcs_axial", pose, handedness="right"
    ) == pytest.approx(0.0, abs=1e-9)


def test_contributions_are_a_decomposition_not_an_identity() -> None:
    pose = _pose((-10, 5, -15), (-10, 8, -20), (-15, 10, -25))
    contrib = segmental_contributions("trunk_extension_jcs", pose, handedness="right")
    glob = evaluate(
        "trunk_extension_jcs",
        {"pelvis_to_thorax_rotation": pose.compose()[:3, :3]},
        handedness="right",
    )[0]
    rem = contribution_remainder("trunk_extension_jcs", pose, handedness="right")
    assert sum(contrib.values()) + rem == pytest.approx(glob)
    assert rem == pytest.approx(0.0, abs=1e-9)  # prefix differences telescope
    own = [lt.isb_angles()[0] for lt in pose.transforms]
    # a level's share is not its own Cardan angle: angles do not add
    assert abs(np.degrees(sum(own)) - glob) > 1e-3
    assert list(contrib.values()) != pytest.approx(np.degrees(own))


def test_contributions_telescope_to_prefix_difference() -> None:
    pose = _pose((0, 0, -10.0), (0, 10.0, -20.0))
    contrib = segmental_contributions("x_factor_jcs_axial", pose, handedness="left")
    lv0, lv1 = pose.levels
    t0 = evaluate(
        "x_factor_jcs_axial",
        {"pelvis_to_thorax_rotation": pose.relative_rotation(lv0, lv0)},
        handedness="left",
    )[0]
    t1 = evaluate(
        "x_factor_jcs_axial",
        {"pelvis_to_thorax_rotation": pose.relative_rotation(lv0, lv1)},
        handedness="left",
    )[0]
    assert contrib[lv0] == pytest.approx(t0)
    assert contrib[lv1] == pytest.approx(t1 - t0)


def test_contributions_reject_non_jcs_term() -> None:
    with pytest.raises(ValueError, match="crunch_factor_legacy_v1"):
        segmental_contributions(
            "crunch_factor_legacy_v1", _pose((0, 0, 1.0)), handedness="right"
        )
    with pytest.raises(TypeError):
        segmental_contributions(
            "x_factor_jcs_axial",
            np.eye(3),
            handedness="right",  # type: ignore[arg-type]
        )


# -- line turn (delegation) ---------------------------------------------------


def _marker_set(
    upper_deg: np.ndarray, pelvis_deg: np.ndarray, rot: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    def line(angle_deg: np.ndarray, centre: tuple[float, float, float]):
        phi = -np.deg2rad(angle_deg) + 0.3  # turn = -(phi - phi0)
        half = 0.2 * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
        c = np.asarray(centre)
        return c + half, c - half  # left, right

    out: dict[str, np.ndarray] = {}
    for (left, right), (ang, c) in {
        ("BackLeft", "BackRight"): (upper_deg, (0.0, 0.0, 1.3)),
        ("WaistLeft", "WaistRight"): (pelvis_deg, (0.0, 0.0, 1.0)),
        ("LShoulderBack", "RShoulderBack"): (upper_deg, (0.0, 0.0, 1.35)),
    }.items():
        lp, rp = line(ang, c)
        out[left], out[right] = lp, rp
    if rot is not None:
        out = {k: v @ rot.T for k, v in out.items()}
    return out


def _turn_inputs(rot: np.ndarray | None = None) -> dict[str, object]:
    n = 10
    t = np.arange(n) * 0.01
    upper = np.linspace(0.0, 90.0, n)
    pelvis = np.linspace(0.0, 40.0, n)
    up = np.array([0.0, 0.0, 1.0]) if rot is None else rot @ [0.0, 0.0, 1.0]
    return {
        "points": _marker_set(upper, pelvis, rot),
        "t": t,
        "events": _events(n),
        "up": up,
    }


def test_line_turn_matches_turn_module() -> None:
    inputs = _turn_inputs()
    expected = turn_mod.marker_turn_lines(
        inputs["points"], inputs["t"], inputs["events"]
    ).x_factor.turn_deg
    out = evaluate("x_factor_line_turn", inputs, handedness="right")
    assert out == pytest.approx(expected, nan_ok=True)
    assert out[-1] == pytest.approx(50.0)


def test_line_turn_delegates_to_turn_module() -> None:
    inputs = _turn_inputs()
    with mock.patch.object(
        golf_terms, "compute_turn_lines", wraps=turn_mod.compute_turn_lines
    ) as spy:
        evaluate("x_factor_line_turn", inputs, handedness="right")
    spy.assert_called_once()


def test_line_turn_left_handed_flips_and_up_is_explicit() -> None:
    right = evaluate("x_factor_line_turn", _turn_inputs(), handedness="right")
    left = evaluate("x_factor_line_turn", _turn_inputs(), handedness="left")
    assert left == pytest.approx(-right)
    tilted = Rotation.from_euler("xyz", [31, -17, 52], degrees=True).as_matrix()
    again = evaluate("x_factor_line_turn", _turn_inputs(tilted), handedness="right")
    assert again == pytest.approx(right, abs=1e-6)


# -- legacy terms (characterisation) ------------------------------------------


def test_crunch_legacy_equals_spinal_load_analyzer() -> None:
    rng = np.random.default_rng(7)
    n = 60
    time = np.linspace(0.0, 1.2, n)
    lateral = rng.uniform(-0.6, 0.6, n)
    rotation = rng.uniform(-1.2, 1.2, n)
    analyzer = SpinalLoadAnalyzer(body_weight=80.0)
    legacy = analyzer._compute_crunch_factor(lateral, rotation, time)
    out = evaluate(
        "crunch_factor_legacy_v1",
        {"lateral_bend_rad": lateral, "axial_rotation_rad": rotation},
        handedness="right",
    )
    np.testing.assert_allclose(out, legacy.crunch_factor, rtol=0, atol=1e-12)


def test_euler_yaw_x_factor_equals_spinal_load_analyzer() -> None:
    rng = np.random.default_rng(11)
    n = 50
    time = np.linspace(0.0, 1.0, n)
    pelvis = rng.uniform(-0.8, 0.8, (n, 3))
    thorax = rng.uniform(-1.2, 1.2, (n, 3))
    analyzer = SpinalLoadAnalyzer(body_weight=80.0)
    legacy = analyzer._compute_x_factor(pelvis, thorax, time)
    out = evaluate(
        "x_factor_euler_yaw_legacy_v1",
        {"thorax_euler": thorax, "pelvis_euler": pelvis},
        handedness="left",  # legacy sign is handedness-free
    )
    np.testing.assert_allclose(out, legacy.x_factor_angle, rtol=0, atol=1e-12)


def test_position_proxy_equals_swing_metrics() -> None:
    rng = np.random.default_rng(3)
    positions = rng.uniform(-1.0, 1.0, (40, 6))

    class _Ctx(SwingMetricsMixin):
        pass

    ctx = _Ctx()
    ctx.joint_positions = positions
    ctx.dt = 0.01
    legacy = ctx.compute_x_factor(4, 1)
    out = evaluate(
        "x_factor_position_proxy",
        {"joint_positions": positions, "shoulder_joint_idx": 4, "hip_joint_idx": 1},
        handedness="right",
    )
    np.testing.assert_allclose(out, legacy, rtol=0, atol=1e-12)


def test_position_proxy_index_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="shoulder_joint_idx"):
        evaluate(
            "x_factor_position_proxy",
            {
                "joint_positions": np.zeros((5, 3)),
                "shoulder_joint_idx": 9,
                "hip_joint_idx": 1,
            },
            handedness="right",
        )


def test_crunch_rate_is_lateral_times_axial_velocity() -> None:
    t = np.linspace(0.0, 1.0, 101)
    axial = 50.0 * t  # deg, 50 deg/s
    lateral = np.full_like(t, 8.0)  # deg
    out = evaluate(
        "crunch_factor_rate",
        {
            "trunk_lateral_flexion_jcs": lateral,
            "x_factor_jcs_axial": axial,
            "time_s": t,
        },
        handedness="right",
    )
    assert out == pytest.approx(np.full_like(t, 400.0))


# -- lab-frame terms ----------------------------------------------------------

UP = np.array([0.0, 0.0, 1.0])
TARGET = np.array([0.0, -1.0, 0.0])


def test_side_bend_tilt_components() -> None:
    axis = np.array([0.0, -np.sin(np.deg2rad(10.0)), np.cos(np.deg2rad(10.0))])
    out = evaluate(
        "side_bend_tilt",
        {"trunk_axis": axis, "up": UP, "target": TARGET},
        handedness="right",
    )
    # [total, toward_target, lateral]; tilt entirely toward the target
    assert out == pytest.approx([10.0, 10.0, 0.0], abs=1e-9)


def test_reverse_spine_angle_toward_target_positive_at_top() -> None:
    n = 10
    ev = _events(n)
    tilt = np.zeros(n)
    tilt[ev.top_idx] = 12.0  # deg toward target
    ang = np.deg2rad(tilt)
    axes = np.stack([np.zeros(n), -np.sin(ang), np.cos(ang)], axis=1)
    inputs = {"trunk_axis": axes, "up": UP, "target": TARGET, "events": ev}
    out = evaluate("reverse_spine_angle", inputs, handedness="right")
    assert out == pytest.approx([12.0])
    away = dict(inputs, target=-TARGET)
    assert evaluate("reverse_spine_angle", away, handedness="right") == (
        pytest.approx([-12.0])
    )


def test_early_extension_positive_toward_ball() -> None:
    n = 10
    ev = _events(n)
    ball = np.array([0.0, 0.0, 0.0])
    # golfer stands at x = +0.6 (perpendicular to the target line, along Y)
    pelvis = np.tile([0.6, 0.0, 1.0], (n, 1)).astype(float)
    pelvis[:, 0] -= np.linspace(0.0, 0.05, n)  # moves 5 cm toward the ball line
    pelvis[:, 1] += np.linspace(0.0, 0.3, n)  # along-target motion must not count
    inputs = {
        "pelvis_position": pelvis,
        "target": TARGET,
        "up": UP,
        "events": ev,
        "ball_position": ball,
    }
    out = evaluate("early_extension", inputs, handedness="right")
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[-1] == pytest.approx(0.05)
    inputs["pelvis_position"] = pelvis[::-1].copy()
    inputs["pelvis_position"][:, 0] = 1.2 - pelvis[:, 0]  # away from the ball
    away = evaluate("early_extension", inputs, handedness="right")
    assert away[-1] < 0


@pytest.mark.parametrize(
    ("term_id", "inputs"),
    [
        (
            "x_factor_euler_yaw_legacy_v1",
            {
                "thorax_euler": np.array([[0.0, 0.0, 0.4], [0.0, 0.0, np.nan]]),
                "pelvis_euler": np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]]),
            },
        ),
        (
            "crunch_factor_legacy_v1",
            {
                "lateral_bend_rad": np.array([0.2, np.nan]),
                "axial_rotation_rad": np.array([0.3, 0.1]),
            },
        ),
    ],
)
def test_legacy_terms_propagate_nan_dropouts(term_id: str, inputs: dict) -> None:
    """Legacy measurands keep the pre-registry NaN passthrough (SPN-4b)."""
    out = evaluate(term_id, inputs, handedness="right")
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
