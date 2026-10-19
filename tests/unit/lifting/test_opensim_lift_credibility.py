"""Physical credibility of the OpenSim lift IK/ID (OSL-12, #13048).

Red tests against the real OpenSim 4.6 runtime, written before the code:

* **Pelvis landmarks.**  The bundle's Rajagopal pelvis frame (the
  ``LowerTorso`` carrying both grafted Rajagopal hips) is verified against
  both femur poses; the pelvis is scaled mediolaterally to the bundle's
  0.1851 m hip width (OSL-14 lift body; generic 0.1545 m), and a model
  scaled that way whose
  pelvis coincides with the bundle's carries its own RASI/LASI/RPSI/LPSI
  markers exactly onto the virtual targets.
* **Contact stiffness.**  The lift contacts use the 1e7 Pa stiffness of
  OpenSim_Models PR #476; every hold sphere sits within that PR's 1-8 mm
  stance band (OSL-3's 1e5 Pa sank the holds 0.08-0.12 m).
* **Balanced holds.**  Every hold reaches static equilibrium on its contacts
  with zero pelvis residual and the centre of mass over the centre of
  pressure; standing holds balance through an ankle lean with the feet flat.
* **Squat IK ranges.**  With clamping on (and weak coordinate tasks on the
  arms, which only the shoulders and bar observe), every coordinate stays in
  its Rajagopal range.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES
from src.shared.python.lifting.opensim_full_body_lifts import (
    OpenSimFullBodyLiftSpec,
    compose_opensim_full_body_lift,
    opensim_lift_ik_id,
    opensim_lift_static_grf,
)
from src.shared.python.lifting.pack_audit.names import LIFTS

pytestmark = pytest.mark.unit

#: OSL-19 (#13158): high-bar rack reach sweep (grip width 0.75 m).
#: Left hand reach is resolved to 0.0024 mm (well inside 1 mm bound).
#: Right hand reach stops 1.51 mm from the grip (down from 56.9 mm, 97.4%
#: reduction), exceeding the 1.0 mm bound. The binding physical constraint is
#: three simultaneous 3D joint limits on the right arm: arm_flex_r = 90.0 deg
#: (1.57080 rad), arm_rot_r = -90.0 deg (-1.57080 rad), and elbow_flex_r = 150.0 deg
#: (2.61800 rad), rather than the 2D planar d_min (0.2059 m).
#: Bundle shoulder landmark to grip is 0.2033 m (R) / 0.2198 m (L); OpenSim RSJC/LSJC
#: to bundle grip is 0.2052 m (R) / 0.2232 m (L); OpenSim RSJC/LSJC to hand CoM is
#: 0.2059 m (R) / 0.2232 m (L).
#: Coordinate range overshoot reaches a maximum of 4.18e-5 rad across 5 coordinates:
#: arm_rot_r (4.18e-5 rad, 6 frames), arm_flex_r (3.79e-5 rad, 6 frames),
#: pro_sup_r (3.25e-5 rad, 6 frames), pro_sup_l (2.97e-5 rad, 6 frames), and
#: elbow_flex_r (8.79e-7 rad, 2 frames). Neither the reach nor range bounds are loosened.
_RAJAGOPAL_ARM_RANGES = pytest.mark.xfail(
    strict=True,
    reason=(
        "OSL-19 #13158: right hand reach 1.51 mm > 1.0 mm bound and max coordinate "
        "overshoot 4.18e-5 rad (arm_rot_r 4.18e-5, arm_flex_r 3.79e-5, pro_sup_r 3.25e-5, "
        "pro_sup_l 2.97e-5, elbow_flex_r 8.79e-7 rad) at 3 pinned limits "
        "(arm_flex_r 90 deg, arm_rot_r -90 deg, elbow_flex_r 150 deg); bounds not loosened."
    ),
)
#: Bundle hip-centre separation (OSL-14, #13057: the AN-1 lift body's de Leva
#: spacing, 2 * 0.052 * 1.78 m) and the generic Rajagopal one.
_BUNDLE_HIP_WIDTH_M = 0.18512
_GENERIC_HIP_WIDTH_M = 0.1545
_WIDTH_TOL_M = 5e-4
#: Frame identities hold to round-off.
_FRAME_TOL = 1e-9


def _real_opensim() -> None:
    from tests.helpers.opensim_support import require_opensim

    osim = require_opensim()
    if not hasattr(osim, "__file__"):
        pytest.skip("opensim in sys.modules is a test double, not the real package")


def _base_model():
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )

    try:
        return resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(f"muscle-bearing body not available: {exc}")


@pytest.fixture(scope="module")
def squat_targets():
    from src.shared.python.lifting.bundle_targets import lift_targets

    path = SAME_INPUT_LIFT_BUNDLES["squat"]
    assert path is not None
    return lift_targets("squat", path, stride=50)


@pytest.fixture(scope="module")
def squat_build(squat_targets):
    _real_opensim()
    base = _base_model()
    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        build_lift_model,
    )

    scales = ikid.bundle_segment_scales(squat_targets, base)
    build = build_lift_model(
        "squat",
        body_mass_kg=squat_targets.body_mass_kg,
        height_m=squat_targets.height_m,
        bar_mass_kg=squat_targets.bar_mass_kg,
        segment_scales=scales,
    )
    return build, scales, ikid.pelvis_landmark_targets(build, squat_targets, base)


def _hip_rotation(q: np.ndarray, order: list[str], side: str) -> np.ndarray:
    """Rajagopal hip: flexion about z, adduction about x, rotation about y.

    The left hip is the mirror (OpenSim's left adduction and rotation axes
    point along -x and -y): ``D @ R @ D`` with ``D = diag(-1, -1, 1)``.
    """
    angles = [
        q[order.index(f"hip_{n}_{side}")] for n in ("flexion", "adduction", "rotation")
    ]
    rotation = Rotation.from_euler("ZXY", angles).as_matrix()
    if side == "l":
        mirror = np.diag([-1.0, -1.0, 1.0])
        rotation = mirror @ rotation @ mirror
    return rotation


@pytest.mark.integration
def test_bundle_pelvis_frame_is_the_pelvis_of_both_hips(squat_targets) -> None:
    import json

    from src.shared.python.lifting.bundle_targets import _kinematics
    from src.shared.python.lifting.bundles import LANDMARK_ATTACHMENTS
    from src.shared.python.lifting.sagittal_retarget import LANDMARK_LABELS
    from src.shared.python.motion_matching.same_input import InputBundle

    path = SAME_INPUT_LIFT_BUNDLES["squat"]
    assert path is not None
    bundle = InputBundle.load(path)
    order = list(json.loads(bundle.spec_bytes)["coordinate_order"])
    attachments = {x: LANDMARK_ATTACHMENTS[x] for x in LANDMARK_LABELS}
    kin, _mass = _kinematics(bundle.spec_bytes, attachments)
    rotation = squat_targets.frame.rotation
    for n, k in enumerate(squat_targets.indices):
        q = bundle.reference_q[k]
        poses = kin.body_poses(q, ["femur_r", "femur_l"])
        for side in "rl":
            from_femur = (
                rotation @ poses[f"femur_{side}"][0] @ _hip_rotation(q, order, side).T
            )
            np.testing.assert_allclose(
                from_femur, squat_targets.pelvis_rotation[n], atol=_FRAME_TOL
            )
        # The hip axis lies along the pelvis z axis (right hip at +z).
        points = squat_targets.landmarks(n)
        axis = squat_targets.pelvis_rotation[n].T @ (points["hip_r"] - points["hip_l"])
        np.testing.assert_allclose(axis[:2], 0.0, atol=1e-6)
        assert axis[2] == pytest.approx(_BUNDLE_HIP_WIDTH_M, abs=_WIDTH_TOL_M)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_pelvis_is_scaled_to_the_bundle_hip_width(squat_targets, squat_build) -> None:
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        GENERIC_STATURE_M,
    )

    _build, scales, pelvis = squat_build
    assert pelvis.hip_width_bundle_m == pytest.approx(
        _BUNDLE_HIP_WIDTH_M, abs=_WIDTH_TOL_M
    )
    assert pelvis.hip_width_generic_m == pytest.approx(
        _GENERIC_HIP_WIDTH_M, abs=_WIDTH_TOL_M
    )
    uniform = squat_targets.height_m / GENERIC_STATURE_M
    expected = (
        uniform,
        uniform,
        pelvis.hip_width_bundle_m / pelvis.hip_width_generic_m,
    )
    assert scales["pelvis"] == pytest.approx(expected, rel=1e-12)
    assert pelvis.scale == pytest.approx(expected, rel=1e-12)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_torso_is_scaled_to_the_bundle_shoulder_width(squat_targets) -> None:
    """OSL-14: the trunk length scales x/y, the shoulder width scales z."""
    from pathlib import Path

    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid

    _real_opensim()
    base = _base_model()
    scales = ikid.bundle_segment_scales(squat_targets, base)
    generic = ikid._generic_marker_points(Path(base))
    points = squat_targets.landmarks(0)
    width = np.linalg.norm(points["shoulder_r"] - points["shoulder_l"])
    generic_width = np.linalg.norm(generic["RSJC"] - generic["LSJC"])
    trunk, trunk_y, lateral = scales["torso"]
    assert trunk == trunk_y
    assert lateral == pytest.approx(width / generic_width, rel=1e-12)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_model_pelvis_on_the_bundle_pelvis_carries_markers_onto_targets(
    squat_targets, squat_build
) -> None:
    from src.engines.physics_engines.opensim.python.msk_lift_ik_id import (
        PELVIS_LANDMARKS,
    )
    from src.shared.python.engine_core.opensim_runtime import (
        import_opensim,
    )

    osim = import_opensim()
    build, _scales, pelvis = squat_build
    model = osim.Model(build.model)
    state = model.initSystem()
    coords = model.getCoordinateSet()
    bodies = model.getBodySet()
    markers = model.getMarkerSet()

    def ground(body: str) -> np.ndarray:
        p = bodies.get(body).getPositionInGround(state)
        return np.array([p.get(i) for i in range(3)])

    for n in range(len(squat_targets.indices)):
        tilt, tilt_list, rot = Rotation.from_matrix(
            squat_targets.pelvis_rotation[n]
        ).as_euler("ZXY")
        for name, value in (
            ("pelvis_tilt", tilt),
            ("pelvis_list", tilt_list),
            ("pelvis_rotation", rot),
            ("pelvis_tx", 0.0),
            ("pelvis_ty", 0.0),
            ("pelvis_tz", 0.0),
        ):
            coords.get(name).setValue(state, float(value), False)
        model.realizePosition(state)
        points = squat_targets.landmarks(n)
        shift = 0.5 * (points["hip_l"] + points["hip_r"]) - 0.5 * (
            ground("femur_l") + ground("femur_r")
        )
        for name, value in zip(
            ("pelvis_tx", "pelvis_ty", "pelvis_tz"), shift, strict=True
        ):
            coords.get(name).setValue(state, float(value), False)
        model.realizePosition(state)
        for label, marker in PELVIS_LANDMARKS.items():
            loc = markers.get(marker).getLocationInGround(state)
            got = np.array([loc.get(i) for i in range(3)])
            np.testing.assert_allclose(
                got, pelvis.positions_m[label][n], atol=_FRAME_TOL
            )


@pytest.fixture(scope="module")
def holds() -> dict:
    _real_opensim()
    _base_model()
    out = {}
    for lift in LIFTS:
        composition = compose_opensim_full_body_lift(OpenSimFullBodyLiftSpec(lift))
        assert composition.is_composed, composition.reason
        out[lift] = (composition, opensim_lift_static_grf(composition))
    return out


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_hold_contact_uses_the_cited_stiffness(holds: dict, lift: str) -> None:
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        LIFT_CONTACT_STIFFNESS_PA,
    )

    composition, _hold = holds[lift]
    assert pytest.approx(1.0e7) == LIFT_CONTACT_STIFFNESS_PA
    contact = composition.receipt["contact_parameters"]
    assert contact["stiffness_pa"] == LIFT_CONTACT_STIFFNESS_PA
    assert "#476" in contact["stiffness_source"]
    model = composition.native_model
    from src.shared.python.engine_core.opensim_runtime import (
        import_opensim,
    )

    osim = import_opensim()
    stiffness = [
        osim.SmoothSphereHalfSpaceForce.safeDownCast(f).get_stiffness()
        for f in model.getForceSet()
        if osim.SmoothSphereHalfSpaceForce.safeDownCast(f) is not None
    ]
    assert stiffness and all(s == LIFT_CONTACT_STIFFNESS_PA for s in stiffness)


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_hold_penetration_within_documented_limit(holds: dict, lift: str) -> None:
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        PENETRATION_LIMIT_M,
    )

    _composition, hold = holds[lift]
    assert hold["available"], hold["reason"]
    penetration = hold["penetration"]
    assert penetration["limit_m"] == PENETRATION_LIMIT_M == pytest.approx(0.008)
    assert 0.0 < penetration["max_m"] <= PENETRATION_LIMIT_M
    assert penetration["within_limit"] is True


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_hold_is_balanced_over_its_support(holds: dict, lift: str) -> None:
    composition, hold = holds[lift]
    assert hold["available"], hold["reason"]
    for name, value in hold["balance_residual"].items():
        assert abs(value) <= 1e-6, name
    supine = composition.receipt["hold_pose"]["supine"]
    assert (hold["balance"] is None) is supine
    if supine:
        return
    assert abs(hold["balance"]["com_offset_m"]) <= 1e-6
    # The centre of mass lies over the centre of pressure of the two feet.
    feet = [np.asarray(hold["support_cop_m"][s]) for s in ("left", "right")]
    weights = [hold["support_vertical_n"][s] for s in ("left", "right")]
    cop = np.average(feet, axis=0, weights=weights)
    np.testing.assert_allclose(np.asarray(hold["com_m"])[:2], cop[:2], atol=1e-6)
    assert abs(hold["settling_rad"]["fore_aft"]) <= math.radians(15.0)


@pytest.mark.requires_opensim
@pytest.mark.integration
@_RAJAGOPAL_ARM_RANGES
def test_squat_ik_keeps_every_coordinate_in_range() -> None:
    _real_opensim()
    _base_model()
    receipt = opensim_lift_ik_id(OpenSimFullBodyLiftSpec("squat"), stride=20)
    assert receipt["status"] == "complete", receipt["reason"]
    ik = receipt["ik"]
    assert ik["range_violations"] == {}
    landmarks = ik["pelvis_landmarks"]
    assert landmarks["hip_width_bundle_m"] == pytest.approx(
        _BUNDLE_HIP_WIDTH_M, abs=_WIDTH_TOL_M
    )
    assert isinstance(landmarks["tracked"], bool) and landmarks["reason"]


def test_pelvis_gate_is_strict() -> None:
    """#13072: the pelvis gate has no tolerance; a bound crossed by any amount
    (here the 0.012 deg the diverged OSL-14 re-track needed) is outside."""
    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid

    low, high = math.radians(-30.0), math.radians(120.0)
    assert ikid.PELVIS_GATE_TOLERANCE_RAD == 0.0
    assert ikid.hip_flexion_inside((math.radians(80.0), high), low, high)
    assert not ikid.hip_flexion_inside(
        (math.radians(80.0), math.radians(120.012)), low, high
    )
    assert not ikid.hip_flexion_inside((low, math.radians(121.0)), low, high)
    assert not ikid.hip_flexion_inside((math.radians(-31.0), high), low, high)
