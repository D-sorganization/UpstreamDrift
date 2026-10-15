"""Unit tests for bare-club grip wrench extraction and interface labelling (NSX-01, #12550)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.club_solids_partition import (
    bare_club_kinematics_from_body_frame,
    club_solids_partition,
)
from src.shared.python.biomechanics.grip_extraction import (
    closure_and_club_analysis,
    holding_hand_wrench,
    net_only_analysis,
)
from src.shared.python.biomechanics.grip_wrench import (
    DEFAULT_GRIP_INTERFACE,
    GRIP_INTERFACES,
    LEGACY_COMPOSITE_GRIP_INTERFACE,
    HandWrench,
    analyze_grip,
)

pytestmark = pytest.mark.unit

MODELS_DIR = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
DRIVER_SPEC_PATH = MODELS_DIR / "full_body_spec_anthro_driver.json"


@pytest.fixture(scope="module")
def driver_spec_bytes() -> bytes:
    return DRIVER_SPEC_PATH.read_bytes()


@pytest.fixture(scope="module")
def driver_spec(driver_spec_bytes: bytes) -> dict:
    return json.loads(driver_spec_bytes)


def test_grip_interface_labels_and_default_contract() -> None:
    assert DEFAULT_GRIP_INTERFACE == "hand_on_bare_club"
    assert LEGACY_COMPOSITE_GRIP_INTERFACE == "wrist_on_hands_plus_club"
    assert set(GRIP_INTERFACES) == {
        "hand_on_bare_club",
        "wrist_on_hands_plus_club",
    }

    left = HandWrench("L", (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
    right = HandWrench("R", (0.0, 0.0, 0.9), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    default_analysis = analyze_grip(left, right, split_method="logged")
    assert default_analysis.interface == "hand_on_bare_club"

    legacy_analysis = analyze_grip(
        left,
        right,
        split_method="logged",
        interface="wrist_on_hands_plus_club",
    )
    assert legacy_analysis.interface == "wrist_on_hands_plus_club"

    with pytest.raises(ValueError, match="interface"):
        analyze_grip(left, right, split_method="logged", interface="invalid_interface")


def test_bare_club_wrench_excludes_welded_hand_inertia(driver_spec: dict) -> None:
    partition = club_solids_partition(driver_spec)
    gravity = np.asarray(driver_spec["gravity_m_s2"], dtype=np.float64)
    p_b = np.array([0.1, -0.2, 1.2], dtype=np.float64)
    rot_wb = np.eye(3, dtype=np.float64)
    v_b = np.array([0.5, -0.3, 0.1], dtype=np.float64)
    omega = np.array([1.5, -2.5, 3.0], dtype=np.float64)
    a_b = np.array([12.0, -8.0, 4.0], dtype=np.float64)
    alpha = np.array([-15.0, 20.0, -10.0], dtype=np.float64)

    bare_club_kw = bare_club_kinematics_from_body_frame(
        partition.bare_club_inertia,
        body_position_world_m=p_b,
        body_rotation_world=rot_wb,
        body_linear_velocity_world_m_s=v_b,
        body_angular_velocity_world_rad_s=omega,
        body_linear_acceleration_world_m_s2=a_b,
        body_angular_acceleration_world_rad_s2=alpha,
        gravity_m_s2=gravity,
    )
    composite_club_kw = bare_club_kinematics_from_body_frame(
        partition.composite_inertia,
        body_position_world_m=p_b,
        body_rotation_world=rot_wb,
        body_linear_velocity_world_m_s=v_b,
        body_angular_velocity_world_rad_s=omega,
        body_linear_acceleration_world_m_s2=a_b,
        body_angular_acceleration_world_rad_s2=alpha,
        gravity_m_s2=gravity,
    )

    closing_point = p_b + np.array([0.0, -1.0478, 0.0635])
    holding_point = p_b + np.array([-0.025, -1.124, 0.059])
    closing_force = np.array([4.0, -3.0, 5.0])
    closing_torque = np.array([0.2, -0.1, 0.3])

    bare_analysis = closure_and_club_analysis(
        closing_side="R",
        closing_point_m=closing_point,
        closing_force_n=closing_force,
        closing_torque_nm=closing_torque,
        holding_point_m=holding_point,
        club=bare_club_kw,
        split_method="constraint_multiplier",
    )
    legacy_analysis = closure_and_club_analysis(
        closing_side="R",
        closing_point_m=closing_point,
        closing_force_n=closing_force,
        closing_torque_nm=closing_torque,
        holding_point_m=holding_point,
        club=composite_club_kw,
        split_method="constraint_multiplier",
        interface="wrist_on_hands_plus_club",
    )

    assert bare_analysis.interface == "hand_on_bare_club"
    assert legacy_analysis.interface == "wrist_on_hands_plus_club"

    expected_bare_net_f = 0.313 * (bare_club_kw["com_acceleration_m_s2"] - gravity)
    expected_legacy_net_f = partition.composite_inertia.mass_kg * (
        composite_club_kw["com_acceleration_m_s2"] - gravity
    )
    np.testing.assert_allclose(
        bare_analysis.net_force_n, expected_bare_net_f, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        legacy_analysis.net_force_n, expected_legacy_net_f, rtol=0, atol=1e-12
    )
    # Bare club force magnitude is dramatically smaller than composite (excludes 1.1082 kg hands)
    assert np.linalg.norm(bare_analysis.net_force_n) < 0.35 * np.linalg.norm(
        legacy_analysis.net_force_n
    )


def test_cross_engine_conventions_match_on_bare_club_wrench(
    driver_spec: dict,
) -> None:
    """MuJoCo, Drake, and Pinocchio multiplier conventions yield identical bare-club wrenches."""
    partition = club_solids_partition(driver_spec)
    gravity = np.asarray(driver_spec["gravity_m_s2"], dtype=np.float64)
    theta = 0.35
    rot_wb = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=np.float64,
    )
    p_b = np.array([0.15, -0.25, 1.05], dtype=np.float64)
    v_b = np.array([0.8, -0.4, 0.2], dtype=np.float64)
    omega = np.array([3.0, -4.0, 2.0], dtype=np.float64)
    a_b = np.array([25.0, -15.0, 10.0], dtype=np.float64)
    alpha = np.array([-30.0, 45.0, -20.0], dtype=np.float64)

    bare_kw = bare_club_kinematics_from_body_frame(
        partition,
        body_position_world_m=p_b,
        body_rotation_world=rot_wb,
        body_linear_velocity_world_m_s=v_b,
        body_angular_velocity_world_rad_s=omega,
        body_linear_acceleration_world_m_s2=a_b,
        body_angular_acceleration_world_rad_s2=alpha,
        gravity_m_s2=gravity,
    )

    placement_b = np.asarray(driver_spec["closure"]["placement_b"], dtype=np.float64)
    rot_wa = rot_wb @ placement_b[:3, :3]
    point_b = p_b + rot_wb @ placement_b[:3, 3]
    club_joint = next(
        j
        for j in driver_spec["joints"]
        if j["child"] == driver_spec["closure"]["body_b"]
    )
    holding_point = (
        p_b
        + rot_wb @ np.asarray(club_joint["child_to_follower"], dtype=np.float64)[:3, 3]
    )

    # True closing wrench on club in world frame
    f_close_w = np.array([18.0, -12.0, 24.0], dtype=np.float64)
    t_close_w = np.array([1.4, -0.9, 0.6], dtype=np.float64)

    # 1. MuJoCo convention: multiplier is -wrench_on_club in world frame [force; torque]
    lam_mj = np.concatenate([-f_close_w, -t_close_w])
    mj_analysis = closure_and_club_analysis(
        closing_side="R",
        closing_point_m=point_b,
        closing_force_n=-lam_mj[:3],
        closing_torque_nm=-lam_mj[3:],
        holding_point_m=holding_point,
        club=bare_kw,
        split_method="constraint_multiplier",
    )

    # 2. Drake convention: multiplier is +wrench_on_club in frame a [torque; force]
    lam_dr = np.concatenate([rot_wa.T @ t_close_w, rot_wa.T @ f_close_w])
    dr_analysis = closure_and_club_analysis(
        closing_side="R",
        closing_point_m=point_b,
        closing_force_n=rot_wa @ lam_dr[3:],
        closing_torque_nm=rot_wa @ lam_dr[:3],
        holding_point_m=holding_point,
        club=bare_kw,
        split_method="constraint_multiplier",
    )

    # 3. Pinocchio convention: lambda_grip is load on human in frame a [force; torque]
    lam_pin = np.concatenate([-rot_wa.T @ f_close_w, -rot_wa.T @ t_close_w])
    pin_analysis = closure_and_club_analysis(
        closing_side="R",
        closing_point_m=point_b,
        closing_force_n=-rot_wa @ lam_pin[:3],
        closing_torque_nm=-rot_wa @ lam_pin[3:],
        holding_point_m=holding_point,
        club=bare_kw,
        split_method="allocation",
    )

    for other in (dr_analysis, pin_analysis):
        assert other.interface == "hand_on_bare_club"
        np.testing.assert_allclose(
            other.net_force_n, mj_analysis.net_force_n, rtol=1e-6, atol=1e-9
        )
        np.testing.assert_allclose(
            other.couple_at_midpoint_nm,
            mj_analysis.couple_at_midpoint_nm,
            rtol=1e-6,
            atol=1e-9,
        )


@pytest.mark.requires_mujoco
def test_mujoco_full_body_model_uses_bare_club_and_preserves_legacy_interface(
    driver_spec_bytes: bytes,
) -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    model = NativeMujocoFullBodyModel(driver_spec_bytes)
    model.kkt_regularization = 0.0
    assert model.club_partition is not None
    assert model.club_partition.bare_club_inertia.mass_kg == pytest.approx(
        0.313, abs=1e-12
    )

    zero_coords = dict.fromkeys(model.coordinate_order, 0.0)
    zero_rates = dict.fromkeys(model.coordinate_order, 0.0)
    zero_efforts = dict.fromkeys(model.coordinate_order, 0.0)

    bare = model.grip_analysis(zero_coords, zero_rates, zero_efforts)
    legacy = model.grip_analysis(
        zero_coords,
        zero_rates,
        zero_efforts,
        interface="wrist_on_hands_plus_club",
    )

    assert bare.interface == "hand_on_bare_club"
    assert legacy.interface == "wrist_on_hands_plus_club"

    _, lam = model.solve_with_multipliers(zero_coords, zero_rates, zero_efforts)
    qacc = np.array(
        [
            model.accelerations(zero_coords, zero_rates, zero_efforts)[n]
            for n in model.coordinate_order
        ]
    )
    # Reorder into model dof order
    qacc_dof = np.zeros(model.model.nv)
    for name, idx in model._indices.items():
        qacc_dof[idx] = qacc[model.coordinate_order.index(name)]

    ne_bare = model._club_newton_euler(qacc_dof, interface="hand_on_bare_club")
    ne_legacy = model._club_newton_euler(qacc_dof, interface="wrist_on_hands_plus_club")
    assert ne_bare["mass_kg"] == pytest.approx(0.313, abs=1e-12)
    assert ne_legacy["mass_kg"] == pytest.approx(1.4212715706945233, abs=1e-9)

    expected_bare_net = ne_bare["mass_kg"] * (
        ne_bare["com_acceleration_m_s2"] - ne_bare["gravity_m_s2"]
    )
    np.testing.assert_allclose(
        bare.net_force_n, expected_bare_net, rtol=1e-6, atol=1e-9
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_drake_club_newton_euler_uses_bare_club_partition(
    driver_spec: dict,
) -> None:
    from src.engines.physics_engines.drake.python.full_body_model import (
        club_newton_euler,
    )

    partition = club_solids_partition(driver_spec)
    gravity = np.asarray(driver_spec["gravity_m_s2"], dtype=float)
    rot_wb = _rotation_z(-0.35)
    origin_b = np.array([0.15, -0.22, 1.05])
    v_origin = np.array([0.5, -1.2, 0.3])
    a_origin = np.array([3.2, -4.5, 6.1])
    omega = np.array([1.4, -2.3, 4.8])
    alpha = np.array([12.0, -18.0, 25.0])

    class _FakeRot:
        def matrix(self) -> np.ndarray:
            return rot_wb

    class _FakePose:
        def rotation(self) -> _FakeRot:
            return _FakeRot()

        def translation(self) -> np.ndarray:
            return origin_b

    class _FakeSpatialVec:
        def __init__(self, rot: np.ndarray, trans: np.ndarray) -> None:
            self._rot = rot
            self._trans = trans

        def rotational(self) -> np.ndarray:
            return self._rot

        def translational(self) -> np.ndarray:
            return self._trans

    class _FakeRotInertia:
        def __init__(self, mat: np.ndarray) -> None:
            self._mat = mat

        def ShiftToCenterOfMass(
            self, _mass: float, _com: np.ndarray
        ) -> _FakeRotInertia:
            return self

        def CopyToFullMatrix3(self) -> np.ndarray:
            return self._mat

    class _FakeSpatialInertia:
        def get_mass(self) -> float:
            return partition.composite_inertia.mass_kg

        def get_com(self) -> np.ndarray:
            return np.asarray(partition.composite_inertia.com_body_m, dtype=float)

        def CalcRotationalInertia(self) -> _FakeRotInertia:
            return _FakeRotInertia(
                np.asarray(
                    partition.composite_inertia.inertia_com_body_kg_m2, dtype=float
                )
            )

    class _FakeBody:
        def index(self) -> int:
            return 0

        def body_frame(self) -> str:
            return "club_frame"

    class _FakePlant:
        def EvalBodyPoseInWorld(self, _ctx: object, _body: object) -> _FakePose:
            return _FakePose()

        def EvalBodySpatialVelocityInWorld(
            self, _ctx: object, _body: object
        ) -> _FakeSpatialVec:
            return _FakeSpatialVec(omega, v_origin)

        def CalcSpatialAccelerationsFromVdot(
            self, _ctx: object, _vdot: np.ndarray
        ) -> list[_FakeSpatialVec]:
            return [_FakeSpatialVec(alpha, a_origin)]

        def GetBodiesWeldedTo(self, _body: object) -> list[_FakeBody]:
            return [_FakeBody()]

        def CalcSpatialInertia(
            self, _ctx: object, _frame: object, _welded: object
        ) -> _FakeSpatialInertia:
            return _FakeSpatialInertia()

    plant = _FakePlant()
    body = _FakeBody()
    ne_bare = club_newton_euler(
        plant,
        None,
        body,
        np.zeros(6),
        gravity,
        partition=partition,
    )
    ne_composite = club_newton_euler(
        plant,
        None,
        body,
        np.zeros(6),
        gravity,
        partition=None,
    )

    assert ne_bare["mass_kg"] == pytest.approx(0.313, abs=1e-12)
    assert ne_composite["mass_kg"] == pytest.approx(1.4212715706945233, abs=1e-9)
    assert not np.allclose(ne_bare["com_m"], ne_composite["com_m"])


def test_pinocchio_force_adapter_uses_bare_club_and_preserves_legacy_interface(
    driver_spec: dict,
) -> None:
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.biomechanics.grip_extraction import (
        closing_side_from_closure,
    )

    partition = club_solids_partition(driver_spec)
    gravity = np.asarray(driver_spec["gravity_m_s2"], dtype=float)
    rot_wb = _rotation_z(0.25)
    origin_b = np.array([0.1, -0.2, 0.95])

    class _FakeSE3:
        def __init__(self, rot: np.ndarray, trans: np.ndarray) -> None:
            self.rotation = rot
            self.translation = trans

        def __mul__(self, other: _FakeSE3) -> _FakeSE3:
            return _FakeSE3(
                self.rotation @ other.rotation,
                self.translation + self.rotation @ other.translation,
            )

        def actInv(self, motion: object) -> object:
            return motion

    class _FakeMotion:
        def __init__(self, linear: np.ndarray, angular: np.ndarray) -> None:
            self.linear = linear
            self.angular = angular

    class _FakePin:
        def forwardKinematics(
            self, _m: object, _d: object, _q: object, _v: object, _a: object
        ) -> None:
            return None

    class _FakeData:
        def __init__(self) -> None:
            self.oMi = [None, _FakeSE3(rot_wb, origin_b)]
            self.v = [None, _FakeMotion(np.zeros(3), np.zeros(3))]
            self.a = [None, _FakeMotion(np.zeros(3), np.zeros(3))]

    class _FakePlant:
        def body_placement(self, _name: str) -> tuple[int, _FakeSE3]:
            return 1, _FakeSE3(np.eye(3), np.zeros(3))

        def closure_frame_pose(
            self, _coords: dict[str, float]
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.eye(3), origin_b + np.array([0.0, 0.0, -0.05])

    adapter = object.__new__(PinocchioForceAdapter)
    adapter._plant = _FakePlant()
    adapter._pin = _FakePin()
    adapter._model = object()
    adapter._data = _FakeData()
    adapter._names = ("root_x", "root_y", "root_z")
    adapter.closing_hand_side = closing_side_from_closure(driver_spec["closure"])
    adapter._club_body_name = str(driver_spec["closure"]["body_b"])
    adapter._gravity = gravity
    adapter.club_partition = partition

    q = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)
    lam = np.array([1.0, -2.0, 3.0, 0.1, -0.1, 0.2])

    net_only = adapter.grip_analysis_from_allocation(q, lam)
    assert net_only.interface == "hand_on_bare_club"
    assert net_only.left is None and net_only.right is None

    bare = adapter.grip_analysis_from_allocation(q, lam, v=v, a=a)
    legacy = adapter.grip_analysis_from_allocation(
        q, lam, v=v, a=a, interface="wrist_on_hands_plus_club"
    )
    assert bare.interface == "hand_on_bare_club"
    assert legacy.interface == "wrist_on_hands_plus_club"
    assert bare.left is not None and bare.right is not None
    np.testing.assert_allclose(
        bare.net_force_n,
        0.313 * np.array([0.0, 0.0, 9.80665]),
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        legacy.net_force_n,
        1.4212715706945233 * np.array([0.0, 0.0, 9.80665]),
        rtol=1e-6,
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="Both v and a"):
        adapter.grip_analysis_from_allocation(q, lam, v=v)
