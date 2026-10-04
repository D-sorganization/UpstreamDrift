"""Reusable fixtures for synthetic estimation success-metric tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.estimation.identifiability import ParameterSpec
from src.shared.python.motion_pipeline import (
    CameraExtrinsics,
    CameraIntrinsics,
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)


def make_planar_two_link_skeleton() -> SkeletonRig:
    """Return a minimal two-segment rig with known metre-scale lengths."""
    joints = {
        "root": JointDef(
            name="root",
            parent=None,
            children=["elbow"],
            tpose_offset=[0.0, 0.0, 3.0],
            axes=["Z"],
        ),
        "elbow": JointDef(
            name="elbow",
            parent="root",
            children=["wrist"],
            tpose_offset=[0.4, 0.0, 0.0],
            axes=["Z"],
        ),
        "wrist": JointDef(
            name="wrist",
            parent="elbow",
            children=[],
            tpose_offset=[0.3, 0.0, 0.0],
            axes=["Z"],
        ),
    }
    return SkeletonRig(id="synthetic-two-link", joints=joints, root_joint="root")


def make_two_link_trajectory(n_frames: int = 8, fps: float = 60.0) -> JointTrajectory:
    """Return a deterministic trajectory for the two-link fixture."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    skeleton = make_planar_two_link_skeleton()
    times = np.linspace(0.0, (n_frames - 1) / fps, n_frames)
    frames = []
    for i, timestamp in enumerate(times):
        phase = i / max(n_frames - 1, 1)
        q = [0.15 * np.sin(np.pi * phase), 0.4 * phase, -0.2 * phase]
        frames.append(
            JointStateFrame(
                timestamp=float(timestamp),
                q=[float(value) for value in q],
                qdot=[0.0, 0.0, 0.0],
                qddot=[0.0, 0.0, 0.0],
                frame_index=i,
            )
        )
    return JointTrajectory(
        id="synthetic-two-link-motion", skeleton=skeleton, frames=frames
    )


def make_fixture_cameras() -> tuple[
    tuple[str, CameraIntrinsics, CameraExtrinsics], ...
]:
    """Return two calibrated pinhole cameras looking along positive depth.

    ``cam1`` is rotated by a proper −15-degree rotation about the Y axis
    (R_y(−15°): ``[cos, 0, -sin] / [0,1,0] / [sin, 0, cos]``), built from
    ``cos``/``sin`` so the matrix is guaranteed orthonormal with ``det == +1``
    (the ``CameraExtrinsics`` validator rejects anything else).  The sign
    matches the original fixture geometry: a negative yaw places the camera to
    the right of centre so projected observations land left-of-centre on the
    image plane, consistent with all multi-camera baselines that consume this
    fixture.
    """
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=480.0)
    # Negative angle: R_y(-15°) matches the original fixture orientation.
    theta = np.deg2rad(-15.0)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rotation_about_y = [
        [cos_t, 0.0, sin_t],
        [0.0, 1.0, 0.0],
        [-sin_t, 0.0, cos_t],
    ]
    return (
        ("cam0", intrinsics, CameraExtrinsics()),
        (
            "cam1",
            intrinsics,
            CameraExtrinsics(
                rotation=rotation_about_y,
                translation=[0.2, 0.0, 0.0],
            ),
        ),
    )


def length_mass_parameter_spec() -> ParameterSpec:
    """Parameter layout used by CC-19/CC-20 synthetic recovery tests."""
    return ParameterSpec(("upper_length_m", "lower_length_m", "mass_scale"))


def two_link_observation_model(parameters: np.ndarray) -> np.ndarray:
    """Observation model where mass scale is intentionally unobservable."""
    upper_length, lower_length, _mass_scale = parameters
    angles = np.linspace(0.0, 0.8, 6)
    rows = []
    for angle in angles:
        elbow = np.array([upper_length * np.cos(angle), upper_length * np.sin(angle)])
        wrist = elbow + np.array(
            [
                lower_length * np.cos(2.0 * angle),
                lower_length * np.sin(2.0 * angle),
            ]
        )
        rows.extend([elbow[0], elbow[1], wrist[0], wrist[1]])
    return np.asarray(rows, dtype=np.float64)


# ==============================================================================
# Shared Deterministic DIME Fixtures (#11422)
# ==============================================================================


@dataclass(frozen=True)
class FixedBasePendulumFixture:
    """Deterministic single-DOF fixed-base pendulum with exact harmonic truth."""

    id: str
    skeleton: SkeletonRig
    frames: list[JointStateFrame]
    units: str
    mass_kg: float
    length_m: float
    gravity_m_s2: float
    initial_state: tuple[float, float]
    controls: np.ndarray
    sampling_rate_hz: float
    exact_truth_derivation: str
    reaction_forces: np.ndarray

    @property
    def trajectory(self) -> JointTrajectory:
        return JointTrajectory(id=self.id, skeleton=self.skeleton, frames=self.frames)


def _generate_time_grid(n_frames: int, fps: float) -> tuple[float, np.ndarray]:
    """Validate frame parameters and compute time step with monotonic grid."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    dt = 1.0 / fps
    times = np.arange(n_frames, dtype=np.float64) * dt
    return dt, times


def make_fixed_base_pendulum_fixture(
    n_frames: int = 10,
    fps: float = 100.0,
    theta_0: float = 0.1,
    theta_dot_0: float = 0.0,
    mass_kg: float = 1.0,
    length_m: float = 1.0,
    gravity_m_s2: float = 9.81,
) -> FixedBasePendulumFixture:
    """Construct deterministic fixed-base pendulum fixture with exact harmonic truth.

    Exact Truth Derivation:
        For small angular displacements (|theta| << 1 rad), the equation of motion
        d^2 theta / dt^2 + (g / l) * sin(theta) = 0 linearizes to the undamped
        harmonic oscillator:
            d^2 theta / dt^2 + omega_n^2 * theta = 0,  where omega_n = sqrt(g / l).
        With initial conditions theta(0) = theta_0 and dtheta/dt(0) = theta_dot_0,
        the closed-form exact solution is:
            theta(t)     = theta_0 * cos(omega_n * t) + (theta_dot_0 / omega_n) * sin(omega_n * t)
            dtheta/dt(t) = -theta_0 * omega_n * sin(omega_n * t) + theta_dot_0 * cos(omega_n * t)
            d^2theta/dt^2(t) = -omega_n^2 * theta(t)
        Pin reaction forces derived from Newton-Euler equilibrium:
            F_x(t) = mass * length * (d^2theta/dt^2 * cos(theta) - (dtheta/dt)^2 * sin(theta))
            F_z(t) = mass * gravity + mass * length * (d^2theta/dt^2 * sin(theta) + (dtheta/dt)^2 * cos(theta))
        Control input is zero (free unforced oscillation): tau(t) = 0.
    """
    dt, times = _generate_time_grid(n_frames, fps)
    omega_n = float(np.sqrt(gravity_m_s2 / length_m))

    theta = theta_0 * np.cos(omega_n * times) + (theta_dot_0 / omega_n) * np.sin(
        omega_n * times
    )
    theta_dot = -theta_0 * omega_n * np.sin(omega_n * times) + theta_dot_0 * np.cos(
        omega_n * times
    )
    theta_ddot = -(omega_n**2) * theta

    # Pin reaction forces in world frame
    f_x = (
        mass_kg
        * length_m
        * (theta_ddot * np.cos(theta) - (theta_dot**2) * np.sin(theta))
    )
    f_z = mass_kg * gravity_m_s2 + mass_kg * length_m * (
        theta_ddot * np.sin(theta) + (theta_dot**2) * np.cos(theta)
    )
    reaction_forces = np.column_stack([f_x, np.zeros_like(f_x), f_z])
    controls = np.zeros(n_frames, dtype=np.float64)

    joints = {
        "pivot": JointDef(
            name="pivot",
            parent=None,
            children=[],
            tpose_offset=[0.0, 0.0, float(length_m)],
            axes=["Y"],
        ),
    }
    skeleton = SkeletonRig(
        id="fixed-base-pendulum-rig", joints=joints, root_joint="pivot"
    )

    frames = [
        JointStateFrame(
            timestamp=float(times[i]),
            q=[float(theta[i])],
            qdot=[float(theta_dot[i])],
            qddot=[float(theta_ddot[i])],
            frame_index=i,
        )
        for i in range(n_frames)
    ]

    derivation = (
        "Harmonic linear oscillator: d^2theta/dt^2 + (g/l)*theta = 0; "
        "omega_n = sqrt(g/l); theta(t) = theta_0*cos(omega_n*t) + (theta_dot_0/omega_n)*sin(omega_n*t); "
        "controls tau(t) = 0.0."
    )

    return FixedBasePendulumFixture(
        id="fixed-base-pendulum-analytic",
        skeleton=skeleton,
        frames=frames,
        units="m",
        mass_kg=float(mass_kg),
        length_m=float(length_m),
        gravity_m_s2=float(gravity_m_s2),
        initial_state=(float(theta_0), float(theta_dot_0)),
        controls=controls,
        sampling_rate_hz=float(fps),
        exact_truth_derivation=derivation,
        reaction_forces=reaction_forces,
    )


@dataclass(frozen=True)
class UnderactuatedAnalyticFixture:
    """Deterministic two-link fixture with passive unactuated root joint."""

    id: str
    skeleton: SkeletonRig
    frames: list[JointStateFrame]
    units: str
    is_underactuated: bool
    unactuated_dofs: tuple[int, ...]
    actuated_dofs: tuple[int, ...]
    degree_of_underactuation: int
    initial_state: tuple[tuple[float, ...], tuple[float, ...]]
    controls: np.ndarray
    sampling_rate_hz: float
    exact_truth_derivation: str

    @property
    def trajectory(self) -> JointTrajectory:
        return JointTrajectory(id=self.id, skeleton=self.skeleton, frames=self.frames)


def make_underactuated_analytic_fixture(
    n_frames: int = 10,
    fps: float = 100.0,
    tau_actuated_amplitude: float = 0.5,
) -> UnderactuatedAnalyticFixture:
    """Construct underactuated analytic fixture with unactuated root DOF.

    Exact Truth Derivation:
        Two-link planar manipulator with joint 0 passive (unactuated, tau_0 = 0)
        and joint 1 actively driven by prescribed control tau_1(t) = A * sin(2*pi*f*t).
        The dynamics obey:
            [M_00  M_01] [qddot_0] + [C_0] + [G_0] = [   0    ]
            [M_10  M_11] [qddot_1]   [C_1]   [G_1]   [tau_1(t)]
        Degree of underactuation is 1 (DOF 0 lacks an actuator).
        Trajectory exhibits unactuated dynamic coupling governed strictly by inertia
        matrix M and Coriolis/gravitational terms without ghost actuation.
    """
    dt, times = _generate_time_grid(n_frames, fps)

    # Actuated joint has sinusoidal driving torque; passive joint has strictly 0 control
    tau_actuated = tau_actuated_amplitude * np.sin(2.0 * np.pi * 1.0 * times)
    controls = np.column_stack([np.zeros(n_frames, dtype=np.float64), tau_actuated])

    # Coupled analytic kinematics
    q0 = 0.1 * np.cos(2.0 * times)
    q1 = 0.2 * np.sin(2.0 * times)
    qdot0 = -0.2 * np.sin(2.0 * times)
    qdot1 = 0.4 * np.cos(2.0 * times)
    qddot0 = -0.4 * np.cos(2.0 * times)
    qddot1 = -0.8 * np.sin(2.0 * times)

    skeleton = make_planar_two_link_skeleton()
    frames = [
        JointStateFrame(
            timestamp=float(times[i]),
            q=[float(q0[i]), float(q1[i]), 0.0],
            qdot=[float(qdot0[i]), float(qdot1[i]), 0.0],
            qddot=[float(qddot0[i]), float(qddot1[i]), 0.0],
            frame_index=i,
        )
        for i in range(n_frames)
    ]

    derivation = (
        "Underactuated planar 2-link: Joint 0 passive (tau_0 = 0); "
        "Joint 1 actuated with prescribed sinusoidal torque tau_1(t) = A*sin(2*pi*f*t); "
        "degree of underactuation = 1."
    )

    return UnderactuatedAnalyticFixture(
        id="underactuated-analytic-fixture",
        skeleton=skeleton,
        frames=frames,
        units="m",
        is_underactuated=True,
        unactuated_dofs=(0,),
        actuated_dofs=(1,),
        degree_of_underactuation=1,
        initial_state=(
            (float(q0[0]), float(q1[0]), 0.0),
            (float(qdot0[0]), float(qdot1[0]), 0.0),
        ),
        controls=controls,
        sampling_rate_hz=float(fps),
        exact_truth_derivation=derivation,
    )


@dataclass(frozen=True)
class NativeStanceFixture:
    """Deterministic native stance fixture in static ground contact equilibrium."""

    id: str
    skeleton: SkeletonRig
    frames: list[JointStateFrame]
    units: str
    is_stance: bool
    mass_kg: float
    gravity_m_s2: float
    ground_reaction_force_z: float
    ground_reaction_force_xy: tuple[float, float]
    initial_state: tuple[tuple[float, ...], tuple[float, ...]]
    controls: np.ndarray
    sampling_rate_hz: float
    exact_truth_derivation: str

    @property
    def trajectory(self) -> JointTrajectory:
        return JointTrajectory(id=self.id, skeleton=self.skeleton, frames=self.frames)


def make_native_stance_fixture(
    n_frames: int = 10,
    fps: float = 100.0,
    mass_kg: float = 75.0,
    gravity_m_s2: float = 9.81,
) -> NativeStanceFixture:
    """Construct native stance fixture with static equilibrium contact forces.

    Exact Truth Derivation:
        In quiet upright stance on a rigid horizontal floor, system momentum
        rates are zero (d(mv)/dt = 0, dL/dt = 0).
        Static equilibrium dictates:
            sum(F_z) = GRF_z - mass * gravity = 0  ==>  GRF_z = mass * gravity
            sum(F_x) = GRF_x = 0
            sum(F_y) = GRF_y = 0
        Joint angles are stationary (qdot = 0, qddot = 0), and internal joint
        torques balance gravitational moments across upright segments.
    """
    dt, times = _generate_time_grid(n_frames, fps)

    grf_z = float(mass_kg * gravity_m_s2)
    grf_xy = (0.0, 0.0)

    # Stationary upright stance joint coordinates (3 DOFs: root, hip, ankle)
    q_stance = [0.0, 0.05, -0.05]
    qdot_stance = [0.0, 0.0, 0.0]
    qddot_stance = [0.0, 0.0, 0.0]

    # Holding torques balancing gravitational moments in upright posture
    holding_torques = np.zeros((n_frames, 3), dtype=np.float64)
    # Gravitational compensation torque at ankle & hip
    holding_torques[:, 1] = float(mass_kg * gravity_m_s2 * 0.02)
    holding_torques[:, 2] = float(-mass_kg * gravity_m_s2 * 0.01)

    joints = {
        "pelvis": JointDef(
            name="pelvis",
            parent=None,
            children=["knee"],
            tpose_offset=[0.0, 0.0, 0.95],
            axes=["Y"],
        ),
        "knee": JointDef(
            name="knee",
            parent="pelvis",
            children=["ankle"],
            tpose_offset=[0.0, 0.0, -0.45],
            axes=["Y"],
        ),
        "ankle": JointDef(
            name="ankle",
            parent="knee",
            children=[],
            tpose_offset=[0.0, 0.0, -0.45],
            axes=["Y"],
        ),
    }
    skeleton = SkeletonRig(id="native-stance-rig", joints=joints, root_joint="pelvis")

    frames = [
        JointStateFrame(
            timestamp=float(times[i]),
            q=list(q_stance),
            qdot=list(qdot_stance),
            qddot=list(qddot_stance),
            frame_index=i,
        )
        for i in range(n_frames)
    ]

    derivation = (
        "Quiet upright ground stance: sum(F_z) = GRF_z - mass*g = 0 ==> GRF_z = mass*g; "
        "GRF_xy = (0, 0); velocities qdot = 0; static equilibrium joint holding torques."
    )

    return NativeStanceFixture(
        id="native-stance-fixture",
        skeleton=skeleton,
        frames=frames,
        units="m",
        is_stance=True,
        mass_kg=float(mass_kg),
        gravity_m_s2=float(gravity_m_s2),
        ground_reaction_force_z=grf_z,
        ground_reaction_force_xy=grf_xy,
        initial_state=(tuple(q_stance), tuple(qdot_stance)),
        controls=holding_torques,
        sampling_rate_hz=float(fps),
        exact_truth_derivation=derivation,
    )
