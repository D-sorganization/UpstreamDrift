"""Barbell held at both hands through compliant bushings (LIFT-4, #11744).

The same-input full-body spec carries the golf dual-grip "club" welded to
both hands.  Here a separate rigid **barbell** body (six coordinates) is held
by one six-axis bushing per hand.  The bushings are the OSV-7 golf grip
(``grip_contact``): the same hand frames (:class:`GripInterface` from the
spec), the same OpenSim ``BushingForce`` law (:func:`bushing_wrench`), the same
stiffness defaults and the same modal damping design (:func:`design_damping`,
zeta = 0.7), evaluated for the barbell's mass and inertia instead of the
club's.  The coupling is two-way: each bushing pushes the barbell and pulls
the hand frame with the opposite wrench, mapped to the body through the
engine's own forward kinematics, so the bar load reaches the feet.

:class:`BarbellBushingPlant` wraps any same-input plant
(:class:`~src.shared.python.motion_matching.same_input.VectorPlant`) and
appends the barbell coordinates :data:`BAR_COORDINATES`: centre-of-mass
position (world, m) and body-fixed X-Y-Z angles (rad) of ``R0^T R`` where
``R0`` is the hand-frame orientation at the attachment pose.  The coordinate
rates are the time derivatives of those coordinates, so the shared RK4
integrator, closure projection and root-effort rules apply unchanged.

Barbell geometry (:class:`Barbell`) follows the IWF men's competition bar:
20 kg, 2.20 m, 28 mm shaft, 1.31 m between the inner collars, 450 mm plates
(International Weightlifting Federation, Technical and Competition Rules &
Regulations, section 2.2).  The plate centroid offset from the bar centre is
an engineering assumption (inner collar plus half a plate stack, 0.70 m).
The sleeves spin freely on the shaft, so the plates' axial inertia does not
couple to the shaft: the axial inertia is the shaft's alone.  The transverse
inertia includes the plates as thin discs.

Hand frame Jacobians are central differences of the engine's forward
kinematics of the spec ``Grip`` frame (on the club body, which the closure
welds to both hands), the same convention as the GCV-20 ball impact force.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    HandWrench,
    SplitMethod,
    analyze_grip,
)
from src.shared.python.grip_contact import (
    BushingWrench,
    ClubDynamics,
    GripFrame,
    GripInterface,
    RigidBodyState,
    bushing_wrench,
    default_bushing,
    design_damping,
    hand_frame_states,
)
from src.shared.python.grip_contact.bushing_law import (
    body_xyz_angles,
    body_xyz_n_matrix,
    cross3,
)

Array = NDArray[np.float64]

BAR_COORDINATES: tuple[str, ...] = (
    "BarbellX",
    "BarbellY",
    "BarbellZ",
    "BarbellRotX",
    "BarbellRotY",
    "BarbellRotZ",
)
BAR_FRAME = "Barbell"
#: Spec frame on the club body; locates the hand (club) body in every engine.
HAND_BODY_FRAME = "Grip"
SIDES = ("L", "R")
_FD_STEP = 1e-3
#: Fourth-order central-difference stencil ``(offset, weight / 12)``.
_STENCIL = ((-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0))
_SPLIT: SplitMethod = "bushing"

__all__ = [
    "BAR_COORDINATES",
    "BAR_FRAME",
    "Barbell",
    "BarbellBushingPlant",
    "BushingSample",
    "barbell_dynamics",
    "barbell_interface",
]


@dataclass(frozen=True)
class Barbell:
    """Loaded barbell: total mass and the geometry that sets its inertia.

    Preconditions: every value finite and positive; ``load_kg`` at least the
    shaft mass (an empty bar carries no plates).
    """

    load_kg: float
    shaft_mass_kg: float = 20.0
    shaft_length_m: float = 2.20
    shaft_radius_m: float = 0.014
    plate_offset_m: float = 0.70
    plate_radius_m: float = 0.225

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not (isinstance(value, int | float) and math.isfinite(value)):
                raise TypeError(f"{name} must be a finite number, got {value!r}")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.load_kg < self.shaft_mass_kg:
            raise ValueError(
                f"load_kg ({self.load_kg}) must be at least the shaft mass "
                f"({self.shaft_mass_kg})"
            )

    @property
    def plate_mass_kg(self) -> float:
        """Plate mass, split evenly between the two sleeves."""
        return self.load_kg - self.shaft_mass_kg

    @property
    def axial_inertia_kg_m2(self) -> float:
        """Inertia about the bar axis: the shaft only (sleeves spin freely)."""
        return 0.5 * self.shaft_mass_kg * self.shaft_radius_m**2

    @property
    def transverse_inertia_kg_m2(self) -> float:
        """Inertia about a transverse axis through the centre (rod + discs)."""
        rod = self.shaft_mass_kg * self.shaft_length_m**2 / 12.0
        discs = self.plate_mass_kg * (
            self.plate_offset_m**2 + 0.25 * self.plate_radius_m**2
        )
        return rod + discs


def barbell_dynamics(barbell: Barbell, interface: GripInterface) -> ClubDynamics:
    """The barbell as a rigid body in the hand (club) body frame.

    The bar axis is the grip frames' x axis (the line through both hands) and
    the centre of mass sits midway between the two grip points.
    """
    if not isinstance(barbell, Barbell):
        raise TypeError("barbell must be a Barbell")
    left, right = interface.left, interface.right
    com = 0.5 * (np.asarray(left.position_m) + np.asarray(right.position_m))
    axes = np.asarray(left.rotation, dtype=float)
    local = np.diag(
        [
            barbell.axial_inertia_kg_m2,
            barbell.transverse_inertia_kg_m2,
            barbell.transverse_inertia_kg_m2,
        ]
    )
    return ClubDynamics(
        float(barbell.load_kg),
        (float(com[0]), float(com[1]), float(com[2])),
        axes @ local @ axes.T,
    )


def barbell_interface(
    spec: Mapping[str, Any], barbell: Barbell, damping_ratio: float = 0.7
) -> GripInterface:
    """OSV-7 hand frames and bushing stiffness, damping designed for the bar."""
    frames = GripInterface.from_spec(spec, bushing=default_bushing())
    club = spec.get("closure", {}).get("body_b")
    lw = next(
        (
            f
            for f in spec.get("frames", [])
            if f.get("name") == "LW" and f.get("body") == club
        ),
        None,
    )
    if lw is not None:
        p_left = tuple(
            float(x) for x in np.asarray(lw["placement"], dtype=float)[:3, 3]
        )
        left = GripFrame("L", p_left, frames.left.rotation)  # type: ignore[arg-type]
        frames = GripInterface(
            left, frames.right, default_bushing(), frames.contact_material
        )
    bushing = design_damping(
        default_bushing(),
        frames.left,
        frames.right,
        barbell_dynamics(barbell, frames),
        damping_ratio,
    )
    return GripInterface(frames.left, frames.right, bushing, frames.contact_material)


class BodyPlant(Protocol):
    """What the wrapper needs from a same-input plant (``VectorPlant``)."""

    coordinate_order: tuple[str, ...]

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array: ...

    def closure_pose_residual(self, q: Array) -> Array: ...

    def closure_rate_matrix(self, q: Array) -> Array: ...

    def kinematic_frames(self, q: Array) -> dict[str, Array]: ...


@dataclass(frozen=True)
class BushingSample:
    """Both hand bushings at one state; wrenches act ON THE BAR (world axes)."""

    wrench: dict[str, BushingWrench]
    bar_grip_m: dict[str, Array]
    bar: RigidBodyState

    def grip_analysis(self, metadata: Mapping[str, Any] | None = None) -> GripAnalysis:
        """GCV-7 reduction (midpoint net force, couple), ``split_method='bushing'``."""
        hands: list[HandWrench] = []
        for side in SIDES:
            p = self.bar_grip_m[side]
            f = self.wrench[side].force_n
            m = self.wrench[side].moment_nm
            hands.append(
                HandWrench(
                    side,
                    (float(p[0]), float(p[1]), float(p[2])),
                    (float(f[0]), float(f[1]), float(f[2])),
                    (float(m[0]), float(m[1]), float(m[2])),
                )
            )
        return analyze_grip(
            hands[0],
            hands[1],
            split_method=_SPLIT,
            club_rotation=self.bar.rotation,
            metadata=metadata,
        )

    def max_deflection_m(self) -> float:
        """Largest translational bushing deflection of either hand."""
        return max(float(np.linalg.norm(self.wrench[s].translation_m)) for s in SIDES)


def _rotation_xyz(angles: Array) -> Array:
    """``Rx(t0) Ry(t1) Rz(t2)``."""
    c0, c1, c2 = (math.cos(float(a)) for a in angles)
    s0, s1, s2 = (math.sin(float(a)) for a in angles)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c0, -s0], [0.0, s0, c0]])
    ry = np.array([[c1, 0.0, s1], [0.0, 1.0, 0.0], [-s1, 0.0, c1]])
    rz = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def _n_matrix_rate(angles: Array, rates: Array) -> Array:
    """Time derivative of :func:`body_xyz_n_matrix` along ``rates``."""
    _, t1, t2 = (float(a) for a in angles)
    c1, s1, c2, s2 = math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2)
    d_t1 = np.array(
        [
            [c2 * s1 / c1**2, -s2 * s1 / c1**2, 0.0],
            [0.0, 0.0, 0.0],
            [-c2 / c1**2, s2 / c1**2, 0.0],
        ]
    )
    d_t2 = np.array(
        [
            [-s2 / c1, -c2 / c1, 0.0],
            [c2, -s2, 0.0],
            [s1 * s2 / c1, s1 * c2 / c1, 0.0],
        ]
    )
    return d_t1 * float(rates[1]) + d_t2 * float(rates[2])


def _vee_skew(matrix: Array) -> Array:
    skew = 0.5 * (matrix - matrix.T)
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


class BarbellBushingPlant:
    """``plant`` plus a barbell held at both hands by OSV-7 bushings.

    Args:
        plant: Same-input plant of the full-body spec (any engine).
        spec_bytes: The same spec document bytes.
        barbell: Bar mass and geometry.
        attach_q: Body pose at which the bar is attached with zero bushing
            deflection; fixes ``R0`` and :attr:`attached_bar_q`.
        interface: Hand frames and bushing; default :func:`barbell_interface`.

    Postconditions: ``coordinate_order`` is the plant's plus
    :data:`BAR_COORDINATES`; at ``(attach_q, attached_bar_q)`` with zero
    rates both bushings are undeflected.
    """

    def __init__(
        self,
        plant: BodyPlant,
        spec_bytes: bytes,
        barbell: Barbell,
        attach_q: Array,
        interface: GripInterface | None = None,
    ) -> None:
        if not isinstance(spec_bytes, bytes | bytearray):
            raise TypeError("spec_bytes must be the raw spec document bytes")
        spec = json.loads(spec_bytes)
        self.barbell = barbell
        self.interface = interface or barbell_interface(spec, barbell)
        self.dynamics = barbell_dynamics(barbell, self.interface)
        self._plant = plant
        self.body_order = tuple(plant.coordinate_order)
        self.nv_body = len(self.body_order)
        self.coordinate_order = (*self.body_order, *BAR_COORDINATES)
        self.gravity = np.asarray(spec["gravity_m_s2"], dtype=float)
        self._hand_to_frame = _frame_to_body(spec)
        com = np.asarray(self.dynamics.com_m, dtype=float)
        self._grip_offsets = {}
        for side in SIDES:
            offset = self.interface.frame(side).matrix()
            offset[:3, 3] -= com
            self._grip_offsets[side] = offset
        q_attach = self._body(np.asarray(attach_q, dtype=float))
        self._columns: list[int] = list(range(self.nv_body))
        hand_jac = self._hand_jacobian(q_attach)
        self._columns = np.flatnonzero(np.abs(hand_jac).max(axis=0) > 0.0).tolist()
        pose = self._hand_pose(q_attach)
        self.reference_rotation = pose[:3, :3].copy()
        position = pose[:3, :3] @ com + pose[:3, 3]
        self.attached_bar_q = np.concatenate([position, np.zeros(3)])

    # -- state helpers ---------------------------------------------------
    def _body(self, values: Array) -> Array:
        vector = np.asarray(values, dtype=float)
        if vector.shape[0] < self.nv_body or not np.isfinite(vector).all():
            raise ValueError(f"expected at least {self.nv_body} finite values")
        return vector[: self.nv_body]

    def _split(self, values: Array) -> tuple[Array, Array]:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.nv_body + 6,) or not np.isfinite(vector).all():
            raise ValueError(
                f"expected {self.nv_body + 6} finite values in coordinate_order"
            )
        return vector[: self.nv_body], vector[self.nv_body :]

    def attached_state(
        self, q_body: Array, v_body: Array, *, hanging: bool = True
    ) -> tuple[Array, Array]:
        """Augmented state with the bar held at the hands, moving rigidly with them.

        At the attachment pose the bushings are undeflected; with ``hanging``
        the bar is lowered by the static deflection ``K^-1 m g`` of the two
        bushings in parallel, so a still bar hangs in equilibrium.  The bar
        velocity is the hands' rigid-body velocity at its centre of mass.
        """
        q_b, v_b = self._body(q_body), self._body(v_body)
        hand, _ = self._hand_state(q_b, v_b)
        arm = hand.rotation @ np.asarray(self.dynamics.com_m, dtype=float)
        v_bar = np.concatenate(
            [
                hand.velocity_m_s + cross3(hand.omega_rad_s, arm),
                self.reference_rotation.T @ hand.omega_rad_s,
            ]
        )
        q_bar = self.attached_bar_q.copy()
        if hanging:
            q_bar[:3] += self._static_deflection(hand.rotation)
        return np.concatenate([q_b, q_bar]), np.concatenate([v_b, v_bar])

    def _static_deflection(self, hand_rotation: Array) -> Array:
        """World offset ``K^-1 m g`` of the bar on both translational springs."""
        bushing = self.interface.bushing
        local = np.diag(bushing.translational_stiffness_n_m)
        stiffness = np.zeros((3, 3))
        for side in SIDES:
            axes = hand_rotation @ np.asarray(self.interface.frame(side).rotation)
            stiffness += axes @ local @ axes.T
        weight = self.dynamics.mass_kg * self.gravity
        return np.asarray(np.linalg.solve(stiffness, weight), dtype=float)

    # -- hand (club body) kinematics -------------------------------------
    def _hand_pose(self, q_body: Array) -> Array:
        frame = self._plant.kinematic_frames(q_body)[HAND_BODY_FRAME]
        return np.asarray(frame, dtype=float) @ self._hand_to_frame

    def _hand_jacobian(self, q_body: Array) -> Array:
        """(6, nv) [linear; angular] world Jacobian of the hand body origin.

        Fourth-order central differences of the engine's forward kinematics
        (step :data:`_FD_STEP`); see the module docstring for the error.
        """
        jac = np.zeros((6, self.nv_body))
        rotation_t = self._hand_pose(q_body)[:3, :3].T
        for k in self._columns:
            step = np.zeros(self.nv_body)
            step[k] = _FD_STEP
            pose_rate = sum(
                (
                    weight * self._hand_pose(q_body + scale * step)
                    for scale, weight in _STENCIL
                ),
                start=np.zeros((4, 4), dtype=float),
            )
            rate = np.asarray(pose_rate / (12.0 * _FD_STEP), dtype=float)
            jac[:3, k] = rate[:3, 3]
            jac[3:, k] = _vee_skew(rate[:3, :3] @ rotation_t)
        return jac

    def _hand_state(self, q_body: Array, v_body: Array) -> tuple[RigidBodyState, Array]:
        pose = self._hand_pose(q_body)
        jac = self._hand_jacobian(q_body)
        twist = jac @ v_body
        state = RigidBodyState(pose[:3, :3], pose[:3, 3], twist[:3], twist[3:])
        return state, jac

    # -- barbell kinematics ----------------------------------------------
    def bar_state(self, q_bar: Array, v_bar: Array) -> RigidBodyState:
        """World pose and twist of the barbell centre-of-mass frame."""
        angles = np.asarray(q_bar[3:], dtype=float)
        rotation = self.reference_rotation @ _rotation_xyz(angles)
        omega_body = np.linalg.solve(body_xyz_n_matrix(angles), v_bar[3:])
        return RigidBodyState(
            rotation, np.array(q_bar[:3]), np.array(v_bar[:3]), rotation @ omega_body
        )

    def _sample(
        self, q_body: Array, v_body: Array, q_bar: Array, v_bar: Array
    ) -> tuple[BushingSample, RigidBodyState, Array]:
        hand, jac = self._hand_state(q_body, v_body)
        hands = hand_frame_states(hand, self.interface)
        bar = self.bar_state(q_bar, v_bar)
        wrench, points = {}, {}
        for side in SIDES:
            grip = bar.frame(self._grip_offsets[side])
            wrench[side] = bushing_wrench(self.interface.bushing, hands[side], grip)
            points[side] = grip.position_m
        return BushingSample(wrench, points, bar), hand, jac

    def sample(self, q: Array, v: Array) -> BushingSample:
        """Per-hand bushing wrenches ON THE BAR at the augmented state."""
        q_body, q_bar = self._split(q)
        v_body, v_bar = self._split(v)
        return self._sample(q_body, v_body, q_bar, v_bar)[0]

    # -- dynamics ----------------------------------------------------------
    def _hand_load(
        self, sample: BushingSample, hand: RigidBodyState, jac: Array
    ) -> Array:
        """Generalized force of both bushings on the body (``J^T`` wrench)."""
        force = np.zeros(3)
        moment = np.zeros(3)
        for side in SIDES:
            w, point = sample.wrench[side], sample.bar_grip_m[side]
            force -= w.force_n
            moment -= w.moment_nm + cross3(point - hand.position_m, w.force_n)
        return jac[:3].T @ force + jac[3:].T @ moment

    def hand_generalized_force(self, q: Array, v: Array) -> Array:
        """Generalized force (body coordinates) the bar exerts through the hands."""
        q_body, q_bar = self._split(q)
        v_body, v_bar = self._split(v)
        return self._hand_load(*self._sample(q_body, v_body, q_bar, v_bar))

    def weight_generalized_force(self, q_body: Array) -> Array:
        """Generalized force of the bar's weight hung rigidly from the hands.

        The static load a lifter anticipates: ``m g`` at the bar centre of
        mass carried by the hand body, mapped with the hand Jacobian.  A
        function of the body pose only (no bushing dynamics).
        """
        q_b = self._body(q_body)
        hand, jac = self._hand_state(q_b, np.zeros(self.nv_body))
        arm = hand.rotation @ np.asarray(self.dynamics.com_m, dtype=float)
        weight = self.dynamics.mass_kg * self.gravity
        return jac[:3].T @ weight + jac[3:].T @ cross3(arm, weight)

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array:
        """Coupled body and bar accelerations; ``tau`` bar entries must be zero."""
        q_body, q_bar = self._split(q)
        v_body, v_bar = self._split(v)
        tau_body, tau_bar = self._split(tau)
        if np.any(tau_bar != 0.0):
            raise ValueError("the barbell is unactuated: its efforts must be zero")
        sample, hand, jac = self._sample(q_body, v_body, q_bar, v_bar)
        load = self._hand_load(sample, hand, jac)
        a_body = self._plant.acceleration(q_body, v_body, tau_body + load)
        return np.concatenate([a_body, self._bar_acceleration(sample, q_bar, v_bar)])

    def _bar_acceleration(
        self, sample: BushingSample, q_bar: Array, v_bar: Array
    ) -> Array:
        bar = sample.bar
        force = self.dynamics.mass_kg * self.gravity
        moment = np.zeros(3)
        for side in SIDES:
            w = sample.wrench[side]
            force = force + w.force_n
            moment = moment + w.moment_nm
            moment = moment + cross3(
                sample.bar_grip_m[side] - bar.position_m, w.force_n
            )
        rotation = bar.rotation
        inertia = rotation @ self.dynamics.inertia_com_kg_m2 @ rotation.T
        omega = bar.omega_rad_s
        alpha = np.linalg.solve(inertia, moment - cross3(omega, inertia @ omega))
        angles = np.asarray(q_bar[3:], dtype=float)
        n_matrix = body_xyz_n_matrix(angles)
        omega_body = rotation.T @ omega
        angle_acc = (
            n_matrix @ (rotation.T @ alpha)
            + _n_matrix_rate(angles, v_bar[3:]) @ omega_body
        )
        return np.concatenate([force / self.dynamics.mass_kg, angle_acc])

    # -- closure and kinematics (body only) -----------------------------
    def closure_pose_residual(self, q: Array) -> Array:
        """The body's weld pose residual (the bar has no closure)."""
        return self._plant.closure_pose_residual(self._body(q))

    def closure_rate_matrix(self, q: Array) -> Array:
        """Body rate map with zero bar columns (6 x (nv + 6))."""
        rows = self._plant.closure_rate_matrix(self._body(q))
        return np.hstack([rows, np.zeros((rows.shape[0], 6))])

    def kinematic_frames(self, q: Array) -> dict[str, Array]:
        """Body spec frames plus :data:`BAR_FRAME` (bar centre-of-mass frame)."""
        q_body, q_bar = self._split(q)
        frames = self._plant.kinematic_frames(q_body)
        bar = self.bar_state(q_bar, np.zeros(6))
        pose = np.eye(4)
        pose[:3, :3] = bar.rotation
        pose[:3, 3] = bar.position_m
        frames[BAR_FRAME] = pose
        return frames

    def record(self) -> dict[str, Any]:
        """Provenance of the bar and its bushings."""
        bushing = self.interface.bushing
        return {
            "barbell": asdict(self.barbell),
            "bar_mass_kg": self.dynamics.mass_kg,
            "bar_com_in_hand_body_m": list(self.dynamics.com_m),
            "bar_inertia_com_kg_m2": self.dynamics.inertia_com_kg_m2.tolist(),
            "bar_coordinates": list(BAR_COORDINATES),
            "reference_rotation": self.reference_rotation.tolist(),
            "attached_bar_q": self.attached_bar_q.tolist(),
            "bushing": asdict(bushing),
            "hand_separation_m": self.interface.hand_separation_m,
            "hand_body_frame": HAND_BODY_FRAME,
        }


def _frame_to_body(spec: Mapping[str, Any]) -> Array:
    """Inverse placement of :data:`HAND_BODY_FRAME` on the closure's club body.

    Raises:
        ValueError: if the frame is missing or not on the club body.
    """
    club = spec["closure"]["body_b"]
    frame = next(
        (f for f in spec.get("frames", []) if f["name"] == HAND_BODY_FRAME), None
    )
    if frame is None or frame["body"] != club:
        raise ValueError(f"spec frame {HAND_BODY_FRAME!r} must sit on body {club!r}")
    return np.linalg.inv(np.asarray(frame["placement"], dtype=float))
