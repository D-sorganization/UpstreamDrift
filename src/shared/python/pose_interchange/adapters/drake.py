"""Drake :class:`PoseConventionAdapter` implementation.

Drake reads URDF and represents a free-flyer's ``q`` vector as
``[x, y, z, roll, pitch, yaw]`` followed by per-joint scalar positions
in radians. This adapter assumes the canonical golfer layout: a 6-DOF
pelvis prefix in ``[xyz, rpy]`` order, then one revolute slot per
canonical joint.

The implementation is mock-mode by design: the per-joint layout comes
from a hardcoded fixture so the adapter works in CI without a Drake
wheel installed. If a real Drake ``MultibodyPlant`` is supplied as
``model``, the layout mapping must be passed explicitly via the
``model.joint_layout`` attribute (a dict-of-:class:`JointSlot`); we do
not try to introspect Drake's plant from this adapter.

Drake's RPY convention here is intrinsic XYZ in degrees-on-the-canonical
side and radians-on-the-engine side; we do the deg <-> rad conversion at
the boundary so the canonical-pose contract stays in degrees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.pose_interchange.adapters._base import (
    build_default_joint_layout,
    decode_joint_angles,
    encode_joint_angles,
)
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
)
from src.shared.python.pose_interchange.protocol import JointSlot

_PELVIS_PREFIX = 6  # [x, y, z, roll, pitch, yaw]
_CANONICAL_V2_BASE_NQ = 7  # [xyz, quat_wxyz]
_CANONICAL_V2_BASE_NV = 6  # [linear_world, angular_body]
_DRAKE_QUATERNION_BASE_NQ = 7  # [quat_wxyz, xyz]
_QUAT_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class DrakeAdapterCapabilities:
    """Capability declaration for the Drake canonical-core adapter."""

    supported: frozenset[str]
    model_exports: tuple[str, ...]
    contact_model: str
    gradient_scalar: str

    def supports(self, capability: str) -> bool:
        """Return whether *capability* is advertised by this adapter."""
        return capability in self.supported

    def to_engine_capabilities(self) -> EngineCapabilities:
        """Materialize the shared engine-core capability report."""
        return EngineCapabilities(
            engine_name="Drake",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.FULL,
            inverse_dynamics=CapabilityLevel.FULL,
            parameter_gradients=CapabilityLevel.PARTIAL,
            state_control_gradients=CapabilityLevel.FULL,
            forward_sim=CapabilityLevel.FULL,
            contact_step=CapabilityLevel.FULL,
            trajectory_opt=CapabilityLevel.FULL,
            extra={
                "gradient_scalar": self.gradient_scalar,
                "contact_model": self.contact_model,
                "model_exports": self.model_exports,
            },
        )


DRAKE_ADAPTER_CAPABILITIES = DrakeAdapterCapabilities(
    supported=frozenset(
        {
            "forward_sim",
            "inverse_dynamics",
            "contact",
            "contact_forces",
            "contact_step",
            "state_control_gradients",
            "trajectory_opt",
            "urdf_export",
            "sdf_export",
        }
    ),
    model_exports=("urdf", "sdf"),
    contact_model="hydroelastic_or_point_contact",
    gradient_scalar="AutoDiffXd",
)
"""Drake canonical-core capability profile for CC-28."""


@dataclass(frozen=True, slots=True)
class DrakeNativeState:
    """Drake quaternion-floating native state.

    Layout:

    - ``q``: ``[quat_wxyz, xyz, joints...]``.
    - ``v``/``a``: ``[angular_world, linear_world, joints...]``.

    The canonical-v2 contract stores ``q`` as ``[xyz, quat_wxyz, joints...]`` and
    stores base angular velocity in the body frame. Drake's
    ``QuaternionFloatingJoint`` orders base velocity as angular then linear, and
    expresses both in the parent frame. This value type pins that boundary.
    """

    q: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    t: float = 0.0
    layout: str = "drake-quaternion-floating-v1"

    def __post_init__(self) -> None:
        q = _readonly_vector(self.q, "q")
        v = _readonly_vector(self.v, "v")
        a = _readonly_vector(self.a, "a")
        _validate_state_shapes(q, v, a)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "t", float(self.t))


@dataclass(frozen=True, slots=True)
class DrakeNamedState:
    """Lossless named-array state used by the draft CC-7 harness."""

    values: Mapping[str, npt.NDArray[np.float64]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _copy_named_arrays(self.values))


@dataclass(frozen=True, slots=True)
class DrakeDivergenceRegistration:
    """Machine-readable divergence registration for the CC-7 registry."""

    id: str
    check_name: str
    metric_name: str
    engines: tuple[str, str]
    tolerance: float
    rationale: str


HYDROELASTIC_CONTACT_DIVERGENCE = DrakeDivergenceRegistration(
    id="drake-hydroelastic-vs-pinocchio-rigid-contact",
    check_name="differential_cross_engine_reference",
    metric_name="position",
    engines=("drake", "pinocchio"),
    tolerance=5.0e-3,
    rationale=(
        "Drake may use hydroelastic contact pressures while Pinocchio's "
        "canonical-core path is rigid/contact-impulse based; contact-rich "
        "rollouts can therefore differ within the registered tolerance."
    ),
)
"""Registered CC-28 contact divergence against the Pinocchio reference."""


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    """Return the joint layout, falling back to the mock fixture.

    If *model* is None, returns the hardcoded mock layout.
    If *model* exposes a ``joint_layout`` mapping attribute, returns
    that.  Anything else is rejected with :class:`TypeError`.
    """
    if model is None:
        return build_default_joint_layout(
            base_offset=_PELVIS_PREFIX, units="rad", sign=1, name_prefix="drake_"
        )
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "DrakeAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class DrakeAdapter:
    """Adapter for Drake URDF/SDF (free-flyer ``q`` is ``[xyz, rpy]``)."""

    engine_name: str = "drake"
    capabilities: DrakeAdapterCapabilities = DRAKE_ADAPTER_CAPABILITIES

    def get_capabilities(self) -> EngineCapabilities:
        """Return Drake's canonical-core engine capability report."""
        return self.capabilities.to_engine_capabilities()

    def supported_model_exports(self) -> tuple[str, ...]:
        """Return canonical model exchange formats accepted by the Drake path."""
        return self.capabilities.model_exports

    def registered_divergences(self) -> tuple[DrakeDivergenceRegistration, ...]:
        """Return known, documented Drake cross-engine divergences."""
        return (HYDROELASTIC_CONTACT_DIVERGENCE,)

    def joint_layout(self, model: Any | None = None) -> Mapping[str, JointSlot]:
        return _layout_from_model(model)

    @overload
    def from_canonical(
        self,
        pose: CanonicalPose,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def from_canonical(
        self,
        pose: Mapping[str, object],
        *,
        model: Any | None = None,
    ) -> DrakeNativeState | DrakeNamedState: ...

    def from_canonical(
        self,
        pose: CanonicalPose | Mapping[str, object],
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64] | DrakeNativeState | DrakeNamedState:
        if isinstance(pose, Mapping):
            return self.from_canonical_state(pose)
        if not isinstance(pose, CanonicalPose):
            raise TypeError(
                "DrakeAdapter.from_canonical expects CanonicalPose or a "
                f"canonical-v2 Mapping, got {type(pose).__name__}"
            )
        if pose.convention_tag != CONVENTION_TAG:
            raise ValueError(
                f"DrakeAdapter.from_canonical: unsupported convention "
                f"{pose.convention_tag!r}"
            )
        layout = _layout_from_model(model)
        size = _PELVIS_PREFIX + max(
            (slot.start_index + slot.length for slot in layout.values()),
            default=_PELVIS_PREFIX,
        )
        # The layout's start_index is already absolute (>= _PELVIS_PREFIX), so
        # we don't need to subtract the prefix.
        size = max(size, _PELVIS_PREFIX)
        max_idx = max(
            (slot.start_index + slot.length for slot in layout.values()),
            default=_PELVIS_PREFIX,
        )
        size = max(size, max_idx)
        q = np.zeros(size, dtype=float)
        q[0:3] = pose.pelvis_translation_m
        q[3:6] = np.radians(pose.pelvis_rotation_xyz_deg)
        encode_joint_angles(pose.joint_angles_deg, layout, q)
        return q

    @overload
    def to_canonical(
        self,
        engine_q: DrakeNativeState | DrakeNamedState,
        *,
        model: Any | None = None,
    ) -> dict[str, npt.NDArray[np.float64] | float]: ...

    @overload
    def to_canonical(
        self,
        engine_q: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> CanonicalPose: ...

    def to_canonical(
        self,
        engine_q: npt.ArrayLike | DrakeNativeState | DrakeNamedState,
        *,
        model: Any | None = None,
    ) -> CanonicalPose | dict[str, npt.NDArray[np.float64] | float]:
        if isinstance(engine_q, DrakeNativeState | DrakeNamedState):
            return self.to_canonical_state(engine_q)
        q = np.asarray(engine_q, dtype=float)
        if q.ndim != 1 or q.shape[0] < _PELVIS_PREFIX:
            raise ValueError(
                f"DrakeAdapter.to_canonical: expected 1-D q with at least "
                f"{_PELVIS_PREFIX} entries, got shape {q.shape}"
            )
        layout = _layout_from_model(model)
        translation = q[0:3].copy()
        rotation_deg = np.degrees(q[3:6])
        joint_angles = decode_joint_angles(q, layout)
        return CanonicalPose(
            pelvis_translation_m=translation,
            pelvis_rotation_xyz_deg=rotation_deg,
            joint_angles_deg=joint_angles,
        )

    def from_canonical_state(
        self,
        state: Mapping[str, object],
    ) -> DrakeNativeState | DrakeNamedState:
        """Encode canonical-v2 state data into Drake native ordering.

        Full canonical-v2 mappings use ``q``, ``v`` and ``a`` arrays. The draft
        CC-7 harness also passes named DOF maps; those are copied losslessly so
        the harness can still verify registry behavior before CC-2 lands.
        """
        if {"q", "v", "a"} <= set(state):
            q = _readonly_vector(state["q"], "q")
            v = _readonly_vector(state["v"], "v")
            a = _readonly_vector(state["a"], "a")
            _validate_state_shapes(q, v, a)
            rotation = _rotation_matrix_from_quat_wxyz(q[3:7])
            drake_q = np.concatenate([q[3:7], q[0:3], q[7:]])
            drake_v = _drake_tangent_from_canonical(v, rotation)
            drake_a = _drake_tangent_from_canonical(a, rotation)
            return DrakeNativeState(
                q=drake_q,
                v=drake_v,
                a=drake_a,
                t=_state_time(state),
            )
        return DrakeNamedState(_copy_named_arrays(state))

    def to_canonical_state(
        self,
        state: DrakeNativeState | DrakeNamedState,
    ) -> dict[str, npt.NDArray[np.float64] | float]:
        """Decode a Drake-native state back to canonical-v2 ordering."""
        if isinstance(state, DrakeNamedState):
            return {key: value.copy() for key, value in state.values.items()}

        _validate_state_shapes(state.q, state.v, state.a)
        canonical_q = np.concatenate([state.q[4:7], state.q[0:4], state.q[7:]])
        rotation = _rotation_matrix_from_quat_wxyz(canonical_q[3:7])
        return {
            "q": canonical_q,
            "v": _canonical_tangent_from_drake(state.v, rotation),
            "a": _canonical_tangent_from_drake(state.a, rotation),
            "t": state.t,
        }


def _readonly_vector(value: object, name: str) -> npt.NDArray[np.float64]:
    arr = np.asarray(value, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    arr.setflags(write=False)
    return arr


def _validate_state_shapes(
    q: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
) -> None:
    if q.shape[0] < _CANONICAL_V2_BASE_NQ:
        raise ValueError(
            f"q must have at least {_CANONICAL_V2_BASE_NQ} entries, got {q.shape[0]}"
        )
    nv = q.shape[0] - 1
    if v.shape[0] != nv or a.shape[0] != nv:
        raise ValueError(
            f"nq must equal nv + 1: q={q.shape[0]}, v={v.shape[0]}, a={a.shape[0]}"
        )
    quat = q[3:7] if _looks_canonical_q(q) else q[0:4]
    norm = float(np.linalg.norm(quat))
    if abs(norm - 1.0) > _QUAT_TOL:
        raise ValueError(f"base quaternion must have unit norm, got {norm:.6g}")


def _looks_canonical_q(q: npt.NDArray[np.float64]) -> bool:
    return q.shape[0] >= _CANONICAL_V2_BASE_NQ and abs(np.linalg.norm(q[3:7]) - 1) < (
        abs(np.linalg.norm(q[0:4]) - 1) + _QUAT_TOL
    )


def _copy_named_arrays(
    state: Mapping[str, object],
) -> dict[str, npt.NDArray[np.float64]]:
    copied: dict[str, npt.NDArray[np.float64]] = {}
    for key, value in state.items():
        if key == "t":
            continue
        copied[str(key)] = _readonly_vector(value, str(key))
    return copied


def _state_time(state: Mapping[str, object]) -> float:
    raw = state.get("t", 0.0)
    if not isinstance(raw, Real):
        raise TypeError("canonical-v2 state time must be numeric")
    return float(raw)


def _rotation_matrix_from_quat_wxyz(
    quat_wxyz: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    q = np.asarray(quat_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quat_wxyz must have shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("quat_wxyz must not be zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _drake_tangent_from_canonical(
    tangent: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    angular_world = rotation @ tangent[3:_CANONICAL_V2_BASE_NV]
    return np.concatenate(
        [angular_world, tangent[0:3], tangent[_CANONICAL_V2_BASE_NV:]]
    )


def _canonical_tangent_from_drake(
    tangent: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    angular_body = rotation.T @ tangent[0:3]
    return np.concatenate(
        [
            tangent[3:_CANONICAL_V2_BASE_NV],
            angular_body,
            tangent[_CANONICAL_V2_BASE_NV:],
        ]
    )
