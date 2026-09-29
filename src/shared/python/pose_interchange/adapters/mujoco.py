"""MuJoCo :class:`PoseConventionAdapter` implementation.

MuJoCo's MJCF free-joint ``qpos`` prefix is ``[x, y, z, qw, qx, qy, qz]``
— quaternion **w-first**. Per-joint slots that follow are radians for
hinges. This adapter mirrors that convention exactly: the canonical
``pelvis_rotation_xyz_deg`` is converted to a unit quaternion in
``[w, x, y, z]`` order before being placed into ``qpos``.

The adapter operates in mock mode by default; if a real MuJoCo
``mjModel`` is supplied as ``model``, the per-joint layout must be
passed alongside as a ``Mapping[str, JointSlot]`` (we do not attempt to
introspect MuJoCo's qpos table here).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

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
    euler_xyz_deg_to_quat_wxyz,
    quat_wxyz_to_euler_xyz_deg,
)
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
)
from src.shared.python.pose_interchange.protocol import JointSlot

_PELVIS_PREFIX = 7  # [x, y, z, qw, qx, qy, qz]
_FREE_FLYER_Q = 7
_FREE_FLYER_V = 6


class MujocoCanonicalV2Capability(str, Enum):
    """CC-10 capability names exposed by the MuJoCo canonical-v2 adapter."""

    FORWARD_DYN = "FORWARD_DYN"
    INVERSE_DYN = "INVERSE_DYN"
    CONTACT = "CONTACT"


@dataclass(frozen=True)
class CanonicalV2State:
    """Canonical-v2 q/v/a state in the frozen docs/conventions layout.

    ``q`` starts with ``[x, y, z, qw, qx, qy, qz]``. ``v`` and ``a`` start with
    ``[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]``.
    MuJoCo free joints use the same scalar-first quaternion order and qvel
    layout, with free-joint angular velocity in the local body frame.
    """

    q: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    t: float = 0.0
    convention: str = "canonical-v2"
    frame: str = "world_Zup"
    units: str = "SI"

    def __post_init__(self) -> None:
        q = _require_vector(self.q, "q", min_size=_FREE_FLYER_Q)
        v = _require_vector(self.v, "v", min_size=_FREE_FLYER_V)
        _require_vector(self.a, "a", expected_size=v.shape[0])
        if q.shape[0] != v.shape[0] + 1:
            raise ValueError(
                "canonical-v2 free-flyer q must have one more entry than v "
                f"(got q={q.shape[0]}, v={v.shape[0]})"
            )
        if self.convention != "canonical-v2":
            raise ValueError(f"unsupported convention {self.convention!r}")
        if self.frame != "world_Zup":
            raise ValueError(f"unsupported frame {self.frame!r}")
        if self.units != "SI":
            raise ValueError(f"unsupported units {self.units!r}")


@dataclass(frozen=True)
class MujocoNativeState:
    """MuJoCo-native free-joint q/v/a state.

    The free-joint prefix is identical to canonical-v2 for MuJoCo:
    ``qpos = [xyz, quat_wxyz]`` and ``qvel = [linear, angular_body]``.
    """

    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    qacc: npt.NDArray[np.float64]
    time: float = 0.0

    def __post_init__(self) -> None:
        qpos = _require_vector(self.qpos, "qpos", min_size=_FREE_FLYER_Q)
        qvel = _require_vector(self.qvel, "qvel", min_size=_FREE_FLYER_V)
        _require_vector(self.qacc, "qacc", expected_size=qvel.shape[0])
        if qpos.shape[0] != qvel.shape[0] + 1:
            raise ValueError(
                "MuJoCo free-joint qpos must have one more entry than qvel "
                f"(got qpos={qpos.shape[0]}, qvel={qvel.shape[0]})"
            )


@runtime_checkable
class MujocoCanonicalBackend(Protocol):
    """Minimal MuJoCo-like dynamics backend used by the CC-10 adapter."""

    def forward_dynamics(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return generalized acceleration for ``(q, v, u)``."""

    def inverse_dynamics(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        a: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return generalized forces via MuJoCo ``mj_inverse``."""


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    if model is None:
        return build_default_joint_layout(
            base_offset=_PELVIS_PREFIX, units="rad", sign=1, name_prefix="mj_"
        )
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "MujocoAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class MujocoAdapter:
    """Adapter for MuJoCo MJCF (free-joint qpos is ``[xyz, quat_wxyz]``)."""

    engine_name: str = "mujoco"

    def __init__(self, backend: MujocoCanonicalBackend | None = None) -> None:
        self._backend = backend

    def capabilities(self) -> frozenset[MujocoCanonicalV2Capability]:
        """Return the CC-10 capability taxonomy implemented by this adapter."""

        return frozenset(
            {
                MujocoCanonicalV2Capability.FORWARD_DYN,
                MujocoCanonicalV2Capability.INVERSE_DYN,
                MujocoCanonicalV2Capability.CONTACT,
            }
        )

    def engine_capabilities(self) -> EngineCapabilities:
        """Return the legacy capability report with MuJoCo CC-10 support."""

        return EngineCapabilities(
            engine_name="MuJoCo",
            mass_matrix=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.FULL,
            inverse_dynamics=CapabilityLevel.FULL,
            forward_sim=CapabilityLevel.FULL,
            contact_step=CapabilityLevel.FULL,
            trajectory_opt=CapabilityLevel.PARTIAL,
            extra={
                "cc10_canonical_v2_capabilities": sorted(
                    capability.value for capability in self.capabilities()
                ),
                "canonical_quat_order": "wxyz",
                "native_quat_order": "wxyz",
                "free_joint_qvel": "linear_world_then_angular_body",
                "contact_divergence": "soft_contact_vs_pinocchio_rigid_contact",
            },
        )

    def joint_layout(self, model: Any | None = None) -> Mapping[str, JointSlot]:
        return _layout_from_model(model)

    def from_canonical(
        self,
        pose: CanonicalPose,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        if pose.convention_tag != CONVENTION_TAG:
            raise ValueError(
                f"MujocoAdapter.from_canonical: unsupported convention "
                f"{pose.convention_tag!r}"
            )
        layout = _layout_from_model(model)
        max_idx = max(
            (slot.start_index + slot.length for slot in layout.values()),
            default=_PELVIS_PREFIX,
        )
        size = max(_PELVIS_PREFIX, max_idx)
        q: npt.NDArray[np.float64] = np.zeros(size, dtype=np.float64)
        q[0:3] = pose.pelvis_translation_m
        q[3:7] = euler_xyz_deg_to_quat_wxyz(pose.pelvis_rotation_xyz_deg)
        encode_joint_angles(pose.joint_angles_deg, layout, q)
        return q

    def to_canonical(
        self,
        engine_q: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> CanonicalPose:
        q = np.asarray(engine_q, dtype=float)
        if q.ndim != 1 or q.shape[0] < _PELVIS_PREFIX:
            raise ValueError(
                f"MujocoAdapter.to_canonical: expected 1-D q with at least "
                f"{_PELVIS_PREFIX} entries, got shape {q.shape}"
            )
        layout = _layout_from_model(model)
        translation = q[0:3].copy()
        rotation_deg = quat_wxyz_to_euler_xyz_deg(q[3:7])
        joint_angles = decode_joint_angles(q, layout)
        return CanonicalPose(
            pelvis_translation_m=translation,
            pelvis_rotation_xyz_deg=rotation_deg,
            joint_angles_deg=joint_angles,
        )

    def from_canonical_v2(self, state: CanonicalV2State) -> MujocoNativeState:
        """Map canonical-v2 q/v/a into MuJoCo native free-joint ordering."""

        return MujocoNativeState(
            qpos=_normalise_free_joint_quat(state.q),
            qvel=state.v.copy(),
            qacc=state.a.copy(),
            time=float(state.t),
        )

    def to_canonical_v2(self, state: MujocoNativeState) -> CanonicalV2State:
        """Map MuJoCo native q/v/a into canonical-v2 ordering."""

        return CanonicalV2State(
            q=_normalise_free_joint_quat(state.qpos),
            v=state.qvel.copy(),
            a=state.qacc.copy(),
            t=float(state.time),
        )

    def forward_dynamics(
        self, state: CanonicalV2State, tau: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Compute MuJoCo forward dynamics for a canonical-v2 state."""

        backend = self._require_backend("forward dynamics")
        native = self.from_canonical_v2(state)
        tau_array = None
        if tau is not None:
            tau_array = _require_vector(
                np.asarray(tau, dtype=np.float64),
                "tau",
                expected_size=native.qvel.shape[0],
            )
        qacc = np.asarray(
            backend.forward_dynamics(native.qpos, native.qvel, tau_array),
            dtype=np.float64,
        )
        return _require_vector(qacc, "qacc", expected_size=native.qvel.shape[0]).copy()

    def inverse_dynamics(self, state: CanonicalV2State) -> npt.NDArray[np.float64]:
        """Compute MuJoCo inverse dynamics via ``mj_inverse``."""

        backend = self._require_backend("inverse dynamics")
        native = self.from_canonical_v2(state)
        tau = np.asarray(
            backend.inverse_dynamics(native.qpos, native.qvel, native.qacc),
            dtype=np.float64,
        )
        return _require_vector(tau, "tau", expected_size=native.qvel.shape[0]).copy()

    def _require_backend(self, operation: str) -> MujocoCanonicalBackend:
        if self._backend is None:
            raise ValueError(f"MuJoCo {operation} requires a dynamics backend")
        return self._backend


def _require_vector(
    value: npt.NDArray[np.float64],
    name: str,
    *,
    min_size: int | None = None,
    expected_size: int | None = None,
) -> npt.NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {arr.shape}")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise ValueError(f"{name} must have length {expected_size}, got {arr.shape[0]}")
    if min_size is not None and arr.shape[0] < min_size:
        raise ValueError(f"{name} must have at least {min_size} entries")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _normalise_free_joint_quat(
    q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    q_arr = _require_vector(q, "q", min_size=_FREE_FLYER_Q).copy()
    quat = q_arr[3:7]
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("MuJoCo free-joint quaternion must be non-zero")
    q_arr[3:7] = quat / norm
    return q_arr
