"""Native Pinocchio golfer ModelGripAdapter (HRG-GS-M9, #13331).

Thin adapter connecting the native Pinocchio golfer model (built with
``hands="separate"``) to the shared grip contact binding contract:
- Resolves a :class:`~src.shared.python.grip_contact.binding.HandClubBinding`
  from ``src/config/grip_bindings/pinocchio_native_golfer.json`` via
  :mod:`src.shared.python.grip_contact.binding_manifest`.
- Samples 4x4 world poses via ``pinocchio.framesForwardKinematics`` and frame
  velocities via ``pinocchio.getFrameVelocity`` in ``LOCAL_WORLD_ALIGNED``.
- Maps externally applied body wrenches (world force and moment about frame
  origin) to ``pinocchio.Force`` objects on the parent joints.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
    LEAD_HAND_FRAME_NAME,
    TRAIL_HAND_FRAME_NAME,
    build_golfer_model,
    is_pinocchio_available,
)
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    HandClubBinding,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.contact_topology import ContactTopology

__all__ = [
    "DEFAULT_NATIVE_MANIFEST_PATH",
    "NativePinocchioGolferGripAdapter",
]

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
DEFAULT_NATIVE_MANIFEST_PATH: Final[Path] = (
    _REPO_ROOT / "src" / "config" / "grip_bindings" / "pinocchio_native_golfer.json"
)


def _require_pinocchio() -> Any:
    """Import and return ``pinocchio`` or raise ``ImportError``."""
    if not is_pinocchio_available():
        raise ImportError(
            "Pinocchio C++ runtime is required for NativePinocchioGolferGripAdapter"
        )
    import pinocchio as pin  # noqa: PLC0415

    return pin


def _load_manifest(manifest_or_path: Mapping[str, Any] | Path | None) -> dict[str, Any]:
    """Load and validate a ``grip-binding/v1`` manifest mapping or JSON file."""
    if manifest_or_path is None:
        raw = json.loads(DEFAULT_NATIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    elif isinstance(manifest_or_path, Path):
        raw = json.loads(manifest_or_path.read_text(encoding="utf-8"))
    elif isinstance(manifest_or_path, Mapping):
        raw = dict(manifest_or_path)
    else:
        raise TypeError(
            "manifest must be a Mapping, Path, or None, "
            f"got {type(manifest_or_path).__name__}"
        )
    validate_manifest(raw)
    return dict(raw)


def _compute_club_grip_frames(
    pin_mod: Any,
    model: Any,
    club_body: str,
    lead_body: str,
    trail_body: str,
) -> dict[str, np.ndarray]:
    """Compute club-local 4x4 grip frame transforms for ``lead_body`` and ``trail_body``."""
    data = model.createData()
    q0 = pin_mod.neutral(model)
    pin_mod.framesForwardKinematics(model, data, q0)

    club_id = int(model.getFrameId(club_body))
    lead_id = int(model.getFrameId(lead_body))
    trail_id = int(model.getFrameId(trail_body))

    omf = data.oMf
    se3_club_inv = omf[club_id].inverse()
    se3_club_lead = se3_club_inv * omf[lead_id]
    se3_club_trail = se3_club_inv * omf[trail_id]

    return {
        "L": np.asarray(se3_club_lead.homogeneous, dtype=float).copy(),
        "R": np.asarray(se3_club_trail.homogeneous, dtype=float).copy(),
    }


def _map_world_wrench_to_joint_force(
    pin_mod: Any,
    model: Any,
    data: Any,
    frame_name: str,
    force_world_n: np.ndarray,
    moment_world_nm: np.ndarray,
) -> tuple[int, Any]:
    """Map a world-aligned wrench at ``frame_name`` to a local ``pinocchio.Force`` on its parent joint."""
    fid = int(model.getFrameId(frame_name))
    frame_obj = model.frames[fid]
    parent_joint = int(frame_obj.parentJoint)
    placement = frame_obj.placement

    omf = data.oMf
    rot_world_frame = np.asarray(omf[fid].rotation, dtype=float)
    f_local_frame = pin_mod.Force(
        rot_world_frame.T @ force_world_n,
        rot_world_frame.T @ moment_world_nm,
    )
    f_joint = placement.act(f_local_frame)
    return parent_joint, f_joint


class NativePinocchioGolferGripAdapter:
    """Native Pinocchio golfer :class:`ModelGripAdapter` with two addressable hands."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        manifest: Mapping[str, Any] | Path | None = None,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
    ) -> None:
        pin_mod = _require_pinocchio()
        self._pin = pin_mod
        self._manifest = _load_manifest(manifest)

        self._club_body = str(self._manifest["club_body"])
        self._lead_hand_body = str(self._manifest["lead_hand_body"])
        self._trail_hand_body = str(self._manifest["trail_hand_body"])

        resolved_model = (
            model if model is not None else build_golfer_model(hands="separate")
        )
        for required_frame in (
            self._club_body,
            self._lead_hand_body,
            self._trail_hand_body,
        ):
            if not resolved_model.existFrame(required_frame):
                raise ValueError(
                    f"Pinocchio model is missing required frame {required_frame!r}; "
                    f"build with build_golfer_model(hands='separate') "
                    f"(expected {LEAD_HAND_FRAME_NAME!r} and {TRAIL_HAND_FRAME_NAME!r})"
                )

        self._model = resolved_model
        self._data = resolved_model.createData()
        self._club_grip_frames = _compute_club_grip_frames(
            pin_mod,
            resolved_model,
            self._club_body,
            self._lead_hand_body,
            self._trail_hand_body,
        )

        q_init = (
            np.asarray(q, dtype=float).copy()
            if q is not None
            else np.asarray(pin_mod.neutral(resolved_model), dtype=float).copy()
        )
        v_init = (
            np.asarray(v, dtype=float).copy()
            if v is not None
            else np.zeros(int(resolved_model.nv), dtype=float)
        )
        self.set_joint_state(q_init, v_init)

        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._wrench_history: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
        self._frame_joint_forces: dict[str, Any] = {}
        self._joint_forces: list[Any] = [
            pin_mod.Force.Zero() for _ in range(int(resolved_model.njoints))
        ]

    @property
    def model(self) -> Any:
        """Underlying ``pinocchio.Model`` instance."""
        return self._model

    @property
    def data(self) -> Any:
        """Underlying ``pinocchio.Data`` instance."""
        return self._data

    @property
    def q(self) -> np.ndarray:
        """Current generalized configuration vector ``(nq,)``."""
        return self._q.copy()

    @property
    def v(self) -> np.ndarray:
        """Current generalized velocity vector ``(nv,)``."""
        return self._v.copy()

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Most recently applied body wrenches ``{body_id: (force_n, moment_nm)}``."""
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    @property
    def wrench_history(self) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
        """History of all ``apply_wrenches`` calls on this adapter."""
        return list(self._wrench_history)

    @property
    def frame_joint_forces(self) -> dict[str, Any]:
        """Per-frame ``pinocchio.Force`` mapped onto each frame's parent joint."""
        return {k: f.copy() for k, f in self._frame_joint_forces.items()}

    @property
    def joint_forces(self) -> list[Any]:
        """Accumulated ``pinocchio.Force`` per joint ``0 .. model.njoints - 1``."""
        return [f.copy() for f in self._joint_forces]

    def set_joint_state(self, q: np.ndarray, v: np.ndarray | None = None) -> None:
        """Update ``(q, v)`` and run ``forwardKinematics`` + ``framesForwardKinematics``."""
        q_arr = np.asarray(q, dtype=float)
        nq = int(self._model.nq)
        nv = int(self._model.nv)
        if q_arr.shape != (nq,) or not bool(np.isfinite(q_arr).all()):
            raise ValueError(f"q must be a finite ({nq},) vector, got {q_arr.shape}")
        v_arr = np.zeros(nv, dtype=float) if v is None else np.asarray(v, dtype=float)
        if v_arr.shape != (nv,) or not bool(np.isfinite(v_arr).all()):
            raise ValueError(f"v must be a finite ({nv},) vector, got {v_arr.shape}")

        self._q = q_arr.copy()
        self._v = v_arr.copy()
        self._pin.forwardKinematics(self._model, self._data, self._q, self._v)
        self._pin.framesForwardKinematics(self._model, self._data, self._q)

    def capabilities(self) -> GripCapabilities:
        """Return declared capabilities for the native Pinocchio golfer adapter."""
        return GripCapabilities(
            attachment_modes=("contact", "contact_v1"),
            supports_physical_pairs=True,
            supports_wrench_application=True,
            native_contact=False,
            available_bodies=(
                self._club_body,
                self._lead_hand_body,
                self._trail_hand_body,
            ),
        )

    def resolve_binding(self, topology: ContactTopology) -> HandClubBinding:
        """Resolve a :class:`HandClubBinding` via the ``pinocchio_native_golfer`` manifest."""
        binding = binding_from_manifest(
            self._manifest, topology, self._club_grip_frames
        )
        validate_resolved_binding(binding, topology, self._club_grip_frames)
        return binding

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and ``LOCAL_WORLD_ALIGNED`` velocities from FK."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        pin_mod = self._pin
        pin_mod.forwardKinematics(self._model, self._data, self._q, self._v)
        pin_mod.framesForwardKinematics(self._model, self._data, self._q)

        ref_frame = pin_mod.ReferenceFrame.LOCAL_WORLD_ALIGNED
        omf = self._data.oMf
        poses: dict[str, np.ndarray] = {}
        velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for body_name in (
            binding.club_body,
            binding.lead_hand_body,
            binding.trail_hand_body,
        ):
            fid = int(self._model.getFrameId(body_name))
            se3_pose = omf[fid]
            poses[body_name] = np.asarray(se3_pose.homogeneous, dtype=float).copy()
            f_vel = pin_mod.getFrameVelocity(self._model, self._data, fid, ref_frame)
            velocities[body_name] = (
                np.asarray(f_vel.linear, dtype=float).copy(),
                np.asarray(f_vel.angular, dtype=float).copy(),
            )
        return BodyStates(poses, velocities=velocities)

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Map external frame wrenches to ``pinocchio.Force`` objects on parent joints."""
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        pin_mod = self._pin
        n_joints = int(self._model.njoints)
        joint_fext = [pin_mod.Force.Zero() for _ in range(n_joints)]
        per_frame_forces: dict[str, Any] = {}
        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for body_id, pair in wrenches.items():
            f_arr = np.asarray(pair[0], dtype=float)
            m_arr = np.asarray(pair[1], dtype=float)
            if f_arr.shape != (3,) or not bool(np.isfinite(f_arr).all()):
                raise ValueError(f"force for {body_id!r} must be a finite 3-vector")
            if m_arr.shape != (3,) or not bool(np.isfinite(m_arr).all()):
                raise ValueError(f"moment for {body_id!r} must be a finite 3-vector")
            cleaned[body_id] = (f_arr.copy(), m_arr.copy())
            if self._model.existFrame(body_id):
                jid, f_joint = _map_world_wrench_to_joint_force(
                    pin_mod, self._model, self._data, body_id, f_arr, m_arr
                )
                per_frame_forces[body_id] = f_joint
                joint_fext[jid] = joint_fext[jid] + f_joint

        self._applied_wrenches = cleaned
        self._wrench_history.append(cleaned)
        self._frame_joint_forces = per_frame_forces
        self._joint_forces = joint_fext
