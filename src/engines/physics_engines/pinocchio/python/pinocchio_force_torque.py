"""Pinocchio force/torque overlay source (FTO-13, #11298, ADR-0052).

Builds an engine-agnostic :class:`ForceTorqueFrame` from Pinocchio's
recursive Newton-Euler pass. The source owns a private ``pin.Data`` so it never
mutates the engine's data.

Conventions (all vectors world-frame Z-up, SI units):

* ``JOINT_REACTION``: ``data.f[i]`` is the spatial force transmitted from the
  parent to body ``i`` at the joint origin, expressed in the local joint frame.
  It is rotated into the world frame here. It is computed with the *actual*
  acceleration; torque is never zeroed.
* ``JOINT_ACTUATOR``: applied generalized torque projected on the world joint
  axis, anchored at the joint origin. Prismatic, free-flyer and other joint
  types are omitted (never reported as zero).
* ``CONTACT``: existing ``ContactSample`` objects passed through unchanged
  (force only; the torque half stays unavailable).

Note: FTO-2 (#11287) owns the shared ``force_overlay.conversions`` module and
FTO-11 owns shared segment axes. Neither is on main yet, so this module keeps
minimal private equivalents (inline frame rotation, ``_segment_axes``). They should
be replaced by the shared helpers once those land.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pinocchio as pin

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadFrame,
    axial_force_from_proximal_reaction,
)
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.contact_law import ContactSample

logger = get_logger(__name__)

ENGINE_NAME = "pinocchio"
SOURCE_NAME = "pinocchio:rnea"
_MIN_SEGMENT_LENGTH_M = 1e-9
_REVOLUTE = frozenset(
    {
        "JointModelRX",
        "JointModelRY",
        "JointModelRZ",
        "JointModelRUBX",
        "JointModelRUBY",
        "JointModelRUBZ",
        "JointModelRevoluteUnaligned",
        "JointModelRevoluteUnboundedUnaligned",
    }
)
_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9_.:-]")


def _safe_name(name: str) -> str:
    """Return ``name`` restricted to the characters allowed in wrench labels."""
    return _LABEL_UNSAFE.sub("_", name) or "unnamed"


def _vec3(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


class PinocchioForceTorqueSource:
    """Compute world-frame joint reactions, actuator torques and axial loads.

    Preconditions: ``model`` is a ``pinocchio.Model``.
    Postconditions: ``sample`` returns a new immutable frame; neither the
    inputs nor any externally owned ``pin.Data`` are modified.
    """

    def __init__(self, model: Any) -> None:
        if not isinstance(model, pin.Model):
            raise TypeError("model must be a pinocchio.Model")
        self._model = model
        self._data = model.createData()

    @property
    def model(self) -> Any:
        """The Pinocchio model this source was built for."""
        return self._model

    def _vector(self, name: str, value: Any, length: int) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != length:
            raise ValueError(
                f"{name} must be a vector of length {length}, got shape {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def sample(
        self,
        q: Any,
        v: Any,
        a: Any,
        tau_applied: Any,
        contact_samples: Mapping[str, ContactSample] | Sequence[ContactSample] = (),
        time_s: float = 0.0,
    ) -> ForceTorqueFrame:
        """Return the force/torque frame at ``(q, v, a)`` with applied ``tau``.

        Raises:
            ValueError: If an input has the wrong length or is non-finite.
        """
        model = self._model
        q_arr = self._vector("q", q, model.nq)
        v_arr = self._vector("v", v, model.nv)
        a_arr = self._vector("a", a, model.nv)
        tau_arr = self._vector("tau", tau_applied, model.nv)
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")

        data = self._data
        pin.rnea(model, data, q_arr, v_arr, a_arr)
        pin.forwardKinematics(model, data, q_arr, v_arr, a_arr)
        pin.computeJointJacobians(model, data, q_arr)

        wrenches: list[OverlayWrench] = []
        reactions: dict[int, tuple[float, float, float]] = {}
        for joint_id in range(1, model.njoints):
            wrenches.extend(self._joint_wrenches(joint_id, tau_arr, reactions))
        wrenches.extend(self._contact_wrenches(contact_samples))

        return ForceTorqueFrame(
            time_s=time_s,
            engine=ENGINE_NAME,
            wrenches=tuple(wrenches),
            axial_loads=self._axial_loads(reactions, time_s),
        )

    # ------------------------------------------------------------------ joints

    def _joint_wrenches(
        self,
        joint_id: int,
        tau: np.ndarray,
        reactions: dict[int, tuple[float, float, float]],
    ) -> list[OverlayWrench]:
        model, data = self._model, self._data
        name = str(model.names[joint_id])
        label = _safe_name(name)
        pose = data.oMi[joint_id]
        rotation = np.asarray(pose.rotation)
        anchor = _vec3(pose.translation)

        local = data.f[joint_id]
        force_world = _vec3(rotation @ np.asarray(local.linear))
        torque_world = _vec3(rotation @ np.asarray(local.angular))
        reactions[joint_id] = force_world
        result = [
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label=f"joint_reaction:{label}",
                body=name,
                point_m=anchor,
                force_n=force_world,
                torque_nm=torque_world,
                source=SOURCE_NAME,
            )
        ]
        moment = self._actuator_moment(joint_id, rotation, tau)
        if moment is not None:
            result.append(
                OverlayWrench(
                    kind=WrenchKind.JOINT_ACTUATOR,
                    label=f"joint_actuator:{label}",
                    body=name,
                    point_m=anchor,
                    torque_nm=moment,
                    source=SOURCE_NAME,
                )
            )
        return result

    def _actuator_moment(
        self, joint_id: int, rotation: np.ndarray, tau: np.ndarray
    ) -> tuple[float, float, float] | None:
        """World moment of the applied torque, or None for unsupported joints."""
        joint = self._model.joints[joint_id]
        kind = joint.shortname()
        start = int(joint.idx_v)
        if kind in _REVOLUTE:
            # Local motion subspace: angular part of the joint-frame Jacobian.
            local_jacobian = pin.getJointJacobian(
                self._model, self._data, joint_id, pin.ReferenceFrame.LOCAL
            )
            axis = np.asarray(local_jacobian, dtype=float).reshape(6, -1)[3:, start]
            norm = float(np.linalg.norm(axis))
            if norm <= 0.0 or not math.isfinite(norm):
                return None
            axis = axis / norm
        elif kind == "JointModelSpherical":
            return _vec3(rotation @ tau[start : start + 3])
        else:
            return None
        return _vec3(float(tau[start]) * (rotation @ axis))

    # ------------------------------------------------------------ axial loads

    def _segment_axes(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Proximal/distal world points per body from ``oMi`` translations.

        Distal is the single child joint origin, or the centre of mass for a
        leaf body. Bodies with several children or a degenerate axis are
        omitted rather than guessed.
        """
        model, data = self._model, self._data
        children: dict[int, list[int]] = {}
        for joint_id in range(1, model.njoints):
            children.setdefault(int(model.parents[joint_id]), []).append(joint_id)
        axes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for joint_id in range(1, model.njoints):
            kids = children.get(joint_id, [])
            proximal = np.asarray(data.oMi[joint_id].translation, dtype=float)
            if len(kids) == 1:
                distal = np.asarray(data.oMi[kids[0]].translation, dtype=float)
            elif not kids:
                lever = np.asarray(model.inertias[joint_id].lever, dtype=float)
                distal = (
                    np.asarray(data.oMi[joint_id].rotation, dtype=float) @ lever
                    + proximal
                )
            else:
                continue
            if np.linalg.norm(distal - proximal) > _MIN_SEGMENT_LENGTH_M:
                axes[joint_id] = (proximal, distal)
        return axes

    def _axial_loads(
        self, reactions: Mapping[int, tuple[float, float, float]], time_s: float
    ) -> AxialLoadFrame | None:
        values: dict[str, float | None] = {}
        for joint_id, (proximal, distal) in self._segment_axes().items():
            values[str(self._model.names[joint_id])] = (
                axial_force_from_proximal_reaction(
                    reactions[joint_id], proximal, distal
                )
            )
        if not values:
            return None
        return AxialLoadFrame(time_s=time_s, values_n=values, source=SOURCE_NAME)

    # ---------------------------------------------------------------- contacts

    @staticmethod
    def _contact_wrenches(
        samples: Mapping[str, ContactSample] | Sequence[ContactSample],
    ) -> list[OverlayWrench]:
        """Pass ``ContactSample`` forces through as world ``CONTACT`` wrenches.

        Samples with no force (no contact) are skipped. The torque half is
        left unavailable because a sphere contact carries no moment.
        """
        if isinstance(samples, Mapping):
            named = list(samples.items())
        else:
            named = [(f"contact_{k}", s) for k, s in enumerate(samples)]
        result = []
        for name, sample in named:
            if not isinstance(sample, ContactSample):
                raise TypeError("contact_samples must contain ContactSample objects")
            force = np.asarray(sample.normal_force_n, dtype=float) + np.asarray(
                sample.friction_force_n, dtype=float
            )
            if not np.any(force):
                continue
            result.append(
                OverlayWrench(
                    kind=WrenchKind.CONTACT,
                    label=f"contact:{_safe_name(str(name))}",
                    body=str(name),
                    point_m=_vec3(sample.contact_point_m),
                    force_n=_vec3(force),
                    source="pinocchio:contact",
                )
            )
        return result
