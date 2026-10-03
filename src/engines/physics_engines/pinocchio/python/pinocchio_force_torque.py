"""Pinocchio force/torque overlay source (ADR-0052, FTO-13, #11298).

Builds a world-frame :class:`ForceTorqueFrame` from Pinocchio's recursive
Newton-Euler pass. Joint reactions come from ``data.f`` evaluated with the
**actual** acceleration (never zeroed torque), applied torques come from the
caller's generalized torque vector, and contact wrenches pass through the
existing ``ContactSample`` records unchanged.

The source owns its own ``pin.Data`` and never mutates an engine's data.

Note: the world-frame conversion and torque-wrench helpers below are minimal
private stand-ins for the FTO-2 converters (#11287), which landed after this
module was written. They should be replaced by ``force_overlay.conversions``
in a follow-up.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pinocchio as pin

from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.body_part_viz.axial_loads import (
    axial_force_from_proximal_reaction,
)
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.contact_law import ContactSample

__all__ = ["PinocchioForceTorqueSource"]

logger = get_logger(__name__)

_ENGINE = "pinocchio"
_REACTION_SOURCE = "pinocchio:rnea:data.f"
_ACTUATOR_SOURCE = "pinocchio:tau_applied"
_CONTACT_SOURCE = "pinocchio:contact_sample"
_AXIS_BY_SUFFIX = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}
_FIXED_AXIS = re.compile(r"^JointModelR(?:UB)?([XYZ])$")
_MIN_SEGMENT_LENGTH_M = 1e-12


def _label(prefix: str, name: str) -> str:
    return f"{prefix}:{re.sub(r'[^A-Za-z0-9_.:-]', '_', name)}"


def _vec3(value: object) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _is_free_flyer(joint: object) -> bool:
    return "FreeFlyer" in joint.shortname()  # type: ignore[attr-defined]


def _local_revolute_axis(joint: object) -> np.ndarray | None:
    """Return the joint-frame axis of a single-DOF revolute joint, else None."""
    name = joint.shortname()  # type: ignore[attr-defined]
    match = _FIXED_AXIS.match(name)
    if match:
        return np.array(_AXIS_BY_SUFFIX[match.group(1)])
    if "Revolute" in name and "Unaligned" in name:
        return np.asarray(joint.extract().axis, dtype=float)  # type: ignore[attr-defined]
    return None


def _require_vector(name: str, value: object, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


class PinocchioForceTorqueSource:
    """Compute world-frame force/torque overlays for a Pinocchio model.

    Preconditions: ``model`` is a ``pin.Model``.
    Postconditions: ``sample`` returns a frame whose wrenches are expressed in
    the world frame; unsupported joints are omitted, never reported as zero.
    """

    def __init__(self, model: Any) -> None:  # pin.Model (loosely typed stubs)
        if not isinstance(model, pin.Model):
            raise TypeError("model must be a pinocchio.Model")
        # Pinocchio's bundled stubs omit FrameType and Model.parents, so the
        # untyped surface is accessed through Any-typed bindings.
        native: Any = pin
        raw_model: Any = model
        self.model: Any = raw_model
        self._data: Any = raw_model.createData()
        self._body_by_joint: dict[int, str] = {}
        for frame in raw_model.frames:
            if frame.type == native.FrameType.BODY:
                self._body_by_joint.setdefault(int(frame.parentJoint), frame.name)
        self._children: dict[int, list[int]] = {}
        for j in range(1, raw_model.njoints):
            self._children.setdefault(int(raw_model.parents[j]), []).append(j)

    def body_name(self, joint_id: int) -> str:
        """Name of the body frame attached to ``joint_id`` (joint name if none)."""
        return self._body_by_joint.get(joint_id, str(self.model.names[joint_id]))

    def acceleration(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Forward dynamics (ABA) acceleration at ``(q, v, tau)``, shape ``(nv,)``.

        Uses the private data; external contact forces are not included.
        """
        model = self.model
        q_arr = _require_vector("q", q, model.nq)
        v_arr = _require_vector("v", v, model.nv)
        tau_arr = _require_vector("tau", tau, model.nv)
        return np.array(pin.aba(model, self._data, q_arr, v_arr, tau_arr))

    def sample(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        tau_applied: np.ndarray,
        contact_samples: Mapping[str, ContactSample] | None = None,
        *,
        time_s: float = 0.0,
    ) -> ForceTorqueFrame:
        """Return the overlay frame for the given state.

        Args:
            q: Configuration, shape ``(nq,)``.
            v: Velocity, shape ``(nv,)``.
            a: Actual acceleration, shape ``(nv,)``.
            tau_applied: Applied generalized torque, shape ``(nv,)``.
            contact_samples: Mapping of body (frame) name to its world-frame
                ``ContactSample``, passed through. Entries whose name is not
                a frame of the model are omitted, never relabelled.
            time_s: Frame timestamp [s].

        Raises:
            ValueError: On wrongly sized or non-finite inputs.
            TypeError: If ``contact_samples`` is not a mapping of
                ``ContactSample`` values.
        """
        model = self.model
        q_arr = _require_vector("q", q, model.nq)
        v_arr = _require_vector("v", v, model.nv)
        a_arr = _require_vector("a", a, model.nv)
        tau_arr = _require_vector("tau_applied", tau_applied, model.nv)
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")

        data = self._data
        pin.rnea(model, data, q_arr, v_arr, a_arr)
        pin.forwardKinematics(model, data, q_arr)

        wrenches: list[OverlayWrench] = []
        axial: dict[str, float | None] = {}
        for i in range(1, model.njoints):
            joint = model.joints[i]
            if _is_free_flyer(joint):
                continue
            wrenches.append(self._reaction(i))
            axial[self.body_name(i)] = self._axial_load(i, wrenches[-1])
            actuator = self._actuator(i, joint, tau_arr)
            if actuator is not None:
                wrenches.append(actuator)
        wrenches.extend(self._contacts(contact_samples))

        loads = (
            AxialLoadFrame(time_s=time_s, values_n=axial, source=_REACTION_SOURCE)
            if axial
            else None
        )
        return ForceTorqueFrame(
            time_s=time_s, engine=_ENGINE, wrenches=tuple(wrenches), axial_loads=loads
        )

    def _reaction(self, i: int) -> OverlayWrench:
        placement = self._data.oMi[i]
        spatial = self._data.f[i]
        rotation = placement.rotation
        return OverlayWrench(
            kind=WrenchKind.JOINT_REACTION,
            label=_label("reaction", self.model.names[i]),
            body=self.body_name(i),
            point_m=_vec3(placement.translation),
            force_n=_vec3(rotation @ spatial.linear),
            torque_nm=_vec3(rotation @ spatial.angular),
            source=_REACTION_SOURCE,
        )

    def _actuator(self, i: int, joint: object, tau: np.ndarray) -> OverlayWrench | None:
        placement = self._data.oMi[i]
        idx_v = joint.idx_v  # type: ignore[attr-defined]
        if "Spherical" in joint.shortname():  # type: ignore[attr-defined]
            moment = placement.rotation @ tau[idx_v : idx_v + 3]
        else:
            axis = _local_revolute_axis(joint)
            if axis is None:
                return None
            moment = float(tau[idx_v]) * (placement.rotation @ axis)
        return OverlayWrench(
            kind=WrenchKind.JOINT_ACTUATOR,
            label=_label("actuator", self.model.names[i]),
            body=self.body_name(i),
            point_m=_vec3(placement.translation),
            torque_nm=_vec3(moment),
            source=_ACTUATOR_SOURCE,
        )

    def _segment_distal_point(self, i: int) -> np.ndarray | None:
        """Distal end of body ``i`` in the world frame, or None if ambiguous.

        Exactly one child joint: that joint origin. No children: the body
        centre of mass. Several children (branching) is ambiguous and yields
        None rather than depending on joint ordering.
        """
        children = self._children.get(i, [])
        if len(children) == 1:
            return np.asarray(self._data.oMi[children[0]].translation)
        if children:
            return None
        lever = self.model.inertias[i].lever
        return np.asarray(self._data.oMi[i].act(lever))

    def _axial_load(self, i: int, reaction: OverlayWrench) -> float | None:
        proximal = np.asarray(self._data.oMi[i].translation)
        distal = self._segment_distal_point(i)
        if distal is None or np.linalg.norm(distal - proximal) <= _MIN_SEGMENT_LENGTH_M:
            return None  # ambiguous or degenerate axis: unavailable, never guessed
        assert reaction.force_n is not None  # reactions always carry both halves
        return axial_force_from_proximal_reaction(reaction.force_n, proximal, distal)

    def _contacts(
        self, samples: Mapping[str, ContactSample] | None
    ) -> list[OverlayWrench]:
        if samples is None:
            return []
        if not isinstance(samples, Mapping):
            raise TypeError("contact_samples must be a mapping of body name to sample")
        wrenches = []
        for body, sample in samples.items():
            if not isinstance(sample, ContactSample):
                raise TypeError("contact_samples must contain ContactSample items")
            if not self.model.existFrame(str(body)):
                logger.warning("Omitting contact on unknown body %r", body)
                continue
            total = np.asarray(sample.normal_force_n) + np.asarray(
                sample.friction_force_n
            )
            wrenches.append(
                OverlayWrench(
                    kind=WrenchKind.CONTACT,
                    label=_label("contact", str(body)),
                    body=str(body),
                    point_m=_vec3(sample.contact_point_m),
                    force_n=_vec3(total),
                    source=_CONTACT_SOURCE,
                )
            )
        return wrenches
