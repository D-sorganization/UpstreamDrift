"""Differentiate shared ground forces in Pinocchio tangent coordinates (#10255)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.contact_derivatives import (
    ContactDerivatives,
    sphere_ground_contact_derivatives,
)
from src.shared.python.motion_matching.contact_law import ContactParameters, GroundPlane

Array = NDArray[np.float64]


class ContactEffortDerivatives(NamedTuple):
    """Generalized contact-force derivatives in Pinocchio tangent order."""

    dq: Array
    dv: Array
    differentiable: bool


class _FrameDerivatives(NamedTuple):
    frame_id: int
    jacobian: Array
    force: Array
    contact: ContactDerivatives


class PinocchioContactDifferentiator:
    """Own scratch kinematics so constrained-dynamics caches remain untouched."""

    def __init__(self, pin: Any, model: Any, frames: Mapping[int, float]) -> None:
        if any(index < 0 or index >= model.nframes for index in frames):
            raise ValueError("Contact frame indices must belong to the model")
        if any(not np.isfinite(radius) or radius <= 0 for radius in frames.values()):
            raise ValueError("Contact sphere radii must be finite and positive")
        self._pin = pin
        self._model = model
        self._frames = dict(frames)
        self._data = model.createData()

    def evaluate(
        self, q: Array, v: Array, ground: GroundPlane, parameters: ContactParameters
    ) -> ContactEffortDerivatives:
        """Return owned derivatives; report nonsmooth contact branches explicitly.

        Preconditions: finite model-sized configuration and tangent velocity.
        Postconditions: square finite matrices in model tangent order, read-only.
        """
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        if q.shape != (self._model.nq,) or v.shape != (self._model.nv,):
            raise ValueError("Contact state dimensions must match the model")
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("Contact state must be finite")
        states = self._frame_derivatives(q, v, ground, parameters)
        dq = self._configuration_derivative(q, v, states)
        dv = np.zeros_like(dq)
        for state in states:
            contact = state.contact
            dv += state.jacobian.T @ contact.dforce_dvelocity @ state.jacobian
        for matrix in (dq, dv):
            if not np.isfinite(matrix).all():
                raise ValueError("Contact effort derivatives must be finite")
            matrix.setflags(write=False)
        return ContactEffortDerivatives(
            dq, dv, all(state.contact.differentiable for state in states)
        )

    def _frame_derivatives(
        self, q: Array, v: Array, ground: GroundPlane, parameters: ContactParameters
    ) -> list[_FrameDerivatives]:
        pin, model, data = self._pin, self._model, self._data
        pin.computeJointJacobians(model, data, q)
        pin.forwardKinematics(model, data, q, v)
        pin.updateFramePlacements(model, data)
        reference = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        states = []
        for frame_id, radius in self._frames.items():
            placement = data.oMf[frame_id]
            velocity = pin.getFrameVelocity(model, data, frame_id, reference)
            contact = sphere_ground_contact_derivatives(
                placement.translation, velocity.linear, radius, ground, parameters
            )
            jacobian = np.array(
                pin.getFrameJacobian(model, data, frame_id, reference)[:3], copy=True
            )
            sample = contact.sample
            force = sample.normal_force_n + sample.friction_force_n
            states.append(_FrameDerivatives(frame_id, jacobian, force, contact))
        return states

    def _configuration_derivative(
        self, q: Array, v: Array, states: list[_FrameDerivatives]
    ) -> Array:
        pin, model, data = self._pin, self._model, self._data
        result = np.zeros((model.nv, model.nv))
        active = [state for state in states if np.any(state.force)]
        if not active:
            return result
        reference = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        for column in range(model.nv):
            direction = np.zeros(model.nv)
            direction[column] = 1.0
            # Jdot(q,e_k) is the exact directional derivative of J along e_k.
            pin.computeJointJacobiansTimeVariation(model, data, q, direction)
            pin.updateFramePlacements(model, data)
            for state in active:
                jdot = np.asarray(
                    pin.getFrameJacobianTimeVariation(
                        model, data, state.frame_id, reference
                    )[:3]
                )
                contact = state.contact
                force_derivative = contact.dforce_dcenter @ state.jacobian[
                    :, column
                ] + contact.dforce_dvelocity @ (jdot @ v)
                result[:, column] += (
                    jdot.T @ state.force + state.jacobian.T @ force_derivative
                )
        return result
