"""Event-aligned impulse and work attribution for articulated trajectories.

This module integrates already-evaluated generalized-force contributions. It
does not identify anatomical force sources and it does not turn descriptive
same-trajectory attribution into a causal forward counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ForwardAttribution:
    """Integrated continuous and impulsive generalized contributions."""

    contribution_names: tuple[str, ...]
    continuous_impulses: FloatArray
    generalized_work_j: FloatArray
    transport_impulse: FloatArray
    total_event_impulse: FloatArray
    total_event_work_j: float
    momentum_change: FloatArray
    kinetic_energy_change_j: float
    momentum_closure_residual: float
    work_closure_residual_j: float

    @property
    def continuous_work_j(self) -> float:
        """Return total continuous generalized work across contributions."""

        return float(np.sum(self.generalized_work_j))


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validate_time_and_segments(
    time_s: Any, segment_ids: Any
) -> tuple[FloatArray, IntArray]:
    time = np.asarray(time_s, dtype=np.float64)
    segments = np.asarray(segment_ids, dtype=np.int64)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise ValueError(
            "time_s must be a finite one-dimensional array with at least two samples"
        )
    if segments.shape != time.shape:
        raise ValueError("segment_ids must have the same shape as time_s")
    differences = np.diff(time)
    if np.any(differences < 0.0):
        raise ValueError("time_s must be nondecreasing")
    transitions = segments[1:] != segments[:-1]
    if np.any((differences == 0.0) & ~transitions):
        raise ValueError("duplicate times require a segment transition")
    if np.any((differences > 0.0) & transitions):
        raise ValueError("segment transitions require duplicate pre/post event times")
    if np.any((differences > 0.0) & (segments[1:] < segments[:-1])):
        raise ValueError("segment_ids must be nondecreasing")
    return time, segments


def _trapezoid(
    values: FloatArray, time: FloatArray, continuous: NDArray[np.bool_]
) -> FloatArray:
    widths = np.diff(time)[continuous]
    averages = 0.5 * (values[:-1][continuous] + values[1:][continuous])
    return np.tensordot(widths, averages, axes=(0, 0))


def integrate_forward_attribution(
    *,
    time_s: FloatArray,
    mass_matrices: FloatArray,
    mass_matrix_rates: FloatArray,
    velocities: FloatArray,
    generalized_forces: FloatArray,
    contribution_names: tuple[str, ...],
    segment_ids: IntArray,
    event_impulses: FloatArray,
    event_work_j: FloatArray,
) -> ForwardAttribution:
    """Integrate generalized impulse and work without crossing events.

    Event boundaries are represented by duplicate pre/post times with a change
    in ``segment_ids``. ``mass_matrix_rates`` is the independently evaluated
    material rate of the generalized mass matrix. The registered momentum
    identity is

    ``delta(M v) = integral(sum(Q) + Mdot v) dt + sum(event impulse)``.

    Postcondition: returned continuous terms never blend pre- and post-event
    samples. Residuals remain visible; this function does not silently repair
    a non-closing trace.
    """

    time, segments = _validate_time_and_segments(time_s, segment_ids)
    sample_count = time.size
    velocity = np.asarray(velocities, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] != sample_count:
        raise ValueError("velocities must have shape (samples, coordinates)")
    coordinate_count = velocity.shape[1]
    velocity = _finite_array("velocities", velocity, (sample_count, coordinate_count))
    mass = _finite_array(
        "mass_matrices",
        mass_matrices,
        (sample_count, coordinate_count, coordinate_count),
    )
    mass_rate = _finite_array(
        "mass_matrix_rates",
        mass_matrix_rates,
        (sample_count, coordinate_count, coordinate_count),
    )
    if not np.allclose(mass, np.swapaxes(mass, 1, 2), rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("mass_matrices must be symmetric")
    if any(np.linalg.eigvalsh(matrix)[0] <= 0.0 for matrix in mass):
        raise ValueError("mass_matrices must be positive definite")
    if not np.allclose(
        mass_rate, np.swapaxes(mass_rate, 1, 2), rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("mass_matrix_rates must be symmetric")

    forces = np.asarray(generalized_forces, dtype=np.float64)
    expected_force_shape = (sample_count, len(contribution_names), coordinate_count)
    forces = _finite_array("generalized_forces", forces, expected_force_shape)
    if not contribution_names or len(set(contribution_names)) != len(
        contribution_names
    ):
        raise ValueError("contribution_names must be nonempty and unique")
    if any(not name.strip() for name in contribution_names):
        raise ValueError("contribution_names must not contain blank names")

    transitions = segments[1:] != segments[:-1]
    event_count = int(np.count_nonzero(transitions))
    raw_impulses = np.asarray(event_impulses, dtype=np.float64)
    raw_event_work = np.asarray(event_work_j, dtype=np.float64)
    if raw_impulses.shape != (event_count, coordinate_count):
        raise ValueError("event_impulses must contain one row per segment transition")
    if raw_event_work.shape != (event_count,):
        raise ValueError("event_work_j must contain one value per segment transition")
    impulses = _finite_array(
        "event_impulses", raw_impulses, (event_count, coordinate_count)
    )
    event_work = _finite_array("event_work_j", raw_event_work, (event_count,))
    continuous = ~transitions

    continuous_impulses = np.moveaxis(_trapezoid(forces, time, continuous), 0, 0)
    power = np.einsum("skd,sd->sk", forces, velocity)
    generalized_work = _trapezoid(power, time, continuous)
    transport_rate = np.einsum("sij,sj->si", mass_rate, velocity)
    transport_impulse = _trapezoid(transport_rate, time, continuous)

    momentum = np.einsum("sij,sj->si", mass, velocity)
    momentum_change = momentum[-1] - momentum[0]
    total_event_impulse = np.sum(impulses, axis=0)
    reconstructed_momentum_change = (
        np.sum(continuous_impulses, axis=0) + transport_impulse + total_event_impulse
    )
    momentum_residual = float(
        np.linalg.norm(momentum_change - reconstructed_momentum_change)
    )

    kinetic_energy = 0.5 * np.einsum("si,sij,sj->s", velocity, mass, velocity)
    kinetic_energy_change = float(kinetic_energy[-1] - kinetic_energy[0])
    total_event_work = float(np.sum(event_work))
    work_residual = float(
        abs(kinetic_energy_change - np.sum(generalized_work) - total_event_work)
    )
    return ForwardAttribution(
        contribution_names=contribution_names,
        continuous_impulses=np.asarray(continuous_impulses, dtype=np.float64),
        generalized_work_j=np.asarray(generalized_work, dtype=np.float64),
        transport_impulse=np.asarray(transport_impulse, dtype=np.float64),
        total_event_impulse=np.asarray(total_event_impulse, dtype=np.float64),
        total_event_work_j=total_event_work,
        momentum_change=np.asarray(momentum_change, dtype=np.float64),
        kinetic_energy_change_j=kinetic_energy_change,
        momentum_closure_residual=momentum_residual,
        work_closure_residual_j=work_residual,
    )


def scale_forward_attribution_inputs(
    *,
    time_s: FloatArray,
    mass_matrices: FloatArray,
    mass_matrix_rates: FloatArray,
    velocities: FloatArray,
    generalized_forces: FloatArray,
    contribution_names: tuple[str, ...],
    segment_ids: IntArray,
    event_impulses: FloatArray,
    event_work_j: FloatArray,
    coordinate_scale: FloatArray,
) -> dict[str, Any]:
    """Represent the same trajectory under ``q_scaled = S q``."""

    velocity = np.asarray(velocities, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError("velocities must have shape (samples, coordinates)")
    coordinate_count = velocity.shape[1]
    scale = _finite_array("coordinate_scale", coordinate_scale, (coordinate_count,))
    if np.any(scale <= 0.0):
        raise ValueError("coordinate_scale must be positive")
    inverse = np.diag(1.0 / scale)
    mass = np.asarray(mass_matrices, dtype=np.float64)
    mass_rate = np.asarray(mass_matrix_rates, dtype=np.float64)
    forces = np.asarray(generalized_forces, dtype=np.float64)
    impulses = np.asarray(event_impulses, dtype=np.float64)
    return {
        "time_s": np.asarray(time_s, dtype=np.float64),
        "mass_matrices": np.einsum("ij,sjk,kl->sil", inverse, mass, inverse),
        "mass_matrix_rates": np.einsum("ij,sjk,kl->sil", inverse, mass_rate, inverse),
        "velocities": velocity * scale,
        "generalized_forces": np.einsum("ij,skj->ski", inverse, forces),
        "contribution_names": contribution_names,
        "segment_ids": np.asarray(segment_ids, dtype=np.int64),
        "event_impulses": np.einsum("ij,ej->ei", inverse, impulses),
        "event_work_j": np.asarray(event_work_j, dtype=np.float64),
    }


__all__ = [
    "ForwardAttribution",
    "integrate_forward_attribution",
    "scale_forward_attribution_inputs",
]
