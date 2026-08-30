"""Event-aligned impulse and work attribution for articulated trajectories.

This module integrates already-evaluated generalized-force contributions. It
does not identify anatomical force sources and it does not turn descriptive
same-trajectory attribution into a causal forward counterfactual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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
    kinetic_transport_work_j: float
    transport_impulse: FloatArray
    total_event_impulse: FloatArray
    total_event_work_j: float
    momentum_change: FloatArray
    kinetic_energy_change_j: float
    momentum_closure_residual: float
    momentum_reference_norm: float
    momentum_closure_relative_residual: float
    work_closure_residual_j: float
    work_reference_j: float
    work_closure_relative_residual: float
    impulse_component_names: tuple[str, ...]
    impulse_shares: FloatArray
    impulse_share_adequacy: NDArray[np.bool_]
    impulse_cancellation_indices: FloatArray
    work_component_names: tuple[str, ...]
    work_shares: FloatArray
    work_share_adequate: bool
    work_cancellation_index: float

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


def differentiate_mass_matrices(
    *,
    time_s: FloatArray,
    mass_matrices: FloatArray,
    segment_ids: IntArray,
) -> FloatArray:
    """Differentiate mass matrices without crossing registered events."""

    time, segments = _validate_time_and_segments(time_s, segment_ids)
    mass = np.asarray(mass_matrices, dtype=np.float64)
    if mass.ndim != 3 or mass.shape[0] != time.size or mass.shape[1] != mass.shape[2]:
        raise ValueError(
            "mass_matrices must have shape (samples, coordinates, coordinates)"
        )
    mass = _finite_array("mass_matrices", mass, mass.shape)
    rates = np.empty_like(mass)
    for segment in np.unique(segments):
        indices = np.flatnonzero(segments == segment)
        if indices.size < 2 or np.any(np.diff(indices) != 1):
            raise ValueError(
                "each segment must contain at least two contiguous samples"
            )
        segment_time = time[indices]
        if np.any(np.diff(segment_time) <= 0.0):
            raise ValueError("time must increase strictly within each segment")
        edge_order: Literal[1, 2] = 2 if indices.size >= 3 else 1
        rates[indices] = np.gradient(
            mass[indices], segment_time, axis=0, edge_order=edge_order
        )
    return rates


def differentiate_mass_along_velocity(
    *,
    positions: FloatArray,
    velocities: FloatArray,
    mass_evaluator: Callable[[FloatArray], FloatArray],
    directional_step_s: float = 1.0e-6,
) -> FloatArray:
    """Evaluate ``Mdot = dM/dq qdot`` by a centered directional derivative."""

    q = np.asarray(positions, dtype=np.float64)
    qd = np.asarray(velocities, dtype=np.float64)
    if q.ndim != 2 or q.shape != qd.shape or not np.all(np.isfinite(q)):
        raise ValueError("positions and velocities must share one finite 2-D shape")
    if not np.all(np.isfinite(qd)):
        raise ValueError("positions and velocities must share one finite 2-D shape")
    if not callable(mass_evaluator):
        raise TypeError("mass_evaluator must be callable")
    if not np.isfinite(directional_step_s) or directional_step_s <= 0.0:
        raise ValueError("directional_step_s must be finite and positive")
    sample_count, coordinate_count = q.shape
    rates = np.empty((sample_count, coordinate_count, coordinate_count))
    for index in range(sample_count):
        offset = directional_step_s * qd[index]
        forward = _finite_array(
            "forward mass matrix",
            mass_evaluator(q[index] + offset),
            (coordinate_count, coordinate_count),
        )
        backward = _finite_array(
            "backward mass matrix",
            mass_evaluator(q[index] - offset),
            (coordinate_count, coordinate_count),
        )
        rates[index] = (forward - backward) / (2.0 * directional_step_s)
    if not np.allclose(rates, np.swapaxes(rates, 1, 2), rtol=1.0e-8, atol=1.0e-10):
        raise ValueError("directional mass-matrix rates must be symmetric")
    return rates


def require_forward_attribution_closure(
    result: ForwardAttribution,
    *,
    momentum_tolerance: float,
    work_tolerance_j: float,
) -> None:
    """Fail closed when a registered integrated balance exceeds tolerance."""

    if not isinstance(result, ForwardAttribution):
        raise TypeError("result must be a ForwardAttribution")
    for name, value in (
        ("momentum_tolerance", momentum_tolerance),
        ("work_tolerance_j", work_tolerance_j),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if result.momentum_closure_residual > momentum_tolerance:
        raise ValueError(
            "momentum closure residual exceeds the registered tolerance: "
            f"{result.momentum_closure_residual:.6e} > {momentum_tolerance:.6e}"
        )
    if result.work_closure_residual_j > work_tolerance_j:
        raise ValueError(
            "work closure residual exceeds the registered tolerance: "
            f"{result.work_closure_residual_j:.6e} > {work_tolerance_j:.6e}"
        )


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
    share_denominator_floor: float = 1.0e-12,
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
    if not np.isfinite(share_denominator_floor) or share_denominator_floor <= 0.0:
        raise ValueError("share_denominator_floor must be finite and positive")

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
    kinetic_transport_power = 0.5 * np.einsum(
        "si,sij,sj->s", velocity, mass_rate, velocity
    )
    kinetic_transport_work = float(
        _trapezoid(kinetic_transport_power, time, continuous)
    )

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
        abs(
            kinetic_energy_change
            - np.sum(generalized_work)
            - kinetic_transport_work
            - total_event_work
        )
    )
    impulse_components = np.concatenate(
        (continuous_impulses, transport_impulse[None, :], total_event_impulse[None, :]),
        axis=0,
    )
    momentum_reference = max(
        float(np.linalg.norm(momentum_change)),
        float(np.sum(np.linalg.norm(impulse_components, axis=1))),
        share_denominator_floor,
    )
    momentum_relative_residual = momentum_residual / momentum_reference
    impulse_adequacy = np.abs(momentum_change) >= share_denominator_floor
    impulse_shares = np.full(impulse_components.shape, np.nan)
    impulse_cancellation = np.full(coordinate_count, np.nan)
    impulse_shares[:, impulse_adequacy] = (
        impulse_components[:, impulse_adequacy] / momentum_change[impulse_adequacy]
    )
    impulse_cancellation[impulse_adequacy] = np.sum(
        np.abs(impulse_components[:, impulse_adequacy]), axis=0
    ) / np.abs(momentum_change[impulse_adequacy])
    work_components = np.concatenate(
        (
            generalized_work,
            np.array([kinetic_transport_work, total_event_work]),
        )
    )
    work_reference = max(
        abs(kinetic_energy_change),
        float(np.sum(np.abs(work_components))),
        share_denominator_floor,
    )
    work_relative_residual = work_residual / work_reference
    work_adequate = abs(kinetic_energy_change) >= share_denominator_floor
    work_shares = (
        work_components / kinetic_energy_change
        if work_adequate
        else np.full(work_components.shape, np.nan)
    )
    work_cancellation = (
        float(np.sum(np.abs(work_components)) / abs(kinetic_energy_change))
        if work_adequate
        else float("nan")
    )
    return ForwardAttribution(
        contribution_names=contribution_names,
        continuous_impulses=np.asarray(continuous_impulses, dtype=np.float64),
        generalized_work_j=np.asarray(generalized_work, dtype=np.float64),
        kinetic_transport_work_j=kinetic_transport_work,
        transport_impulse=np.asarray(transport_impulse, dtype=np.float64),
        total_event_impulse=np.asarray(total_event_impulse, dtype=np.float64),
        total_event_work_j=total_event_work,
        momentum_change=np.asarray(momentum_change, dtype=np.float64),
        kinetic_energy_change_j=kinetic_energy_change,
        momentum_closure_residual=momentum_residual,
        momentum_reference_norm=momentum_reference,
        momentum_closure_relative_residual=momentum_relative_residual,
        work_closure_residual_j=work_residual,
        work_reference_j=work_reference,
        work_closure_relative_residual=work_relative_residual,
        impulse_component_names=contribution_names + ("mass_transport", "event"),
        impulse_shares=impulse_shares,
        impulse_share_adequacy=impulse_adequacy,
        impulse_cancellation_indices=impulse_cancellation,
        work_component_names=contribution_names + ("kinetic_transport", "event"),
        work_shares=work_shares,
        work_share_adequate=work_adequate,
        work_cancellation_index=work_cancellation,
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
    share_denominator_floor: float = 1.0e-12,
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
        "share_denominator_floor": share_denominator_floor,
    }


__all__ = [
    "ForwardAttribution",
    "differentiate_mass_along_velocity",
    "differentiate_mass_matrices",
    "integrate_forward_attribution",
    "require_forward_attribution_closure",
    "scale_forward_attribution_inputs",
]
