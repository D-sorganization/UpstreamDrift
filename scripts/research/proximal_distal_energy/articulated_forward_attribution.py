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


@dataclass(frozen=True, slots=True)
class ForwardAttributionInputs:
    """Complete typed input contract for one event-aligned attribution."""

    time_s: FloatArray
    mass_matrices: FloatArray
    mass_matrix_rates: FloatArray
    velocities: FloatArray
    generalized_forces: FloatArray
    contribution_names: tuple[str, ...]
    segment_ids: IntArray
    event_impulses: FloatArray
    event_work_j: FloatArray
    share_denominator_floor: float = 1.0e-12


@dataclass(frozen=True, slots=True)
class _ValidatedInputs:
    time: FloatArray
    mass: FloatArray
    mass_rate: FloatArray
    velocity: FloatArray
    forces: FloatArray
    contribution_names: tuple[str, ...]
    segments: IntArray
    impulses: FloatArray
    event_work: FloatArray
    continuous: NDArray[np.bool_]
    denominator_floor: float


@dataclass(frozen=True, slots=True)
class _IntegratedComponents:
    continuous_impulses: FloatArray
    generalized_work: FloatArray
    kinetic_transport_work: float
    transport_impulse: FloatArray
    total_event_impulse: FloatArray
    total_event_work: float
    momentum_change: FloatArray
    kinetic_energy_change: float
    momentum_residual: float
    work_residual: float


@dataclass(frozen=True, slots=True)
class _ImpulseSummary:
    reference: float
    relative_residual: float
    component_names: tuple[str, ...]
    shares: FloatArray
    adequacy: NDArray[np.bool_]
    cancellation: FloatArray


@dataclass(frozen=True, slots=True)
class _WorkSummary:
    reference: float
    relative_residual: float
    component_names: tuple[str, ...]
    shares: FloatArray
    adequate: bool
    cancellation: float


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


def _validate_forward_inputs(inputs: ForwardAttributionInputs) -> _ValidatedInputs:
    if not isinstance(inputs, ForwardAttributionInputs):
        raise TypeError("inputs must be a ForwardAttributionInputs")
    time, segments = _validate_time_and_segments(inputs.time_s, inputs.segment_ids)
    sample_count = time.size
    velocity = np.asarray(inputs.velocities, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] != sample_count:
        raise ValueError("velocities must have shape (samples, coordinates)")
    coordinate_count = velocity.shape[1]
    velocity = _finite_array("velocities", velocity, velocity.shape)
    mass = _finite_array(
        "mass_matrices",
        inputs.mass_matrices,
        (sample_count, coordinate_count, coordinate_count),
    )
    mass_rate = _finite_array(
        "mass_matrix_rates",
        inputs.mass_matrix_rates,
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
    expected_force_shape = (
        sample_count,
        len(inputs.contribution_names),
        coordinate_count,
    )
    forces = _finite_array(
        "generalized_forces", inputs.generalized_forces, expected_force_shape
    )
    if not inputs.contribution_names or len(set(inputs.contribution_names)) != len(
        inputs.contribution_names
    ):
        raise ValueError("contribution_names must be nonempty and unique")
    if any(not name.strip() for name in inputs.contribution_names):
        raise ValueError("contribution_names must not contain blank names")
    floor = inputs.share_denominator_floor
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("share_denominator_floor must be finite and positive")
    transitions = segments[1:] != segments[:-1]
    event_count = int(np.count_nonzero(transitions))
    raw_impulses = np.asarray(inputs.event_impulses, dtype=np.float64)
    raw_event_work = np.asarray(inputs.event_work_j, dtype=np.float64)
    if raw_impulses.shape != (event_count, coordinate_count):
        raise ValueError("event_impulses must contain one row per segment transition")
    if raw_event_work.shape != (event_count,):
        raise ValueError("event_work_j must contain one value per segment transition")
    impulses = _finite_array(
        "event_impulses", raw_impulses, (event_count, coordinate_count)
    )
    event_work = _finite_array("event_work_j", raw_event_work, (event_count,))
    return _ValidatedInputs(
        time=time,
        mass=mass,
        mass_rate=mass_rate,
        velocity=velocity,
        forces=forces,
        contribution_names=inputs.contribution_names,
        segments=segments,
        impulses=impulses,
        event_work=event_work,
        continuous=~transitions,
        denominator_floor=floor,
    )


def _integrate_components(inputs: _ValidatedInputs) -> _IntegratedComponents:
    continuous_impulses = np.moveaxis(
        _trapezoid(inputs.forces, inputs.time, inputs.continuous), 0, 0
    )
    power = np.einsum("skd,sd->sk", inputs.forces, inputs.velocity)
    generalized_work = _trapezoid(power, inputs.time, inputs.continuous)
    transport_rate = np.einsum("sij,sj->si", inputs.mass_rate, inputs.velocity)
    transport_impulse = _trapezoid(transport_rate, inputs.time, inputs.continuous)
    kinetic_transport_power = 0.5 * np.einsum(
        "si,sij,sj->s", inputs.velocity, inputs.mass_rate, inputs.velocity
    )
    kinetic_transport_work = float(
        _trapezoid(kinetic_transport_power, inputs.time, inputs.continuous)
    )
    momentum = np.einsum("sij,sj->si", inputs.mass, inputs.velocity)
    momentum_change = momentum[-1] - momentum[0]
    total_event_impulse = np.sum(inputs.impulses, axis=0)
    reconstructed_momentum_change = (
        np.sum(continuous_impulses, axis=0) + transport_impulse + total_event_impulse
    )
    momentum_residual = float(
        np.linalg.norm(momentum_change - reconstructed_momentum_change)
    )
    kinetic_energy = 0.5 * np.einsum(
        "si,sij,sj->s", inputs.velocity, inputs.mass, inputs.velocity
    )
    kinetic_energy_change = float(kinetic_energy[-1] - kinetic_energy[0])
    total_event_work = float(np.sum(inputs.event_work))
    work_residual = float(
        abs(
            kinetic_energy_change
            - np.sum(generalized_work)
            - kinetic_transport_work
            - total_event_work
        )
    )
    return _IntegratedComponents(
        continuous_impulses=np.asarray(continuous_impulses, dtype=np.float64),
        generalized_work=np.asarray(generalized_work, dtype=np.float64),
        kinetic_transport_work=kinetic_transport_work,
        transport_impulse=np.asarray(transport_impulse, dtype=np.float64),
        total_event_impulse=np.asarray(total_event_impulse, dtype=np.float64),
        total_event_work=total_event_work,
        momentum_change=np.asarray(momentum_change, dtype=np.float64),
        kinetic_energy_change=kinetic_energy_change,
        momentum_residual=momentum_residual,
        work_residual=work_residual,
    )


def _summarize_impulse(
    inputs: _ValidatedInputs, components: _IntegratedComponents
) -> _ImpulseSummary:
    impulse_components = np.concatenate(
        (
            components.continuous_impulses,
            components.transport_impulse[None, :],
            components.total_event_impulse[None, :],
        ),
        axis=0,
    )
    reference = max(
        float(np.linalg.norm(components.momentum_change)),
        float(np.sum(np.linalg.norm(impulse_components, axis=1))),
        inputs.denominator_floor,
    )
    adequacy = np.abs(components.momentum_change) >= inputs.denominator_floor
    shares = np.full(impulse_components.shape, np.nan)
    cancellation = np.full(components.momentum_change.shape, np.nan)
    shares[:, adequacy] = (
        impulse_components[:, adequacy] / components.momentum_change[adequacy]
    )
    cancellation[adequacy] = np.sum(
        np.abs(impulse_components[:, adequacy]), axis=0
    ) / np.abs(components.momentum_change[adequacy])
    return _ImpulseSummary(
        reference=reference,
        relative_residual=components.momentum_residual / reference,
        component_names=inputs.contribution_names + ("mass_transport", "event"),
        shares=shares,
        adequacy=adequacy,
        cancellation=cancellation,
    )


def _summarize_work(
    inputs: _ValidatedInputs, components: _IntegratedComponents
) -> _WorkSummary:
    work_components = np.concatenate(
        (
            components.generalized_work,
            np.array([components.kinetic_transport_work, components.total_event_work]),
        )
    )
    reference = max(
        abs(components.kinetic_energy_change),
        float(np.sum(np.abs(work_components))),
        inputs.denominator_floor,
    )
    adequate = abs(components.kinetic_energy_change) >= inputs.denominator_floor
    shares = (
        work_components / components.kinetic_energy_change
        if adequate
        else np.full(work_components.shape, np.nan)
    )
    cancellation = (
        float(np.sum(np.abs(work_components)) / abs(components.kinetic_energy_change))
        if adequate
        else float("nan")
    )
    return _WorkSummary(
        reference=reference,
        relative_residual=components.work_residual / reference,
        component_names=inputs.contribution_names + ("kinetic_transport", "event"),
        shares=shares,
        adequate=adequate,
        cancellation=cancellation,
    )


def integrate_forward_attribution(
    inputs: ForwardAttributionInputs,
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

    validated = _validate_forward_inputs(inputs)
    components = _integrate_components(validated)
    impulse = _summarize_impulse(validated, components)
    work = _summarize_work(validated, components)
    return ForwardAttribution(
        contribution_names=validated.contribution_names,
        continuous_impulses=components.continuous_impulses,
        generalized_work_j=components.generalized_work,
        kinetic_transport_work_j=components.kinetic_transport_work,
        transport_impulse=components.transport_impulse,
        total_event_impulse=components.total_event_impulse,
        total_event_work_j=components.total_event_work,
        momentum_change=components.momentum_change,
        kinetic_energy_change_j=components.kinetic_energy_change,
        momentum_closure_residual=components.momentum_residual,
        momentum_reference_norm=impulse.reference,
        momentum_closure_relative_residual=impulse.relative_residual,
        work_closure_residual_j=components.work_residual,
        work_reference_j=work.reference,
        work_closure_relative_residual=work.relative_residual,
        impulse_component_names=impulse.component_names,
        impulse_shares=impulse.shares,
        impulse_share_adequacy=impulse.adequacy,
        impulse_cancellation_indices=impulse.cancellation,
        work_component_names=work.component_names,
        work_shares=work.shares,
        work_share_adequate=work.adequate,
        work_cancellation_index=work.cancellation,
    )


def scale_forward_attribution_inputs(
    inputs: ForwardAttributionInputs,
    coordinate_scale: FloatArray,
) -> ForwardAttributionInputs:
    """Represent the same trajectory under ``q_scaled = S q``."""

    validated = _validate_forward_inputs(inputs)
    coordinate_count = validated.velocity.shape[1]
    scale = _finite_array("coordinate_scale", coordinate_scale, (coordinate_count,))
    if np.any(scale <= 0.0):
        raise ValueError("coordinate_scale must be positive")
    inverse = np.diag(1.0 / scale)
    return ForwardAttributionInputs(
        time_s=validated.time,
        mass_matrices=np.einsum("ij,sjk,kl->sil", inverse, validated.mass, inverse),
        mass_matrix_rates=np.einsum(
            "ij,sjk,kl->sil", inverse, validated.mass_rate, inverse
        ),
        velocities=validated.velocity * scale,
        generalized_forces=np.einsum("ij,skj->ski", inverse, validated.forces),
        contribution_names=validated.contribution_names,
        segment_ids=validated.segments,
        event_impulses=np.einsum("ij,ej->ei", inverse, validated.impulses),
        event_work_j=validated.event_work,
        share_denominator_floor=validated.denominator_floor,
    )


__all__ = [
    "ForwardAttribution",
    "ForwardAttributionInputs",
    "differentiate_mass_along_velocity",
    "differentiate_mass_matrices",
    "integrate_forward_attribution",
    "require_forward_attribution_closure",
    "scale_forward_attribution_inputs",
]
