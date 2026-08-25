"""Higher-order Euler--Bernoulli reference for the reduced shaft model.

The study reuses the repository's canonical finite-element shaft assembly,
adds declared clubhead inertia at the free end, identifies two synthetic
parameters from two modal frequencies, and compares a one-mode reduction with
the converged distributed model under the same loads.  Synthetic identification
tests the inference path; it is not equipment calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from scipy.optimize import least_squares

from src.shared.python.physics.flexible_shaft import (
    FiniteElementShaftModel,
    create_standard_shaft,
)


FloatArray = npt.NDArray[np.float64]


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class BeamReferenceConfig:
    """Geometry, inertia, mesh, damping, and excitation contract."""

    length_m: float
    butt_diameter_m: float
    tip_diameter_m: float
    wall_thickness_m: float
    density_kg_m3: float
    youngs_modulus_pa: float
    head_mass_kg: float
    head_rotary_inertia_kg_m2: float
    element_count: int
    mode_count: int
    damping_ratio: float
    duration_s: float
    step_s: float
    low_frequency_pulse_s: float
    high_frequency_pulse_s: float
    peak_tip_force_n: float
    peak_tip_moment_nm: float

    @classmethod
    def publication_default(cls) -> BeamReferenceConfig:
        """Return the declared synthetic structural-comparison case."""
        return cls(
            length_m=1.0,
            butt_diameter_m=0.015,
            tip_diameter_m=0.0085,
            wall_thickness_m=0.001,
            density_kg_m3=1600.0,
            youngs_modulus_pa=112.0e9,
            head_mass_kg=0.205,
            head_rotary_inertia_kg_m2=4.8e-4,
            element_count=24,
            mode_count=6,
            damping_ratio=0.018,
            duration_s=0.18,
            step_s=0.0000125,
            low_frequency_pulse_s=0.080,
            high_frequency_pulse_s=0.004,
            peak_tip_force_n=8.0,
            peak_tip_moment_nm=4.0,
        )

    def __post_init__(self) -> None:
        for name in (
            "length_m",
            "butt_diameter_m",
            "tip_diameter_m",
            "wall_thickness_m",
            "density_kg_m3",
            "youngs_modulus_pa",
            "head_mass_kg",
            "head_rotary_inertia_kg_m2",
            "duration_s",
            "step_s",
            "low_frequency_pulse_s",
            "high_frequency_pulse_s",
            "peak_tip_force_n",
            "peak_tip_moment_nm",
        ):
            _positive(name, float(getattr(self, name)))
        if self.element_count < 2:
            raise ValueError("element_count must be at least two")
        if self.mode_count < 1 or self.mode_count > 2 * self.element_count:
            raise ValueError("mode_count must be within the free beam dimension")
        if not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must be finite and in [0, 1)")
        if self.tip_diameter_m > self.butt_diameter_m:
            raise ValueError("tip diameter must not exceed butt diameter")
        if self.wall_thickness_m * 2.0 >= self.tip_diameter_m:
            raise ValueError("wall thickness leaves no hollow section")
        if self.step_s >= min(self.high_frequency_pulse_s, self.low_frequency_pulse_s):
            raise ValueError("step_s must resolve both excitation pulses")
        count = self.duration_s / self.step_s
        if not np.isclose(count, round(count), atol=1e-10, rtol=0.0):
            raise ValueError("duration_s must be an integer multiple of step_s")


@dataclass(frozen=True, slots=True)
class BeamIdentificationConfig:
    """Synthetic modal-identification and local-uncertainty contract."""

    declared_truth_youngs_modulus_pa: float
    declared_truth_head_mass_kg: float
    initial_youngs_modulus_pa: float
    initial_head_mass_kg: float
    frequency_sigma_hz: float
    element_count: int

    @classmethod
    def publication_default(cls) -> BeamIdentificationConfig:
        return cls(
            declared_truth_youngs_modulus_pa=112.0e9,
            declared_truth_head_mass_kg=0.205,
            initial_youngs_modulus_pa=85.0e9,
            initial_head_mass_kg=0.16,
            frequency_sigma_hz=0.05,
            element_count=24,
        )

    def __post_init__(self) -> None:
        for name in (
            "declared_truth_youngs_modulus_pa",
            "declared_truth_head_mass_kg",
            "initial_youngs_modulus_pa",
            "initial_head_mass_kg",
            "frequency_sigma_hz",
        ):
            _positive(name, float(getattr(self, name)))
        if self.element_count < 2:
            raise ValueError("element_count must be at least two")


@dataclass(frozen=True, slots=True)
class BeamIdentificationResult:
    """Identified values, assumed-noise intervals, and residual evidence."""

    converged: bool
    target_frequencies_hz: FloatArray
    fitted_frequencies_hz: FloatArray
    maximum_frequency_residual_hz: float
    youngs_modulus_pa: float
    head_mass_kg: float
    youngs_modulus_interval_pa: tuple[float, float]
    head_mass_interval_kg: tuple[float, float]
    declared_truth_youngs_modulus_pa: float
    declared_truth_head_mass_kg: float
    interval_basis: str = "local_linear_assumed_frequency_noise"


@dataclass(frozen=True, slots=True)
class ModalResponse:
    """Time history and work--energy terms for one modal truncation."""

    time_s: FloatArray
    tip_deflection_m: FloatArray
    tip_velocity_m_s: FloatArray
    mechanical_energy_j: FloatArray
    input_power_w: FloatArray
    damping_power_w: FloatArray
    energy_closure_j: float


@dataclass(frozen=True, slots=True)
class BeamReferenceStudy:
    """Scalar findings plus paired reduced and distributed responses."""

    identification: BeamIdentificationResult
    converged_frequencies_hz: FloatArray
    element_convergence_relative: float
    low_reduced: ModalResponse
    low_reference: ModalResponse
    high_reduced: ModalResponse
    high_reference: ModalResponse
    reference_energy_closure_j: float
    reduced_energy_closure_j: float
    reference_peak_tip_deflection_m: float
    reduced_peak_tip_deflection_m: float
    low_frequency_tip_rms_discrepancy_m: float
    high_frequency_tip_rms_discrepancy_m: float
    claim_status: str = "synthetic_structural_comparison_only"


def model_matrices(
    config: BeamReferenceConfig,
    *,
    youngs_modulus_pa: float | None = None,
    head_mass_kg: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return shared FE mass/stiffness matrices with declared tip inertia."""
    modulus = (
        config.youngs_modulus_pa
        if youngs_modulus_pa is None
        else float(youngs_modulus_pa)
    )
    head_mass = config.head_mass_kg if head_mass_kg is None else float(head_mass_kg)
    _positive("youngs_modulus_pa", modulus)
    _positive("head_mass_kg", head_mass)
    shaft = create_standard_shaft(
        length=config.length_m,
        n_stations=config.element_count + 1,
        tip_diameter=config.tip_diameter_m,
        butt_diameter=config.butt_diameter_m,
        wall_thickness=config.wall_thickness_m,
    )
    shaft = replace(
        shaft,
        youngs_modulus=modulus,
        density=config.density_kg_m3,
        damping_ratio=config.damping_ratio,
    )
    model = FiniteElementShaftModel(n_elements=config.element_count)
    model.initialize(shaft)
    mass = np.asarray(model.M, dtype=np.float64).copy()
    stiffness = np.asarray(model.K, dtype=np.float64).copy()
    # Last node contributes translation then cross-section rotation.
    mass[-2, -2] += head_mass
    mass[-1, -1] += config.head_rotary_inertia_kg_m2
    return mass, stiffness


def modal_basis(
    mass: FloatArray, stiffness: FloatArray, mode_count: int
) -> tuple[FloatArray, FloatArray]:
    """Return ascending frequencies and mass-normalized eigenvectors."""
    if (
        mass.ndim != 2
        or mass.shape != stiffness.shape
        or mass.shape[0] != mass.shape[1]
    ):
        raise ValueError("mass and stiffness must be square matrices of equal shape")
    if mode_count < 1 or mode_count > mass.shape[0]:
        raise ValueError("mode_count is outside the matrix dimension")
    eigenvalues, eigenvectors = eigh(
        stiffness, mass, subset_by_index=(0, mode_count - 1)
    )
    if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("beam eigenproblem did not return finite positive modes")
    frequencies = np.sqrt(eigenvalues) / (2.0 * np.pi)
    return np.asarray(frequencies), np.asarray(eigenvectors)


def _first_two_frequencies(
    base: BeamReferenceConfig, modulus: float, head_mass: float
) -> FloatArray:
    mass, stiffness = model_matrices(
        base, youngs_modulus_pa=modulus, head_mass_kg=head_mass
    )
    frequencies, _ = modal_basis(mass, stiffness, 2)
    return frequencies


def identify_beam_parameters(
    config: BeamIdentificationConfig,
) -> BeamIdentificationResult:
    """Recover a declared synthetic modulus and tip mass from two modes."""
    base = replace(
        BeamReferenceConfig.publication_default(),
        element_count=config.element_count,
    )
    target = _first_two_frequencies(
        base,
        config.declared_truth_youngs_modulus_pa,
        config.declared_truth_head_mass_kg,
    )

    def residual(log_values: FloatArray) -> FloatArray:
        modulus, head_mass = np.exp(log_values)
        return _first_two_frequencies(base, modulus, head_mass) - target

    solved = least_squares(
        residual,
        np.log([config.initial_youngs_modulus_pa, config.initial_head_mass_kg]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        diff_step=1e-5,
        max_nfev=100,
    )
    modulus, head_mass = np.exp(solved.x)
    fitted = _first_two_frequencies(base, modulus, head_mass)
    information = solved.jac.T @ solved.jac
    if np.linalg.matrix_rank(information) != 2:
        raise ValueError("synthetic modal identification is rank deficient")
    covariance_log = config.frequency_sigma_hz**2 * np.linalg.inv(information)
    standard_log = np.sqrt(np.diag(covariance_log))
    lower = np.exp(solved.x - 1.96 * standard_log)
    upper = np.exp(solved.x + 1.96 * standard_log)
    return BeamIdentificationResult(
        converged=bool(solved.success),
        target_frequencies_hz=target,
        fitted_frequencies_hz=fitted,
        maximum_frequency_residual_hz=float(np.max(np.abs(fitted - target))),
        youngs_modulus_pa=float(modulus),
        head_mass_kg=float(head_mass),
        youngs_modulus_interval_pa=(float(lower[0]), float(upper[0])),
        head_mass_interval_kg=(float(lower[1]), float(upper[1])),
        declared_truth_youngs_modulus_pa=config.declared_truth_youngs_modulus_pa,
        declared_truth_head_mass_kg=config.declared_truth_head_mass_kg,
    )


def _half_sine_force(time_s: float, pulse_s: float, peak_n: float) -> float:
    if time_s < 0.0 or time_s > pulse_s:
        return 0.0
    return float(peak_n * np.sin(np.pi * time_s / pulse_s))


def _modal_response(
    config: BeamReferenceConfig,
    mode_count: int,
    pulse_s: float,
    *,
    include_tip_moment: bool,
) -> ModalResponse:
    mass, stiffness = model_matrices(config)
    frequencies, modes = modal_basis(mass, stiffness, mode_count)
    omega = 2.0 * np.pi * frequencies
    tip_shapes = modes[-2, :]
    tip_rotation_shapes = modes[-1, :]
    intervals = int(round(config.duration_s / config.step_s))
    time = np.arange(intervals + 1, dtype=np.float64) * config.step_s
    state = np.zeros((intervals + 1, 2 * mode_count))

    def rhs(sample_time: float, sample: FloatArray) -> FloatArray:
        coordinate = sample[:mode_count]
        velocity = sample[mode_count:]
        force = _half_sine_force(sample_time, pulse_s, config.peak_tip_force_n)
        moment = (
            _half_sine_force(sample_time, pulse_s, config.peak_tip_moment_nm)
            if include_tip_moment
            else 0.0
        )
        acceleration = (
            tip_shapes * force
            + tip_rotation_shapes * moment
            - 2.0 * config.damping_ratio * omega * velocity
            - omega**2 * coordinate
        )
        return np.concatenate((velocity, acceleration))

    for index in range(intervals):
        sample = state[index]
        sample_time = float(time[index])
        step = config.step_s
        k1 = rhs(sample_time, sample)
        k2 = rhs(sample_time + step / 2.0, sample + step * k1 / 2.0)
        k3 = rhs(sample_time + step / 2.0, sample + step * k2 / 2.0)
        k4 = rhs(sample_time + step, sample + step * k3)
        state[index + 1] = sample + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    coordinate = state[:, :mode_count]
    velocity = state[:, mode_count:]
    tip = coordinate @ tip_shapes
    tip_velocity = velocity @ tip_shapes
    tip_rotation_velocity = velocity @ tip_rotation_shapes
    force = np.array(
        [
            _half_sine_force(float(value), pulse_s, config.peak_tip_force_n)
            for value in time
        ]
    )
    moment = np.array(
        [
            (
                _half_sine_force(float(value), pulse_s, config.peak_tip_moment_nm)
                if include_tip_moment
                else 0.0
            )
            for value in time
        ]
    )
    energy = 0.5 * np.sum(velocity**2 + (omega * coordinate) ** 2, axis=1)
    input_power = force * tip_velocity + moment * tip_rotation_velocity
    damping_power = -np.sum(2.0 * config.damping_ratio * omega * velocity**2, axis=1)
    expected_change = np.trapezoid(input_power + damping_power, time)
    closure = abs(float(energy[-1] - energy[0] - expected_change))
    return ModalResponse(
        time_s=time,
        tip_deflection_m=tip,
        tip_velocity_m_s=tip_velocity,
        mechanical_energy_j=energy,
        input_power_w=input_power,
        damping_power_w=damping_power,
        energy_closure_j=closure,
    )


def run_beam_reference_study() -> BeamReferenceStudy:
    """Execute identification, convergence, and paired-load comparisons."""
    config = BeamReferenceConfig.publication_default()
    identification = identify_beam_parameters(
        BeamIdentificationConfig.publication_default()
    )
    frequencies_by_mesh = []
    for elements in (12, 24, 48):
        mesh_config = replace(config, element_count=elements)
        mass, stiffness = model_matrices(mesh_config)
        frequencies, _ = modal_basis(mass, stiffness, 3)
        frequencies_by_mesh.append(frequencies)
    convergence = float(
        np.max(
            np.abs(frequencies_by_mesh[1] - frequencies_by_mesh[2])
            / frequencies_by_mesh[2]
        )
    )
    low_reduced = _modal_response(
        config, 1, config.low_frequency_pulse_s, include_tip_moment=False
    )
    low_reference = _modal_response(
        config,
        config.mode_count,
        config.low_frequency_pulse_s,
        include_tip_moment=False,
    )
    high_reduced = _modal_response(
        config, 1, config.high_frequency_pulse_s, include_tip_moment=True
    )
    high_reference = _modal_response(
        config,
        config.mode_count,
        config.high_frequency_pulse_s,
        include_tip_moment=True,
    )
    low_rms = float(
        np.sqrt(
            np.mean(
                (low_reduced.tip_deflection_m - low_reference.tip_deflection_m) ** 2
            )
        )
    )
    high_rms = float(
        np.sqrt(
            np.mean(
                (high_reduced.tip_deflection_m - high_reference.tip_deflection_m) ** 2
            )
        )
    )
    return BeamReferenceStudy(
        identification=identification,
        converged_frequencies_hz=frequencies_by_mesh[-1],
        element_convergence_relative=convergence,
        low_reduced=low_reduced,
        low_reference=low_reference,
        high_reduced=high_reduced,
        high_reference=high_reference,
        reference_energy_closure_j=max(
            low_reference.energy_closure_j, high_reference.energy_closure_j
        ),
        reduced_energy_closure_j=max(
            low_reduced.energy_closure_j, high_reduced.energy_closure_j
        ),
        reference_peak_tip_deflection_m=float(
            np.max(np.abs(high_reference.tip_deflection_m))
        ),
        reduced_peak_tip_deflection_m=float(
            np.max(np.abs(high_reduced.tip_deflection_m))
        ),
        low_frequency_tip_rms_discrepancy_m=low_rms,
        high_frequency_tip_rms_discrepancy_m=high_rms,
    )


__all__ = [
    "BeamIdentificationConfig",
    "BeamIdentificationResult",
    "BeamReferenceConfig",
    "BeamReferenceStudy",
    "ModalResponse",
    "identify_beam_parameters",
    "modal_basis",
    "model_matrices",
    "run_beam_reference_study",
]
