"""Trajectory-level synthetic qualification of a bilateral point-force sensor map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bilateral_wrench_identifiability import (
    internal_axial_measurement,
    point_force_wrench_map,
)

FloatArray = NDArray[np.float64]
DEFAULT_CHANNEL_SCALES = (100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 100.0)


@dataclass(frozen=True, slots=True)
class SensorQualificationConfig:
    """Declared synthetic sensor and contact-geometry assumptions."""

    sample_count: int = 301
    trial_count: int = 32
    normalized_noise_std: float = 0.002
    normalized_cross_talk: float = 0.005
    cross_talk_calibration_error_fraction: float = 0.10
    contact_migration_m: float = 0.004
    apply_cross_talk_correction: bool = True
    track_contact_centers: bool = False
    grip_span_m: float = 0.20
    channel_scales: tuple[float, ...] = DEFAULT_CHANNEL_SCALES
    seed: int = 20260814

    def __post_init__(self) -> None:
        """Validate the declared simulation contract."""

        if self.sample_count < 3:
            raise ValueError("sample_count must be at least 3")
        if self.trial_count < 1:
            raise ValueError("trial_count must be positive")
        for name in (
            "normalized_noise_std",
            "normalized_cross_talk",
            "cross_talk_calibration_error_fraction",
            "contact_migration_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        scales = np.asarray(self.channel_scales, dtype=float)
        if scales.shape != (7,):
            raise ValueError("channel_scales must contain seven values")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("channel_scales must be positive and finite")
        if not np.isfinite(self.grip_span_m) or self.grip_span_m <= 0.0:
            raise ValueError("grip_span_m must be positive and finite")
        if self.contact_migration_m >= 0.5 * self.grip_span_m:
            raise ValueError("contact_migration_m must remain below half the grip span")


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    """Allocation and net-wrench recovery errors for one estimator."""

    allocation_rmse_n: float
    allocation_p95_n: float
    normalized_net_wrench_rmse: float
    axial_mode_rmse_n: float


@dataclass(frozen=True, slots=True)
class SensorQualificationResult:
    """Synthetic practical-identifiability result with explicit scope boundaries."""

    augmented: RecoveryMetrics
    net_wrench_only: RecoveryMetrics
    scope: str = "synthetic_point_force_sensor_qualification"
    human_validation: str = "untested"
    anatomical_strategy: str = "not_identified"


def _nominal_contacts(span_m: float) -> FloatArray:
    return np.array(((-0.5 * span_m, 0.0, 0.0), (0.5 * span_m, 0.0, 0.0)))


def _contact_trajectory(config: SensorQualificationConfig) -> FloatArray:
    phase = np.linspace(0.0, 2.0 * np.pi, config.sample_count, endpoint=False)
    contacts = np.repeat(
        _nominal_contacts(config.grip_span_m)[None, :, :], config.sample_count, axis=0
    )
    amplitude = config.contact_migration_m
    contacts[:, 0, 0] += 0.35 * amplitude * np.sin(phase)
    contacts[:, 0, 1] += amplitude * np.sin(phase + 0.3)
    contacts[:, 1, 0] -= 0.25 * amplitude * np.cos(phase)
    contacts[:, 1, 1] += 0.8 * amplitude * np.cos(phase - 0.2)
    return contacts


def _force_trajectory(sample_count: int) -> FloatArray:
    phase = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    common = np.column_stack(
        (
            75.0 + 12.0 * np.sin(phase),
            30.0 * np.cos(phase + 0.2),
            45.0 * np.sin(2.0 * phase - 0.3),
        )
    )
    differential = np.column_stack(
        (
            18.0 + 14.0 * np.sin(phase - 0.4),
            24.0 * np.cos(1.5 * phase),
            16.0 * np.sin(0.5 * phase + 0.7),
        )
    )
    return np.concatenate((common + differential, common - differential), axis=1)


def _augmented_map(contacts: FloatArray, scales: FloatArray) -> FloatArray:
    physical = np.vstack(
        (point_force_wrench_map(contacts), internal_axial_measurement(contacts))
    )
    return physical / scales[:, None]


def _cross_talk_pattern(channel_count: int) -> FloatArray:
    row, column = np.indices((channel_count, channel_count))
    pattern = np.sin((row + 1.0) * (column + 2.0))
    np.fill_diagonal(pattern, 0.0)
    return pattern / np.max(np.abs(pattern))


def _measurement_correction(
    config: SensorQualificationConfig, pattern: FloatArray
) -> FloatArray:
    identity = np.eye(pattern.shape[0])
    if not config.apply_cross_talk_correction:
        return identity
    estimated = config.normalized_cross_talk * (
        1.0 + config.cross_talk_calibration_error_fraction
    )
    return np.linalg.inv(identity + estimated * pattern)


def _metrics(
    estimates: FloatArray,
    truth: FloatArray,
    contacts: FloatArray,
    net_scales: FloatArray,
) -> RecoveryMetrics:
    errors = estimates - truth
    allocation_rmse = float(np.sqrt(np.mean(errors**2)))
    allocation_p95 = float(np.percentile(np.linalg.norm(errors, axis=1), 95.0))
    net_errors = np.vstack(
        [
            (point_force_wrench_map(contact) @ error) / net_scales
            for contact, error in zip(contacts, errors, strict=True)
        ]
    )
    axial_errors = []
    for contact, error in zip(contacts, errors, strict=True):
        axial = internal_axial_measurement(contact)[0]
        axial_errors.append(float(axial @ error))
    return RecoveryMetrics(
        allocation_rmse_n=allocation_rmse,
        allocation_p95_n=allocation_p95,
        normalized_net_wrench_rmse=float(np.sqrt(np.mean(net_errors**2))),
        axial_mode_rmse_n=float(np.sqrt(np.mean(np.square(axial_errors)))),
    )


def _run_augmented_trials(
    config: SensorQualificationConfig,
    truth: FloatArray,
    contacts: FloatArray,
    scales: FloatArray,
) -> FloatArray:
    rng = np.random.default_rng(config.seed)
    pattern = _cross_talk_pattern(7)
    mixing = np.eye(7) + config.normalized_cross_talk * pattern
    correction = _measurement_correction(config, pattern)
    nominal = _nominal_contacts(config.grip_span_m)
    estimates: list[FloatArray] = []
    for _ in range(config.trial_count):
        for contact, forces in zip(contacts, truth, strict=True):
            true_map = _augmented_map(contact, scales)
            normalized = mixing @ (true_map @ forces)
            normalized += rng.normal(0.0, config.normalized_noise_std, size=7)
            estimate_contact = contact if config.track_contact_centers else nominal
            estimate_map = _augmented_map(estimate_contact, scales)
            estimate, *_ = np.linalg.lstsq(
                estimate_map, correction @ normalized, rcond=None
            )
            estimates.append(estimate)
    return np.vstack(estimates)


def _net_only_estimates(truth: FloatArray, contacts: FloatArray) -> FloatArray:
    estimates = []
    for contact, forces in zip(contacts, truth, strict=True):
        matrix = point_force_wrench_map(contact)
        estimates.append(np.linalg.pinv(matrix) @ (matrix @ forces))
    return np.vstack(estimates)


def run_sensor_qualification(
    config: SensorQualificationConfig,
) -> SensorQualificationResult:
    """Run a deterministic synthetic trajectory-level sensor qualification.

    The postcondition is a finite, reproducible result. It is not human or
    anatomical validation and does not cover distributed contact moments.
    """

    scales = np.asarray(config.channel_scales, dtype=float)
    contacts = _contact_trajectory(config)
    truth = _force_trajectory(config.sample_count)
    augmented_estimates = _run_augmented_trials(config, truth, contacts, scales)
    repeated_truth = np.tile(truth, (config.trial_count, 1))
    repeated_contacts = np.tile(contacts, (config.trial_count, 1, 1))
    augmented = _metrics(
        augmented_estimates, repeated_truth, repeated_contacts, scales[:6]
    )
    net_estimates = _net_only_estimates(truth, contacts)
    net_only = _metrics(net_estimates, truth, contacts, scales[:6])
    result = SensorQualificationResult(augmented=augmented, net_wrench_only=net_only)
    if not all(
        np.isfinite(value)
        for metrics in (result.augmented, result.net_wrench_only)
        for value in (
            metrics.allocation_rmse_n,
            metrics.allocation_p95_n,
            metrics.normalized_net_wrench_rmse,
            metrics.axial_mode_rmse_n,
        )
    ):
        raise RuntimeError("sensor qualification produced non-finite metrics")
    return result
