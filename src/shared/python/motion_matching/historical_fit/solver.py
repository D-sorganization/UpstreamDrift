"""Native pixel reprojection fits using the canonical spline MAP estimator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial

import numpy as np

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    HermiteBoundsDomain,
    MapEstimatorOptions,
    MapEstimatorProblem,
    SharedParameterBlock,
    SplineTrajectoryEvaluation,
    finite_difference_jacobian,
    solve_single_trial_map,
)
from src.shared.python.motion_matching.pipeline.plant import MatchingPlant
from src.shared.python.motion_matching.constraint_kinematics import (
    ConstraintLinearization,
)
from .contracts import CameraProjection, ImageFitConfig, ImageFitInputs, ImageFitResult


class _Fit:
    def __init__(
        self,
        native: MatchingPlant,
        attachments: Mapping[str, tuple[str, Sequence[float]]],
        camera: CameraProjection,
        inputs: ImageFitInputs,
        config: ImageFitConfig,
    ) -> None:
        self.native, self.attachments, self.camera = native, attachments, camera
        self.inputs, self.config = inputs, config
        order = native.coordinate_order
        if inputs.seed.shape != (len(order),):
            raise ValueError("Seed must match the native coordinate order")
        if len(attachments) != inputs.observed_pixels.shape[1] or not attachments:
            raise ValueError("Attachments must match observed marker order")
        if any(name not in order for name in inputs.free_coordinates):
            raise ValueError("Free coordinates must belong to the native model")
        self.indices = np.array([order.index(name) for name in inputs.free_coordinates])
        if inputs.initial_samples is not None:
            locked = [index for index in range(len(order)) if index not in self.indices]
            if not np.array_equal(
                inputs.initial_samples[:, locked],
                np.tile(inputs.seed[locked], (len(inputs.source_times), 1)),
            ):
                raise ValueError("Warm-start samples cannot change locked coordinates")
        knots = inputs.source_times if inputs.knot_times is None else inputs.knot_times
        probes = [
            left + fraction * (right - left)
            for left, right in zip(knots[:-1], knots[1:], strict=True)
            for fraction in config.interior_fractions
        ]
        self.evaluation_times = np.unique(np.concatenate([inputs.source_times, probes]))
        self.source_indices = np.searchsorted(
            self.evaluation_times, inputs.source_times
        )
        self.ik = (
            native.create_ik(attachments)
            if config.constraint_options is not None
            else None
        )

    def source_evaluation(
        self, evaluation: SplineTrajectoryEvaluation
    ) -> SplineTrajectoryEvaluation:
        """Select original evidence rows without inventing observations at probes."""
        if not np.array_equal(evaluation.times, self.evaluation_times):
            raise ValueError(
                "Fit evaluation must match configured source and probe times"
            )
        indices = self.source_indices
        return replace(
            evaluation,
            times=evaluation.times[indices],
            q=evaluation.q[indices],
            v=evaluation.v[indices],
            a=evaluation.a[indices],
            q_basis=evaluation.q_basis[indices],
            v_basis=evaluation.v_basis[indices],
            a_basis=evaluation.a_basis[indices],
        )

    def constraint_linearizations(self, q: np.ndarray) -> list[ConstraintLinearization]:
        """Use the declared public IK capability with stable rows and native order."""
        if self.ik is None or self.config.constraint_options is None:
            return []
        rows = [
            self.ik.constraint_residual_jacobian(pose, self.config.constraint_options)
            for pose in q
        ]
        if any(not isinstance(row, ConstraintLinearization) for row in rows):
            raise ValueError(
                "Constraint linearization must be a validated public record"
            )
        for row in rows:
            if (
                row.coordinate_order != tuple(self.native.coordinate_order)
                or row.row_labels != rows[0].row_labels
                or row.jacobian.shape != (len(row.row_labels), len(self.inputs.seed))
                or row.residual.shape != (len(row.row_labels),)
                or not np.isfinite(row.residual).all()
                or not np.isfinite(row.jacobian).all()
            ):
                raise ValueError(
                    "Constraint linearization must retain finite fixed rows and native coordinate order"
                )
        return rows

    def expand(self, free: np.ndarray) -> np.ndarray:
        result = np.tile(self.inputs.seed, (len(free), 1))
        result[:, self.indices] = free
        return result

    def image_residuals(self, q: np.ndarray) -> np.ndarray:
        rows = []
        for pose, observed, weights in zip(
            q, self.inputs.observed_pixels, self.inputs.confidence, strict=True
        ):
            points = self.native.marker_positions(pose, self.attachments)
            rows.append(self.camera.residual(points, observed, weights))
        return np.asarray(rows)

    def residual(
        self, evaluation: SplineTrajectoryEvaluation, parameters: Mapping[str, float]
    ) -> np.ndarray:
        source = self.source_evaluation(evaluation)
        q = self.expand(source.q)
        pieces: list[np.ndarray] = [self.image_residuals(q).ravel()]
        scaled_prior = (q - self.inputs.seed) / self.inputs.coordinate_scales
        pieces.append(self.config.prior_weight * scaled_prior.ravel())
        duration = self.inputs.source_times[-1] - self.inputs.source_times[0]
        scaled_speed = source.v / self.inputs.coordinate_scales[self.indices]
        pieces.append(self.config.smoothness_weight * duration * scaled_speed.ravel())
        if self.ik is not None:
            pieces.append(
                np.concatenate(
                    [
                        row.residual
                        for row in self.constraint_linearizations(
                            self.expand(evaluation.q)
                        )
                    ]
                )
            )
        elif self.config.closure_weight:
            pieces.append(
                self.config.closure_weight
                * np.array([self.native.closure_residuals(pose) for pose in q]).ravel()
            )
        return np.concatenate(pieces)

    def rms(self, residual: np.ndarray) -> float:
        return float(np.sqrt(np.sum(residual**2) / np.sum(self.inputs.confidence)))

    def _frame_residual(
        self, free: np.ndarray, observed: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        pose = self.expand(free[None, :])[0]
        points = self.native.marker_positions(pose, self.attachments)
        pixels = self.camera.residual(points, observed, weights).ravel()
        if self.ik is not None or not self.config.closure_weight:
            return pixels
        return np.concatenate(
            [pixels, self.config.closure_weight * self.native.closure_residuals(pose)]
        )

    def jacobian(
        self,
        evaluation: SplineTrajectoryEvaluation,
        parameters: Mapping[str, float],
        layout: object,
    ) -> np.ndarray:
        """Differentiate native frame poses once, then apply exact spline bases.

        Native pose derivatives use the canonical central-difference helper.
        They depend on free DOF count rather than the number of spline knots.
        Pose and speed priors use exact derivatives; no engine internals enter.
        """
        pixel_size = 2 * len(self.attachments)
        source = self.source_evaluation(evaluation)
        image_rows: list[np.ndarray] = []
        closure_rows: list[np.ndarray] = []
        for free, observed, weights, basis in zip(
            source.q,
            self.inputs.observed_pixels,
            self.inputs.confidence,
            source.q_basis,
            strict=True,
        ):
            native_jacobian = finite_difference_jacobian(
                partial(self._frame_residual, observed=observed, weights=weights), free
            )
            chained = native_jacobian @ basis
            image_rows.append(chained[:pixel_size])
            if self.ik is None and self.config.closure_weight:
                closure_rows.append(chained[pixel_size:])
        frames, _, columns = source.q_basis.shape
        prior = np.zeros((frames, len(self.inputs.seed), columns))
        prior[:, self.indices] = (
            self.config.prior_weight
            * source.q_basis
            / self.inputs.coordinate_scales[self.indices][None, :, None]
        )
        duration = self.inputs.source_times[-1] - self.inputs.source_times[0]
        speed = (
            self.config.smoothness_weight
            * duration
            * source.v_basis
            / self.inputs.coordinate_scales[self.indices][None, :, None]
        )
        pieces = [
            np.vstack(image_rows),
            prior.reshape(-1, columns),
            speed.reshape(-1, columns),
        ]
        if self.ik is not None:
            rows = self.constraint_linearizations(self.expand(evaluation.q))
            pieces.append(
                np.vstack(
                    [
                        row.jacobian[:, self.indices] @ basis
                        for row, basis in zip(rows, evaluation.q_basis, strict=True)
                    ]
                )
            )
        elif closure_rows:
            pieces.append(np.vstack(closure_rows))
        return np.vstack(pieces)


def _trajectory_domain(fit: _Fit, knots: np.ndarray) -> HermiteBoundsDomain | None:
    """Bind explicit native named limits without changing fixed coordinates."""
    if not fit.config.coordinate_bounds:
        return None
    bounds = {
        name: (lower, upper) for name, lower, upper in fit.config.coordinate_bounds
    }
    for name, (lower, upper) in bounds.items():
        if name not in fit.native.coordinate_order:
            raise ValueError("Coordinate bounds name an unknown native coordinate")
        if name not in fit.inputs.free_coordinates:
            value = fit.inputs.seed[fit.native.coordinate_order.index(name)]
            if not lower <= value <= upper:
                raise ValueError("Coordinate bounds reject the locked native seed")
    return HermiteBoundsDomain(
        tuple(knots), tuple(bounds.get(name) for name in fit.inputs.free_coordinates)
    )


def fit_image_trajectory(
    native: MatchingPlant,
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    camera: CameraProjection,
    inputs: ImageFitInputs,
    config: ImageFitConfig = ImageFitConfig(),
) -> ImageFitResult:
    """Fit native poses to pixels; camera, geometry and priors remain explicit assumptions."""
    fit = _Fit(native, attachments, camera, inputs, config)
    knots = inputs.source_times if inputs.knot_times is None else inputs.knot_times
    trajectory = CubicHermiteSplineTrajectory(knots, len(fit.indices))
    initial = (
        np.tile(inputs.seed[fit.indices], (len(inputs.source_times), 1))
        if inputs.initial_samples is None
        else inputs.initial_samples[:, fit.indices]
    )
    coefficients = trajectory.initial_coefficients_from_samples(
        inputs.source_times, initial
    )
    initial_residual = fit.image_residuals(
        fit.expand(trajectory.evaluate(coefficients, inputs.source_times).q)
    )
    problem = MapEstimatorProblem(
        trajectory,
        fit.evaluation_times,
        coefficients,
        SharedParameterBlock(()),
        fit.residual,
        jacobian=fit.jacobian,
        options=MapEstimatorOptions(
            max_iterations=config.max_iterations,
            non_finite_policy="raise",
            trajectory_domain=_trajectory_domain(fit, knots),
        ),
    )
    solved = solve_single_trial_map(problem)
    q = fit.expand(trajectory.evaluate(solved.coefficients, inputs.source_times).q)
    final_residual = fit.image_residuals(q)
    weighted = final_residual.reshape(inputs.confidence.shape + (2,))
    distances = np.empty(inputs.confidence.shape, dtype=object)
    distances[:] = None
    valid = inputs.confidence > 0
    distances[valid] = np.sqrt(
        np.sum(weighted[valid] ** 2, axis=1) / inputs.confidence[valid]
    )
    constraint_rows = fit.constraint_linearizations(
        fit.expand(trajectory.evaluate(solved.coefficients, fit.evaluation_times).q)
    )
    return ImageFitResult(
        inputs.source_times.copy(),
        q,
        fit.rms(final_residual),
        fit.rms(initial_residual),
        distances,
        int(np.count_nonzero(valid)),
        native.plant_sha,
        native.coordinate_order,
        solved.success,
        solved.message,
        knots,
        solved.coefficients,
        inputs.free_coordinates,
        constraint_times=fit.evaluation_times if constraint_rows else np.empty(0),
        constraint_residuals=np.array([row.residual for row in constraint_rows])
        if constraint_rows
        else np.empty((0, 0)),
        constraint_row_labels=constraint_rows[0].row_labels if constraint_rows else (),
    )
