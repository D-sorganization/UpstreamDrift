"""Canonical native research result persistence, shared by workers and seed authors."""

from __future__ import annotations
from dataclasses import asdict
from typing import Any
import numpy as np
import json
from src.shared.python.motion_matching.historical_fit.contracts import (
    ImageFitConfig,
    ImageFitResult,
    ImageSplineStart,
)
from src.shared.python.motion_matching.historical_fit.spline_expansion import (
    SplineCoordinateExpansion,
    expand_image_spline_coordinates,
)
from src.shared.python.estimation import CubicHermiteSplineTrajectory
from .necromatcher_spline import preserved_fit_spline
from .necromatcher_fit_jobs import _digest, _parse_shaft_recipe
from src.shared.python.motion_matching.historical_fit.shaft_geometry import (
    AuthoredShaftAxis,
)
from src.shared.python.motion_matching.historical_fit.shaft_residuals import (
    ShaftAxisAssessment,
)


def _validate_seed_samples(
    source: dict[str, Any],
    result: ImageFitResult,
    dense_output: tuple[tuple[int, ...], list[dict[str, Any]], np.ndarray],
    expansion: SplineCoordinateExpansion,
) -> None:
    indices, frames, dense_q = dense_output
    original = source["evidence"]["original_fit"]
    if tuple(source["frame_indices"]) != indices or source["frames"] != frames:
        raise ValueError(
            "Coordinate expansion seed must preserve dense frame identities"
        )
    parent_q = np.asarray(source["q"], dtype=float)
    expected_shape = (len(indices), len(result.coordinate_order))
    if (
        parent_q.shape != expected_shape
        or dense_q.shape != expected_shape
        or not np.all(np.isfinite(parent_q))
        or not np.all(np.isfinite(dense_q))
        or not np.allclose(parent_q, dense_q, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("Coordinate expansion seed must preserve dense parent poses")
    locked = [
        i
        for i, name in enumerate(result.coordinate_order)
        if name not in original["free_coordinates"]
    ]
    if not np.allclose(
        parent_q[:, locked],
        np.asarray(expansion.reference_pose)[locked],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Coordinate expansion seed reference conflicts with locked poses"
        )
    training_q = np.asarray(original["q"], dtype=float)
    training_times = np.asarray(original["source_times"], dtype=float)
    if (
        training_q.shape != result.q.shape
        or training_times.shape != result.source_times.shape
        or not np.array_equal(training_times, result.source_times)
        or not np.allclose(training_q, result.q, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError(
            "Coordinate expansion seed must preserve training poses and clock"
        )


def _validate_coordinate_seed(
    request: dict[str, Any],
    source: dict[str, Any],
    result: ImageFitResult,
    expansion: SplineCoordinateExpansion | None,
) -> None:
    operation = request["options"].get("operation", "fit")
    if operation != "coordinate_expansion" and expansion is None:
        return
    if operation != "coordinate_expansion" or not isinstance(
        expansion, SplineCoordinateExpansion
    ):
        raise ValueError(
            "Coordinate expansion seed requires explicit operation and typed receipt"
        )
    if result.optimizer_ran or result.converged or result.initialization is not None:
        raise ValueError(
            "Coordinate expansion seed cannot claim optimization or range projection"
        )
    if (
        request["options"]["frame_indices"]
        != source["evidence"]["original_fit"]["frame_indices"]
    ):
        raise ValueError(
            "Coordinate expansion seed must preserve training frame identities"
        )
    parent = preserved_fit_spline(source)
    if parent is None:
        raise ValueError("Coordinate expansion seed requires a preserved parent")
    expected = expand_image_spline_coordinates(
        parent, expansion.expanded_start.free_coordinates, expansion.reference_pose
    )
    if expansion != expected:
        raise ValueError(
            "Coordinate expansion seed receipt differs from preserved parent"
        )
    actual = ImageSplineStart.from_coefficients(
        result.knot_times,
        result.spline_coefficients,
        result.coordinate_order,
        result.free_coordinates,
        result.model_sha,
    )
    if actual != expansion.expanded_start or result.initial_spline != actual:
        raise ValueError(
            "Coordinate expansion seed result differs from expanded snapshot"
        )
    if result.rms_pixels != result.initial_rms_pixels:
        raise ValueError(
            "Coordinate expansion seed initial and evaluated RMS must agree"
        )
    trajectory = CubicHermiteSplineTrajectory(
        np.asarray(actual.knot_times), len(actual.free_coordinates)
    )
    q = np.tile(expansion.reference_pose, (len(result.source_times), 1))
    indices = [actual.coordinate_order.index(name) for name in actual.free_coordinates]
    q[:, indices] = trajectory.evaluate(
        np.asarray(actual.spline_coefficients), result.source_times
    ).q
    if not np.allclose(q, result.q, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "Coordinate expansion seed poses disagree with its canonical spline"
        )


def native_research_blockers(
    result: ImageFitResult, config: ImageFitConfig, ranges: dict[str, Any] | None
) -> list[str]:
    """Retain qualification blockers independently of numerical optimizer status."""
    blockers = [
        "monocular_research_only",
        "physical_clock_unknown",
        "camera_unqualified",
        "independent_dynamics_not_replayed",
        "nonlinear_continuous_constraints_not_certified",
        "historical_anatomy_unqualified",
    ]
    configured = {
        name: [lower, upper] for name, lower, upper in config.coordinate_bounds
    }
    exact = (
        bool(configured)
        and ranges is not None
        and ranges.get("range_source")
        == "bound_native_definition.coordinate_ranges_deg"
        and ranges.get("named_bounds") == configured
    )
    if not exact:
        blockers.append("anatomical_ranges_not_enforced")
        blockers.append("native_authored_ranges_not_fully_enforced")
    elif ranges and ranges.get("unbounded_names"):
        blockers.append("native_coordinates_without_authored_ranges")
    if not result.optimizer_ran:
        blockers.append("authored_initialization_only")
    elif not result.converged:
        blockers.append("optimizer_not_converged")
    return blockers


def native_fit_evidence(
    result: ImageFitResult,
    original: dict[str, Any],
    options: dict[str, Any],
    max_grip: float,
) -> dict[str, Any]:
    """Serialize the actual evaluator result using one canonical record layout."""
    telemetry = getattr(result, "telemetry", None)
    return {
        **(
            {"solver_telemetry": telemetry.to_record()} if telemetry is not None else {}
        ),
        "camera": original["camera"],
        "attachments": original["attachments"],
        "free_coordinates": list(result.free_coordinates),
        "coordinate_order": list(result.coordinate_order),
        "frame_indices": list(options["frame_indices"]),
        "source_times": result.source_times.tolist(),
        "q": result.q.tolist(),
        "knot_times": result.knot_times.tolist(),
        "spline_coefficients": result.spline_coefficients.tolist(),
        "initial_rms_pixels": result.initial_rms_pixels,
        "rms_pixels": result.rms_pixels,
        "converged": result.converged,
        "optimizer_ran": result.optimizer_ran,
        "initialization": asdict(result.initialization)
        if result.initialization is not None
        else None,
        "initial_spline": result.initial_spline.to_record()
        if result.initial_spline is not None
        else None,
        "initial_coefficient_sha256": result.initial_spline.coefficient_sha256
        if result.initial_spline is not None
        else None,
        "spline_start": ImageSplineStart.from_coefficients(
            result.knot_times,
            result.spline_coefficients,
            tuple(result.coordinate_order),
            result.free_coordinates,
            result.model_sha,
        ).to_record(),
        "model_sha": result.model_sha,
        "initialization_source": options.get("initialization_source", "sampled_parent"),
        "message": result.optimizer_message,
        "config": asdict(ImageFitConfig.from_record(options["config"])),
        "max_grip_separation_m": max_grip,
        "constraint_assessment": {
            "tested_times": result.constraint_times.tolist(),
            "row_labels": list(result.constraint_row_labels),
            "scaled_residuals": result.constraint_residuals.tolist(),
            "maximum_dimensionless_residual": result.maximum_constraint_residual,
            "continuous_certified": False,
        },
    }


def _add_shaft_record(
    output: dict[str, Any], request: dict[str, Any], result: ImageFitResult
) -> None:
    if "shaft_images" not in request:
        if getattr(result, "additional_image_assessments", ()):
            raise ValueError("Shaft diagnostics require an explicit admitted recipe")
        return
    recipe = request["shaft_images"]
    declared, _ = _parse_shaft_recipe(recipe)
    evidence = declared.evidence
    assessments = result.additional_image_assessments
    if (
        len(assessments) != 1
        or not isinstance(assessments[0], ShaftAxisAssessment)
        or assessments[0].evidence_sha256 != evidence.sha256
    ):
        raise ValueError("Shaft diagnostics differ from explicit recipe evidence")
    assessment = assessments[0]
    times = tuple(float(frame.frame.presentation_time) for frame in evidence.frames)
    if (
        assessment.frame_indices
        != tuple(frame.frame_index for frame in evidence.frames)
        or assessment.source_times != times
        or min(times) < result.source_times[0]
        or max(times) > result.source_times[-1]
    ):
        raise ValueError("Shaft raw diagnostic source frames differ from recipe")
    if not isinstance(request.get("shaft_axis"), dict):
        raise ValueError("Shaft recipe requires an explicit typed authored axis record")
    raw_axis = dict(request["shaft_axis"])
    if raw_axis.pop("semantic", None) != "infinite_authored_shaft_axis":
        raise ValueError("Shaft authored axis semantic differs")
    axis = AuthoredShaftAxis(**raw_axis)
    if (
        axis.native_model_sha != result.model_sha
        or recipe["evidence_sha256"] != evidence.sha256
    ):
        raise ValueError("Shaft axis/model/evidence identity differs from result")
    output["provenance"]["shaft_images"] = {
        "recipe_sha256": _digest(recipe),
        "axis_sha256": _digest(axis.to_record()),
        "axis": axis.to_record(),
    }
    output["evidence"]["shaft_axis"] = {
        "recipe": recipe,
        "assessments": [asdict(assessment)],
        "body_rms_pixels": result.rms_pixels,
        "body_observed_point_count": result.observed_point_count,
        "shaft_raw_metric": "unweighted_perpendicular_scalar_pixel_rms",
        "confidence_uncertainty_status": "authored_uncalibrated",
        "physical_time_qualified": False,
        "physical_geometry_qualified": False,
    }


def build_native_fit_payload(
    request: dict[str, Any],
    source: dict[str, Any],
    result: ImageFitResult,
    dense_output: tuple[tuple[int, ...], list[dict[str, Any]], np.ndarray],
    stamp: dict[str, Any],
    max_grip: float,
    ranges: dict[str, Any] | None = None,
    coordinate_expansion: SplineCoordinateExpansion | None = None,
) -> dict[str, Any]:
    """Build legacy research results or an explicitly verified unoptimized seed."""
    _validate_coordinate_seed(request, source, result, coordinate_expansion)
    if coordinate_expansion is not None:
        _validate_seed_samples(source, result, dense_output, coordinate_expansion)
    original = source["evidence"]["original_fit"]
    options = request["options"]
    dense_indices, frames, q = dense_output
    blockers = native_research_blockers(
        result, ImageFitConfig.from_record(options["config"]), ranges
    )
    if coordinate_expansion is not None:
        blockers.remove("authored_initialization_only")
        blockers.append("coordinate_expansion_only")
    output = {
        **source,
        "frame_indices": list(dense_indices),
        "frames": frames,
        "q": q.tolist(),
        "provenance": {
            "native_definition": source["provenance"]["native_definition"],
            "warm_start_provenance": source["provenance"],
            "description": "Source-stamped native research "
            + options.get("operation", "fit"),
            "operation": options.get("operation", "fit"),
            "warm_start_fit_id": request["source_fit_id"],
            "warm_start_fit_hash": request["source_fit_hash"],
            "request_options": options,
            "execution_stamp": request["execution_stamp"],
            "worker_stamp": {
                key: stamp[key]
                for key in ("started_at_utc", "source_sha256", "runtime_sha256")
            },
        },
        "evidence": {
            "rejection_reasons": blockers,
            "original_fit": native_fit_evidence(result, original, options, max_grip),
        },
    }
    if coordinate_expansion is not None:
        output["provenance"]["coordinate_expansion"] = asdict(coordinate_expansion)
    _add_shaft_record(output, request, result)
    return output
