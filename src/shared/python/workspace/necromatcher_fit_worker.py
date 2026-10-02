"""Clean-interpreter native refit producer; invoked by the owned job service."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.shared.python.motion_matching.historical_fit import (
    CameraProjection,
    CaptureImageEvidence,
    ImageFitConfig,
    ImageFitInputs,
    ImageFitResult,
    ImageSplineStart,
    fit_image_trajectory,
    read_capture_evidence,
)
from src.shared.python.motion_matching.pipeline.plant import MatchingPlant
from src.shared.python.estimation import CubicHermiteSplineTrajectory
from .necromatcher import NecromatcherLibrary
from .necromatcher_fit_jobs import fit_execution_stamp
from .necromatcher_native import NativeFitBinding, load_native_fit_binding
from .necromatcher_review import CaptureReview
from .necromatcher_spline import preserved_fit_spline

logger = logging.getLogger(__name__)


def _compute_operation(
    operation: str,
    native: MatchingPlant,
    attachments: dict[str, Any],
    camera: CameraProjection,
    inputs: ImageFitInputs,
    config: ImageFitConfig,
    initial_spline: ImageSplineStart | None = None,
) -> ImageFitResult:
    if operation == "fit":
        return fit_image_trajectory(
            native, attachments, camera, inputs, config, initial_spline
        )
    if operation != "author_initialization":
        raise ValueError("Unknown native fit operation")
    if initial_spline is not None:
        raise ValueError("Preserved spline requires fit operation")
    if config.initialization_policy != "authored_range_project_zero_slopes":
        raise ValueError(
            "Author operation requires explicit authored initialization policy"
        )
    from src.shared.python.motion_matching.historical_fit import (
        initialize_image_trajectory,
    )

    result = initialize_image_trajectory(native, attachments, camera, inputs, config)
    if (
        result.optimizer_ran is not False
        or result.converged is not False
        or result.initialization is None
    ):
        raise ValueError(
            "Author initialization returned a contradictory execution receipt"
        )
    return result


def _range_provenance(
    binding: NativeFitBinding, config: ImageFitConfig, operation: str
) -> dict[str, Any]:
    configured = {
        name: (lower, upper) for name, lower, upper in config.coordinate_bounds
    }
    if not configured and operation == "fit":
        return {"range_source": "none"}
    authored = binding.authored_coordinate_bounds()
    if configured != dict(authored.named_bounds):
        if operation == "author_initialization":
            raise ValueError(
                "Author initialization requires all exact native authored ranges"
            )
        return {"range_source": "custom_coordinate_bounds"}
    if not configured:
        raise ValueError(
            "Author initialization requires nonempty native authored ranges"
        )
    return authored.to_record()


def _preserved_start(
    binding: NativeFitBinding, options: dict[str, Any], source_times: np.ndarray
) -> ImageSplineStart:
    source = binding.fit
    original = source["evidence"]["original_fit"]
    start = preserved_fit_spline(source)
    if start is None:
        raise ValueError("Parent lacks a preserved physical spline")
    if start.model_sha != binding.plant.plant_sha or start.coordinate_order != tuple(
        binding.plant.coordinate_order
    ):
        raise ValueError(
            "Parent preserved spline model or coordinate order differs from binding"
        )
    if start.free_coordinates != tuple(original["free_coordinates"]):
        raise ValueError(
            "Parent preserved spline free coordinate order differs from record"
        )
    if len(start.knot_times) != options["knot_count"]:
        raise ValueError("Preserved spline knot count differs from explicit request")
    if (start.knot_times[0], start.knot_times[-1]) != (
        source_times[0],
        source_times[-1],
    ):
        raise ValueError(
            "Preserved spline source clock interval differs from requested evidence"
        )
    parent_times = np.array(
        [
            frame["pts_ticks"]
            * frame["timebase_numerator"]
            / frame["timebase_denominator"]
            for frame in source["frames"]
        ]
    )
    if np.any(parent_times < start.knot_times[0]) or np.any(
        parent_times > start.knot_times[-1]
    ):
        raise ValueError("Parent source clock exceeds preserved spline interval")
    trajectory = CubicHermiteSplineTrajectory(
        np.asarray(start.knot_times), len(start.free_coordinates)
    )
    free = trajectory.evaluate(np.asarray(start.spline_coefficients), parent_times).q
    samples = np.asarray(source["q"], dtype=float)
    expected = np.tile(samples[0], (len(samples), 1))
    indices = [start.coordinate_order.index(name) for name in start.free_coordinates]
    expected[:, indices] = free
    if not np.allclose(samples, expected, rtol=1e-8, atol=1e-10):
        raise ValueError("Parent samples disagree with the preserved canonical spline")
    return start


def _worker_inputs(
    binding: NativeFitBinding,
    options: dict[str, Any],
    evidence: CaptureImageEvidence,
    samples: np.ndarray,
) -> tuple[ImageFitInputs, ImageSplineStart | None]:
    mode = options.get("initialization_source", "sampled_parent")
    if mode not in ("sampled_parent", "preserved_spline"):
        raise ValueError("Unknown refit initialization source")
    config = ImageFitConfig.from_record(options["config"])
    start = None
    if mode == "preserved_spline":
        if (
            options.get("operation", "fit") != "fit"
            or config.initialization_policy != "strict"
        ):
            raise ValueError(
                "Preserved spline requires fit operation and strict policy"
            )
        start = _preserved_start(binding, options, evidence.source_times)
    knots = (
        np.asarray(start.knot_times)
        if start is not None
        else np.linspace(
            evidence.source_times[0], evidence.source_times[-1], options["knot_count"]
        )
    )
    return ImageFitInputs(
        evidence.source_times,
        evidence.observed_pixels,
        evidence.confidence,
        samples[0],
        np.asarray(options["coordinate_scales"]),
        tuple(binding.fit["evidence"]["original_fit"]["free_coordinates"]),
        knots,
        None if start is not None else samples,
    ), start


def _dense_reprojection_metrics(
    native: MatchingPlant,
    camera: CameraProjection,
    attachments: dict[str, Any],
    evidence: CaptureImageEvidence,
    q: np.ndarray,
    fit_indices: tuple[int, ...],
) -> dict[str, float | None]:
    residuals = np.array(
        [
            camera.residual(
                native.marker_positions(pose, attachments), observed, weights
            )
            for pose, observed, weights in zip(
                q, evidence.observed_pixels, evidence.confidence, strict=True
            )
        ]
    )
    held_out = np.array([index not in fit_indices for index in evidence.frame_indices])
    result: dict[str, float | None] = {}
    for name, selected in (
        ("dense_rms_pixels", np.ones(len(q), dtype=bool)),
        ("held_out_rms_pixels", held_out),
    ):
        weight = float(np.sum(evidence.confidence[selected]))
        result[name] = (
            float(np.sqrt(np.sum(residuals[selected] ** 2) / weight))
            if weight > 0
            else None
        )
    return result


def compute_native_refit(request: dict[str, Any]) -> dict[str, Any]:
    """Compute a source-clock fit; never declare dynamics qualification."""
    stamp = fit_execution_stamp()
    expected = request["execution_stamp"]
    if (
        stamp["source_sha256"] != expected["source_sha256"]
        or stamp["runtime_sha256"] != expected["runtime_sha256"]
    ):
        raise ValueError(
            "Native worker implementation or runtime differs from launch stamp"
        )
    library = NecromatcherLibrary(request["library_root"])
    identity = request["source_fit_id"]
    if library.load_asset(identity).metadata["hash"] != request["source_fit_hash"]:
        raise ValueError("Warm-start fit differs from launch identity")
    binding = load_native_fit_binding(library, identity)
    source = binding.fit
    options = request["options"]
    operation = options.get("operation", "fit")
    config = ImageFitConfig.from_record(options["config"])
    ranges = _range_provenance(binding, config, operation)
    indices = tuple(options["frame_indices"])
    # Reuse native XML, camera and attachment validation from the review path.
    binding.project(indices[0])
    original = source["evidence"]["original_fit"]
    native = binding.plant
    camera, attachments = binding.review_inputs()
    samples = np.asarray(source["q"])[
        [source["frame_indices"].index(i) for i in indices]
    ]
    with CaptureReview(library, source["capture_id"]) as review:
        evidence = read_capture_evidence(
            review,
            tuple(attachments),
            indices,
            unknown_visibility_weight=options["unknown_visibility_weight"],
        )
        inputs, initial_spline = _worker_inputs(binding, options, evidence, samples)
        result = _compute_operation(
            operation,
            native,
            attachments,
            camera,
            inputs,
            config,
            initial_spline,
        )
        dense_indices = tuple(
            i for i in source["frame_indices"] if indices[0] <= i <= indices[-1]
        )
        dense = read_capture_evidence(
            review,
            tuple(attachments),
            dense_indices,
            unknown_visibility_weight=options["unknown_visibility_weight"],
        )
        q = result.evaluate_source_times(dense.source_times)
        frames = [review.frame(i)["frame"] for i in dense_indices]
    closure = np.array([native.closure_residuals(pose) for pose in q])
    output = _build_fit_payload(
        request,
        source,
        result,
        (dense_indices, frames, q),
        stamp,
        float(np.max(np.linalg.norm(closure, axis=1))),
        ranges,
    )
    output["evidence"]["original_fit"].update(
        _dense_reprojection_metrics(native, camera, attachments, dense, q, indices)
    )
    output["provenance"]["coordinate_bounds_provenance"] = ranges
    if fit_execution_stamp()["source_sha256"] != expected["source_sha256"]:
        raise ValueError("Native worker implementation changed during execution")
    return output


def _research_blockers(
    result: ImageFitResult, config: ImageFitConfig, ranges: dict[str, Any] | None
) -> list[str]:
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


def _fit_evidence(
    result: ImageFitResult,
    original: dict[str, Any],
    options: dict[str, Any],
    max_grip: float,
) -> dict[str, Any]:
    return {
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


def _build_fit_payload(
    request: dict[str, Any],
    source: dict[str, Any],
    result: ImageFitResult,
    dense_output: tuple[tuple[int, ...], list[dict[str, Any]], np.ndarray],
    stamp: dict[str, Any],
    max_grip: float,
    ranges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original = source["evidence"]["original_fit"]
    options = request["options"]
    dense_indices, frames, q = dense_output
    blockers = _research_blockers(
        result, ImageFitConfig.from_record(options["config"]), ranges
    )
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
            "original_fit": _fit_evidence(result, original, options, max_grip),
        },
    }
    return output


def main() -> None:
    """Serve one trusted local request and return its finite JSON fit."""
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        fit = compute_native_refit(request)
        sys.stdout.write(json.dumps({"fit": fit}, allow_nan=False))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        ImportError,
    ):
        logger.exception("Native research refit failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
