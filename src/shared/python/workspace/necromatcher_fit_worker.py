"""Clean-interpreter native refit producer; invoked by the owned job service."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

if __name__ == "__main__":
    import mujoco  # noqa: F401 -- clean worker loads SDK before workspace/native helpers

import numpy as np

from src.shared.python.motion_matching.historical_fit import (
    AdditionalImageResiduals,
    ShaftAxisEvidence,
    ShaftAxisResidualTerm,
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
from src.shared.python.estimation import CubicHermiteSplineTrajectory, SolverTelemetry
from .necromatcher import NecromatcherLibrary
from .necromatcher_fit_jobs import (
    fit_execution_stamp,
    _shaft_record,
    _parse_shaft_recipe,
    _verify_shaft_request,
)
from .necromatcher_shaft_evidence import BoundShaftEvidence, load_shaft_image_residuals
from .necromatcher_native import NativeFitBinding, load_native_fit_binding
from .necromatcher_review import CaptureReview
from .necromatcher_spline import preserved_fit_spline, verify_preserved_fit_samples
from .necromatcher_fit_metrics import (
    dense_reprojection_metrics as _dense_reprojection_metrics,
)
from .necromatcher_contacts import contact_schedule_binding
from .necromatcher_fit_records import (
    build_native_fit_payload as _build_fit_payload,
    native_research_blockers as _research_blockers,  # noqa: F401 -- compatibility
)
from .necromatcher_fit_telemetry import write_worker_telemetry

logger = logging.getLogger(__name__)


def _compute_operation(
    operation: str,
    native: MatchingPlant,
    attachments: dict[str, Any],
    camera: CameraProjection,
    inputs: ImageFitInputs,
    config: ImageFitConfig,
    initial_spline: ImageSplineStart | None = None,
    additional_images: AdditionalImageResiduals | None = None,
) -> ImageFitResult:
    if operation == "fit":
        arguments = (native, attachments, camera, inputs, config, initial_spline)
        return (
            fit_image_trajectory(*arguments, additional_images)
            if additional_images is not None
            else fit_image_trajectory(*arguments)
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

    result = (
        initialize_image_trajectory(
            native, attachments, camera, inputs, config, None, additional_images
        )
        if additional_images is not None
        else initialize_image_trajectory(native, attachments, camera, inputs, config)
    )
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
    verify_preserved_fit_samples(source, start)
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


def _load_refit_binding(
    library: NecromatcherLibrary, request: dict[str, Any]
) -> tuple[NativeFitBinding, AdditionalImageResiduals | None]:
    if "shaft_images" not in request:
        return load_native_fit_binding(library, request["source_fit_id"]), None
    recipe = request["shaft_images"]
    declared, weight = _parse_shaft_recipe(recipe)
    evidence = declared.evidence
    binding, bundle = load_shaft_image_residuals(
        library, request["source_fit_id"], evidence, weight
    )
    bound = BoundShaftEvidence(
        evidence,
        bundle.source_identity.source_clock_sha256,
        len(evidence.frames),
        sum(frame.segment.status == "observed" for frame in evidence.frames),
    )
    if _shaft_record(bound, recipe["unknown_visibility_weight"]) != recipe:
        raise ValueError("Worker shaft recipe binding differs from queue admission")
    return binding, bundle


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
    binding, additional_images = _load_refit_binding(library, request)
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
        contact_binding = contact_schedule_binding(config, source, review)
        evidence = read_capture_evidence(
            review,
            tuple(attachments),
            indices,
            unknown_visibility_weight=options["unknown_visibility_weight"],
        )
        inputs, initial_spline = _worker_inputs(binding, options, evidence, samples)
        arguments = (
            operation,
            native,
            attachments,
            camera,
            inputs,
            config,
            initial_spline,
        )
        result = (
            _compute_operation(*arguments, additional_images)
            if additional_images is not None
            else _compute_operation(*arguments)
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
    if additional_images is not None:
        term = additional_images.terms[0]
        if not isinstance(term, ShaftAxisResidualTerm):
            raise ValueError("Native shaft worker requires a typed shaft term")
        request = {**request, "shaft_axis": term.axis.to_record()}
        _verify_shaft_request(
            library, library.load_fit(identity), request["shaft_images"]
        )
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
    if contact_binding is not None:
        output["provenance"]["contact_schedule_binding"] = contact_binding
    if fit_execution_stamp()["source_sha256"] != expected["source_sha256"]:
        raise ValueError("Native worker implementation changed during execution")
    return output


def _emit_telemetry(
    path: Path,
    request: dict[str, Any],
    fit: dict[str, Any] | None,
    elapsed: float,
    reason: str,
) -> None:
    """Optional transport faults never replace a primary computation outcome."""
    try:
        write_worker_telemetry(path, request, fit, elapsed, reason)
    except (OSError, ValueError, TypeError, KeyError):
        logger.exception(
            "Optional telemetry publication failed; primary outcome retained"
        )
        if fit is not None:
            try:
                original = fit["evidence"]["original_fit"]
                if not isinstance(original, dict):
                    raise TypeError("Original fit evidence must be a mapping")
                value = SolverTelemetry.from_record(original.get("solver_telemetry"))
                original["solver_telemetry"] = replace(
                    value,
                    worker_elapsed_s=None,
                    unavailable_reason="worker_telemetry_publication_failed",
                ).to_record()
            except (ValueError, TypeError, KeyError):
                logger.exception(
                    "Malformed result telemetry retained for strict admission rejection"
                )


def main() -> None:
    """Serve one trusted local request and return its finite JSON fit."""
    started = time.perf_counter()
    request = None
    request_path = None
    try:
        request_path = Path(sys.argv[1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        fit = compute_native_refit(request)
        _emit_telemetry(
            request_path,
            request,
            fit,
            time.perf_counter() - started,
            "Worker computed fit payload",
        )
        sys.stdout.write(json.dumps({"fit": fit}, allow_nan=False))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        ImportError,
    ) as exc:
        if request is not None and request_path is not None:
            _emit_telemetry(
                request_path, request, None, time.perf_counter() - started, str(exc)
            )
        logger.exception("Native research refit failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
