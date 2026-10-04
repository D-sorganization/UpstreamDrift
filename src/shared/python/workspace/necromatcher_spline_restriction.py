"""Authenticated, SDK-free admission of lossless source-clock spline restrictions.

The pure receipt proves coefficient lineage, not capture or physical qualification.
Library callers must keep their bounded authenticated-read context open for reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from typing import Any

import numpy as np
from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    HermiteBoundsDomain,
)
from src.shared.python.motion_matching.historical_fit.contracts import (
    ImageFitConfig,
    ImageFitResult,
    ImageSplineStart,
)
from src.shared.python.motion_matching.historical_fit.spline_restriction import (
    SplineIntervalRestriction,
    restrict_image_spline_interval,
)
from src.shared.python.shadow_tracker import FrameIdentity
from .necromatcher_spline import preserved_fit_spline, verify_preserved_fit_samples


def _equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _recipe(options: Mapping[str, Any]) -> tuple[tuple[int, ...], ImageFitConfig]:
    from .necromatcher_fit_jobs import NativeRefitOptions

    values = dict(options)
    values["frame_indices"] = tuple(values["frame_indices"])
    values["coordinate_scales"] = tuple(values["coordinate_scales"])
    values["config"] = ImageFitConfig.from_record(values["config"])
    parsed = NativeRefitOptions(**values)
    if set(options) != set(asdict(parsed)) or not _equal(
        options["config"], asdict(parsed.config)
    ):
        raise ValueError("Restriction requires a complete canonical configuration")
    if (
        parsed.operation != "restrict_initialization"
        or parsed.initialization_source != "restricted_spline"
        or parsed.config.initialization_policy != "strict"
    ):
        raise ValueError("Restriction requires explicit strict restriction recipe")
    return parsed.frame_indices, parsed.config


def _parent_start(source: Mapping[str, Any]) -> ImageSplineStart:
    start = preserved_fit_spline(source)
    if start is None:
        raise ValueError("Restriction requires preserved parent coefficients")
    expected = verify_preserved_fit_samples(source, start)
    q = np.asarray(source["q"], dtype=float)
    if not np.allclose(q, expected, rtol=0.0, atol=1e-12):
        raise ValueError("Restriction parent source curve differs beyond roundoff")
    locked = [
        i
        for i, n in enumerate(start.coordinate_order)
        if n not in start.free_coordinates
    ]
    if not np.allclose(q[:, locked], q[0, locked], rtol=0.0, atol=1e-12):
        raise ValueError("Restriction requires constant whole-parent locked reference")
    return start


def spline_restriction_prior(options: Mapping[str, Any]) -> dict[str, Any]:
    """Declare the unchanged selected-first saved parent pose as the prior."""
    indices, _ = _recipe(options)
    return {"policy": "selected_first_parent_pose", "frame_index": indices[0]}


def _strict_bounds(
    start: ImageSplineStart, source: Mapping[str, Any], config: ImageFitConfig
) -> None:
    bounds = {name: (lo, hi) for name, lo, hi in config.coordinate_bounds}
    if set(bounds) - set(start.coordinate_order):
        raise ValueError("Restriction bounds name an unknown coordinate")
    free = [start.coordinate_order.index(n) for n in start.free_coordinates]
    trajectory = CubicHermiteSplineTrajectory(np.asarray(start.knot_times), len(free))
    positions, velocities = trajectory.unpack(np.asarray(start.spline_coefficients))
    full_q = np.tile(
        np.asarray(source["q"], dtype=float)[0], (len(start.knot_times), 1)
    )
    full_v = np.zeros_like(full_q)
    full_q[:, free], full_v[:, free] = positions, velocities
    full_trajectory = CubicHermiteSplineTrajectory(
        np.asarray(start.knot_times), len(start.coordinate_order)
    )
    domain = HermiteBoundsDomain(
        start.knot_times, tuple(bounds.get(n) for n in start.coordinate_order)
    )
    domain.encode(full_trajectory.pack(full_q, full_v))


def derive_spline_restriction(
    library: Any,
    source_fit_id: str,
    options: Mapping[str, Any],
    source_scope: Any = None,
) -> SplineIntervalRestriction:
    """Rebind parent and scope; return exact retained-knots coefficient lineage.

    Postcondition: selected endpoints equal the authenticated reviewed window and
    the retained knot count equals the declared recipe. No native or time
    qualification is inferred from this SDK-free admission.
    """
    from .necromatcher_fit import admit_refit_scope

    indices, config = _recipe(options)
    source = library.load_fit(source_fit_id)
    start = _parent_start(source)
    bound = admit_refit_scope(library, source, indices, config, requested=source_scope)
    if bound is None or (indices[0], indices[-1]) != (
        bound.scope.first_frame,
        bound.scope.end_exclusive_frame - 1,
    ):
        raise ValueError("Restriction selection must span the exact reviewed scope")
    domain = bound.selected_domain(indices)
    receipt = restrict_image_spline_interval(
        start, float(domain.first_pts), float(domain.last_pts)
    )
    if len(receipt.restricted_start.knot_times) != options["knot_count"]:
        raise ValueError("Restriction knot count differs from retained parent knots")
    _strict_bounds(receipt.restricted_start, source, config)
    return receipt


def _pose_samples(
    source: Mapping[str, Any], start: ImageSplineStart, indices: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frames = dict(zip(source["frame_indices"], source["frames"], strict=True))
    selected = [frames[i] for i in indices]
    times = np.array(
        [float(FrameIdentity.from_dict(f).presentation_time) for f in selected]
    )
    q = np.tile(np.asarray(source["q"], dtype=float)[0], (len(times), 1))
    free = [start.coordinate_order.index(n) for n in start.free_coordinates]
    trajectory = CubicHermiteSplineTrajectory(np.asarray(start.knot_times), len(free))
    q[:, free] = trajectory.evaluate(np.asarray(start.spline_coefficients), times).q
    return times, q, selected


def validate_spline_restriction_request(
    library: Any, request: Mapping[str, Any]
) -> SplineIntervalRestriction | None:
    """Authenticate opt-in request lineage before compilation or publication.

    Legacy requests without either restriction key remain unchanged. A receipt
    or prior on any other mode is rejected rather than ignored.
    """
    from .necromatcher_source_scope import SourceFitScope

    options = request["options"]
    if (
        options.get("operation") != "restrict_initialization"
        and options.get("initialization_source") != "restricted_spline"
        and not any(
            k in request
            for k in ("spline_interval_restriction", "spline_restriction_prior")
        )
    ):
        return None
    indices, _ = _recipe(options)
    scope = SourceFitScope.from_record(request["source_scope"])
    expected = derive_spline_restriction(
        library, request["source_fit_id"], options, scope
    )
    if not _equal(
        request.get("spline_interval_restriction"), expected.to_record()
    ) or not _equal(
        request.get("spline_restriction_prior"), spline_restriction_prior(options)
    ):
        raise ValueError(
            "Restriction request receipt or prior differs from canonical derivation"
        )
    if (
        request["source_fit_hash"]
        != library.load_asset(request["source_fit_id"]).metadata["hash"]
    ):
        raise ValueError("Restriction request parent asset hash differs")
    source = library.load_fit(request["source_fit_id"])
    _, _, frames = _pose_samples(source, expected.original_start, indices)
    first, last = (
        FrameIdentity.from_dict(f).presentation_time for f in (frames[0], frames[-1])
    )
    binding = {
        "frame_indices": list(indices),
        "first_pts": [first.numerator, first.denominator],
        "last_pts": [last.numerator, last.denominator],
        "source_clock_sha256": scope.source_clock_sha256,
    }
    if not _equal(request.get("source_scope_binding"), binding):
        raise ValueError("Restriction request selected-domain binding differs")
    return expected


def validate_restriction_seed_result(
    request: Mapping[str, Any],
    source: Mapping[str, Any],
    result: ImageFitResult,
    dense_output: Any,
) -> None:
    """Reject forged/unoptimized-seed results before serialization.

    Caller supplies a freshly authenticated source; this check cannot authenticate
    arbitrary mappings. Source-clock samples preserve the parent curve to 1e-12;
    acceleration at retained knots remains the pure receipt's one-sided convention.
    """
    declared = request.get("spline_interval_restriction")
    if (
        declared is None
        and "spline_restriction_prior" not in request
        and request["options"].get("operation") != "restrict_initialization"
        and request["options"].get("initialization_source") != "restricted_spline"
    ):
        return
    indices, config = _recipe(request["options"])
    if not isinstance(declared, Mapping):
        raise ValueError("Restriction requires a typed receipt record")
    receipt = SplineIntervalRestriction.from_record(declared)
    start = _parent_start(source)
    times, q, _ = _pose_samples(source, start, indices)
    expected = restrict_image_spline_interval(start, float(times[0]), float(times[-1]))
    if (
        receipt != expected
        or len(receipt.restricted_start.knot_times) != request["options"]["knot_count"]
    ):
        raise ValueError("Restriction receipt differs from canonical parent interval")
    prior = spline_restriction_prior(request["options"])
    if not _equal(request.get("spline_restriction_prior"), prior):
        raise ValueError("Restriction prior differs from selected-first parent pose")
    actual = ImageSplineStart.from_coefficients(
        result.knot_times,
        result.spline_coefficients,
        result.coordinate_order,
        result.free_coordinates,
        result.model_sha,
    )
    if actual != receipt.restricted_start or result.initial_spline != actual:
        raise ValueError(
            "Restriction result snapshot differs from exact restricted start"
        )
    _strict_bounds(actual, source, config)
    if (
        result.optimizer_ran
        or result.converged
        or result.initialization is not None
        or result.initial_rms_pixels != result.rms_pixels
    ):
        raise ValueError(
            "Restriction-only seed cannot claim optimization or projection"
        )
    if result.telemetry is not None and any(
        getattr(result.telemetry, name) is not None for name in ("nfev", "njev")
    ):
        raise ValueError("Restriction-only result cannot report optimizer counts")
    if (
        not np.array_equal(times, result.source_times)
        or result.q.shape != q.shape
        or not np.allclose(q, result.q, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Restriction training poses or source clock differ")
    dense_indices, dense_frames, dense_q = dense_output
    if tuple(dense_indices) != tuple(range(indices[0], indices[-1] + 1)):
        raise ValueError("Restriction dense support differs from selected interval")
    _, expected_q, expected_frames = _pose_samples(source, start, tuple(dense_indices))
    if (
        not _equal(dense_frames, expected_frames)
        or np.asarray(dense_q).shape != expected_q.shape
        or not np.allclose(dense_q, expected_q, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Restriction dense frames or poses differ from parent curve")


def validate_spline_restriction_payload(
    library: Any, payload: Mapping[str, Any]
) -> None:
    """Canonically rederive stored restriction lineage; never trust its DTO alone."""
    provenance = payload["provenance"]
    declared = provenance.get("spline_interval_restriction")
    if (
        declared is None
        and "spline_restriction_prior" not in provenance
        and provenance.get("operation") != "restrict_initialization"
        and provenance.get("request_options", {}).get("operation")
        != "restrict_initialization"
        and provenance.get("request_options", {}).get("initialization_source")
        != "restricted_spline"
        and payload["evidence"].get("original_fit", {}).get("initialization_source")
        != "restricted_spline"
        and "spline_interval_restriction_only"
        not in payload["evidence"].get("rejection_reasons", ())
    ):
        return
    source_id = provenance["warm_start_fit_id"]
    source = library.load_fit(source_id)
    if (
        provenance["warm_start_fit_hash"]
        != library.load_asset(source_id).metadata["hash"]
    ):
        raise ValueError("Restriction parent asset hash differs")
    options = provenance["request_options"]
    if provenance.get("operation") != "restrict_initialization":
        raise ValueError("Restriction metadata requires restriction-only operation")
    expected = validate_spline_restriction_request(
        library,
        {
            "source_fit_id": source_id,
            "source_fit_hash": provenance["warm_start_fit_hash"],
            "options": options,
            "source_scope": provenance["source_fit_scope"],
            "source_scope_binding": provenance["source_fit_scope_binding"],
            "spline_interval_restriction": declared,
            "spline_restriction_prior": provenance.get("spline_restriction_prior"),
        },
    )
    if expected is None:
        raise ValueError("Restriction request unexpectedly absent")
    if (
        not _equal(declared, expected.to_record())
        or preserved_fit_spline(payload) != expected.restricted_start
    ):
        raise ValueError("Stored restriction receipt differs from canonical lineage")
    for key in (
        "model_id",
        "model_hash",
        "capture_id",
        "capture_hash",
        "coordinate_order",
        "coordinate_units",
    ):
        if not _equal(payload[key], source[key]):
            raise ValueError("Restriction source/model identity differs")
    original = payload["evidence"]["original_fit"]
    indices, _ = _recipe(options)
    times, q, _ = _pose_samples(source, expected.original_start, indices)
    if (
        not _equal(original["frame_indices"], indices)
        or not np.array_equal(original["source_times"], times)
        or not np.allclose(original["q"], q, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Stored restriction training samples differ")
    if (
        original["optimizer_ran"] is not False
        or original["converged"] is not False
        or original["initialization"] is not None
    ):
        raise ValueError(
            "Stored restriction seed falsely claims optimized initialization"
        )
    if not _equal(
        provenance.get("spline_restriction_prior"),
        spline_restriction_prior(options),
    ):
        raise ValueError("Stored restriction prior differs")
    _validate_stored_seed(payload, source, expected, options)


def _validate_stored_seed(
    payload: Mapping[str, Any],
    source: Mapping[str, Any],
    receipt: SplineIntervalRestriction,
    options: Mapping[str, Any],
) -> None:
    original = payload["evidence"]["original_fit"]
    indices = tuple(options["frame_indices"])
    dense = tuple(range(indices[0], indices[-1] + 1))
    _, q, frames = _pose_samples(source, receipt.original_start, dense)
    actual_q = np.asarray(payload["q"], dtype=float)
    if (
        not _equal(payload["frame_indices"], dense)
        or not _equal(payload["frames"], frames)
        or actual_q.shape != q.shape
        or not np.allclose(q, actual_q, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Stored restriction dense source curve differs")
    if (
        not _equal(original["initial_spline"], receipt.restricted_start.to_record())
        or original["initial_coefficient_sha256"]
        != receipt.restricted_start.coefficient_sha256
    ):
        raise ValueError("Stored restriction initial snapshot differs")
    if (
        not _equal(original["config"], options["config"])
        or original["initial_rms_pixels"] != original["rms_pixels"]
    ):
        raise ValueError("Stored restriction configuration or initial RMS differs")
    parent_original = source["evidence"]["original_fit"]
    if any(
        not _equal(original[k], parent_original[k]) for k in ("camera", "attachments")
    ):
        raise ValueError("Restriction cannot replace parent camera or attachments")
    telemetry = original.get("solver_telemetry")
    if telemetry is not None and any(
        telemetry.get(k) is not None for k in ("nfev", "njev")
    ):
        raise ValueError("Restriction-only seed cannot report optimizer counts")
    blockers = payload["evidence"]["rejection_reasons"]
    if (
        "spline_interval_restriction_only" not in blockers
        or "authored_initialization_only" in blockers
    ):
        raise ValueError("Stored restriction-only qualification blocker differs")
