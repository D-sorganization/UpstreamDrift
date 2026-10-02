"""Independently audit saved source-clock splines; never perform fitting.

Run this script in a clean interpreter. Its CLI imports MuJoCo before workspace
modules and writes a new receipt exclusively outside the immutable library.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from zipfile import ZipFile

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintLinearization,
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane
    from src.shared.python.motion_matching.historical_fit import ImageFitResult
    from src.shared.python.workspace import NecromatcherLibrary, NativeFitBinding
    from src.shared.python.workspace.necromatcher_ranges import AuthoredCoordinateBounds

_POSE_BATCH = 32


class NativeAuditIK(Protocol):
    """Only public native geometry capabilities are consumed by the auditor."""

    def constraint_residual_jacobian(
        self, q: NDArray[np.float64], options: ConstraintOptions
    ) -> ConstraintLinearization: ...
    def closure_error(self, q: NDArray[np.float64]) -> tuple[float, float]: ...
    def sphere_heights(
        self, q: NDArray[np.float64], ground: GroundPlane
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class _Point:
    time: float
    kind: Literal[
        "source_frame", "adjacent_source_midpoint", "coordinate_global_extremum"
    ]
    indices: tuple[int, ...]
    rational: Fraction | None = None
    frame: Mapping[str, Any] | None = None
    coordinate: str | None = None
    extremum: str | None = None


def _fraction(frame: Mapping[str, Any]) -> Fraction:
    ticks, numerator, denominator = (
        frame[name]
        for name in ("pts_ticks", "timebase_numerator", "timebase_denominator")
    )
    if (
        any(type(value) is not int for value in (ticks, numerator, denominator))
        or min(numerator, denominator) <= 0
    ):
        raise ValueError(
            "Audit requires exact integer container PTS and positive timebase"
        )
    return Fraction(ticks * numerator, denominator)


def _source_points(
    frames: Sequence[tuple[int, Mapping[str, Any]]], interval: tuple[float, float]
) -> list[_Point]:
    points = [
        _Point(
            float(_fraction(frame)),
            "source_frame",
            (index,),
            _fraction(frame),
            deepcopy(frame),
        )
        for index, frame in frames
    ]
    if len(points) < 2 or any(
        right.time <= left.time or right.indices[0] <= left.indices[0]
        for left, right in pairwise(points)
    ):
        raise ValueError("Audit source PTS and frame indices must strictly increase")
    if points[0].time != interval[0] or points[-1].time != interval[1]:
        raise ValueError(
            "Audit source frames must span the exact preserved source interval"
        )
    return points


def _sample_points(sources: list[_Point], assessment: Any) -> list[_Point]:
    points = list(sources)
    for left, right in pairwise(sources):
        if left.rational is None or right.rational is None:
            raise ValueError("Source midpoint requires exact rational parents")
        rational = (left.rational + right.rational) / 2
        points.append(
            _Point(
                float(rational),
                "adjacent_source_midpoint",
                (left.indices[0], right.indices[0]),
                rational,
            )
        )
    for item in assessment.coordinates:
        points.extend(
            (
                _Point(
                    item.minimum_source_time,
                    "coordinate_global_extremum",
                    (),
                    coordinate=item.name,
                    extremum="minimum",
                ),
                _Point(
                    item.maximum_source_time,
                    "coordinate_global_extremum",
                    (),
                    coordinate=item.name,
                    extremum="maximum",
                ),
            )
        )
    return points


def _evaluate(
    result: ImageFitResult, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    import numpy as np

    return np.vstack(
        [
            result.evaluate_source_times(times[start : start + _POSE_BATCH])
            for start in range(0, len(times), _POSE_BATCH)
        ]
    )


def _certificate(
    result: ImageFitResult, bounds: AuthoredCoordinateBounds, assessment: Any
) -> dict[str, Any]:
    from src.shared.python.estimation import HermiteBoundsDomain

    try:
        domain = HermiteBoundsDomain(
            tuple(map(float, result.knot_times)),
            tuple(bounds.named_bounds.get(name) for name in result.free_coordinates),
        )
        domain.encode(result.spline_coefficients)
        verified = not assessment.violating_coordinates
        reason = (
            "Canonical Bernstein domain and scalar extrema agree"
            if verified
            else "Locked or scalar extrema violate authored bounds"
        )
    except ValueError as exc:
        verified, reason = False, str(exc)
    return {
        "method": "canonical_HermiteBoundsDomain.encode",
        "conservative_bernstein_q_domain_verified": verified,
        "bounded_coordinates": list(assessment.bounded_coordinates),
        "unbounded_coordinates": list(assessment.unbounded_coordinates),
        "reason": reason,
        "continuous_nonlinear_certified": False,
    }


def _native_row(
    ik: NativeAuditIK,
    pose: NDArray[np.float64],
    point: _Point,
    options: ConstraintOptions,
) -> dict[str, Any]:
    import numpy as np

    value = ik.constraint_residual_jacobian(pose, options)
    expected = tuple(f"grip_position:{axis}" for axis in "xyz") + tuple(
        f"grip_rotation:{axis}" for axis in "xyz"
    )
    if value.row_labels[:6] != expected:
        raise ValueError(
            "Native constraint row identities do not match grip XYZ convention"
        )
    gap, angle = map(
        float, (np.linalg.norm(value.residual[:3]), np.linalg.norm(value.residual[3:6]))
    )
    if not np.allclose((gap, angle), ik.closure_error(pose), atol=1e-10, rtol=1e-10):
        raise ValueError("Native grip methods disagree")
    heights = ik.sphere_heights(pose, options.ground)
    if not heights or not np.isfinite(list(heights.values())).all():
        raise ValueError("Native ground sphere heights must be finite and declared")
    labels = tuple(f"ground:{name}" for name in heights)
    if value.row_labels[6:] != labels or not np.allclose(
        value.residual[6:],
        [min(0.0, height) for height in heights.values()],
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("Native ground methods disagree")
    rational = point.rational
    return {
        "kind": point.kind,
        "source_time": point.time,
        "pts_rational": None
        if rational is None
        else {"numerator": rational.numerator, "denominator": rational.denominator},
        "time_representation": "exact_container_pts"
        if rational is not None
        else "canonical_float_source_clock",
        "source_frame_indices": list(point.indices),
        "source_frame": None if point.frame is None else dict(point.frame),
        "coordinate": point.coordinate,
        "extremum": point.extremum,
        "grip_gap_m": gap,
        "grip_rotation_rad": angle,
        "grip_angle_deg": float(np.rad2deg(angle)),
        "ground_penetration_m": max(0.0, -min(heights.values())),
        "contact_sphere_heights_m": heights,
        "image_objective_point_added": False,
    }


def audit_trajectory(
    result: ImageFitResult,
    bounds: AuthoredCoordinateBounds,
    source_frames: Sequence[tuple[int, Mapping[str, Any]]],
    ik: NativeAuditIK,
    ground: GroundPlane,
) -> dict[str, Any]:
    """Audit canonical q extrema and finite nonlinear samples as separate claims."""
    import numpy as np
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )

    assessment = assess_spline_bounds(result, bounds.named_bounds)
    sources = _source_points(source_frames, assessment.source_interval)
    points = _sample_points(sources, assessment)
    poses = _evaluate(result, np.asarray([point.time for point in points]))
    options = ConstraintOptions(ground, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    rows = [
        _native_row(ik, pose, point, options)
        for pose, point in zip(poses, points, strict=True)
    ]
    extrema = {
        **asdict(assessment),
        "bounded_coordinates": list(assessment.bounded_coordinates),
        "unbounded_coordinates": list(assessment.unbounded_coordinates),
        "violating_coordinates": list(assessment.violating_coordinates),
    }
    return {
        "authored_bounds": bounds.to_record(),
        "q_bound_certificate": _certificate(result, bounds, assessment),
        "coordinate_extrema": extrema,
        "source_frame_count": len(sources),
        "midpoint_count": len(sources) - 1,
        "canonical_global_extrema_count": 2 * len(assessment.coordinates),
        "sampled_native_constraints": rows,
        "maximum_sampled_grip_gap_m": max(row["grip_gap_m"] for row in rows),
        "maximum_sampled_grip_rotation_rad": max(
            row["grip_rotation_rad"] for row in rows
        ),
        "maximum_sampled_ground_penetration_m": max(
            row["ground_penetration_m"] for row in rows
        ),
        "continuous_nonlinear_certified": False,
        "scientific_acceptance": False,
        "physical_time_qualified": False,
        "optimization_performed": False,
        "saved_optimizer_ran": result.optimizer_ran,
        "saved_optimizer_converged": result.converged,
    }


def _pixel_metrics(
    binding: NativeFitBinding, poses: NDArray[np.float64], evidence: Any
) -> tuple[float, NDArray[Any]]:
    import numpy as np

    camera, attachments = binding.review_inputs()
    residuals = np.asarray(
        [
            camera.residual(
                binding.plant.marker_positions(pose, attachments), observed, weights
            )
            for pose, observed, weights in zip(
                poses, evidence.observed_pixels, evidence.confidence, strict=True
            )
        ]
    )
    weight_sum = float(np.sum(evidence.confidence))
    if weight_sum <= 0:
        raise ValueError(
            "Preserved image objective requires observed confidence weight"
        )
    rms = float(np.sqrt(np.sum(residuals**2) / weight_sum))
    errors = np.empty(evidence.confidence.shape, dtype=object)
    errors[:] = None
    present = evidence.confidence > 0
    weighted = residuals.reshape(evidence.confidence.shape + (2,))
    errors[present] = np.sqrt(
        np.sum(weighted[present] ** 2, axis=1) / evidence.confidence[present]
    )
    return rms, errors


def _reconstruct(binding: NativeFitBinding, evidence: Any) -> ImageFitResult:
    import numpy as np
    from src.shared.python.motion_matching.historical_fit import ImageFitResult

    original = binding.fit["evidence"]["original_fit"]
    if not np.array_equal(evidence.source_times, original["source_times"]):
        raise ValueError("Original source PTS differ from preserved spline samples")
    poses = np.asarray(original["q"], dtype=float)
    rms, errors = _pixel_metrics(binding, poses, evidence)
    if not np.isclose(rms, original["rms_pixels"], rtol=1e-10, atol=1e-10):
        raise ValueError("Original observed RMS differs from preserved result")
    saved = original.get("constraint_assessment", {})
    times = np.asarray(saved.get("tested_times", []), dtype=float)
    labels = tuple(saved.get("row_labels", []))
    return ImageFitResult(
        source_times=evidence.source_times,
        q=poses,
        rms_pixels=rms,
        initial_rms_pixels=original["initial_rms_pixels"],
        pixel_errors=errors,
        observed_point_count=int(np.count_nonzero(evidence.confidence)),
        model_sha=binding.plant.plant_sha,
        coordinate_order=tuple(original["coordinate_order"]),
        converged=original["converged"],
        optimizer_message=original["message"],
        knot_times=np.asarray(original["knot_times"]),
        spline_coefficients=np.asarray(original["spline_coefficients"]),
        free_coordinates=tuple(original["free_coordinates"]),
        constraint_times=times,
        constraint_residuals=np.asarray(saved.get("scaled_residuals", [])).reshape(
            len(times), len(labels)
        ),
        constraint_row_labels=labels,
        optimizer_ran=original.get("optimizer_ran", True),
    )


def _check_parents(library: NecromatcherLibrary, binding: NativeFitBinding) -> None:
    for identity, expected in (
        (binding.fit_id, binding.fit_hash),
        (binding.model_id, binding.model_hash),
        (binding.fit["capture_id"], binding.fit["capture_hash"]),
    ):
        if library.load_asset(identity).metadata["hash"] != expected:
            raise ValueError("Bound audit parent identity changed")
    library.load_fit(binding.fit_id)


def _verify_audit_stamp(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    script_before: str,
    script_after: str,
) -> None:
    if (
        any(before[key] != after[key] for key in ("source_sha256", "runtime_sha256"))
        or script_before != script_after
    ):
        raise ValueError("Audit implementation or runtime changed during assessment")


def audit_saved_fit(library: NecromatcherLibrary, fit_id: str) -> dict[str, Any]:
    """Audit one existing saved fit with source and parent identities bracketed."""
    import numpy as np
    import json
    from src.shared.python.workspace import load_native_fit_binding
    from src.shared.python.workspace.necromatcher_review import CaptureReview
    from src.shared.python.workspace.necromatcher_fit_jobs import fit_execution_stamp
    from src.shared.python.workspace.artifact_handoff import compute_file_sha256
    from src.shared.python.motion_matching.historical_fit import (
        ImageFitConfig,
        read_capture_evidence,
    )

    before, script_hash = fit_execution_stamp(), compute_file_sha256(Path(__file__))
    binding = load_native_fit_binding(library, fit_id)
    _check_parents(library, binding)
    fit, original = binding.fit, binding.fit["evidence"]["original_fit"]
    _, attachments = binding.review_inputs()
    unknown = fit["provenance"]["request_options"]["unknown_visibility_weight"]
    with CaptureReview(library, fit["capture_id"]) as review:
        evidence = read_capture_evidence(
            review,
            tuple(attachments),
            tuple(original["frame_indices"]),
            unknown_visibility_weight=unknown,
        )
        result = _reconstruct(binding, evidence)
        interval = result.source_times[0], result.source_times[-1]
        frames = [
            (index, review.frame(index)["frame"]) for index in range(review.frame_count)
        ]
        frames = [
            (index, frame)
            for index, frame in frames
            if interval[0] <= float(_fraction(frame)) <= interval[1]
        ]
        dense = read_capture_evidence(
            review,
            tuple(attachments),
            tuple(index for index, _ in frames),
            unknown_visibility_weight=unknown,
        )
    dense_poses = _evaluate(result, dense.source_times)
    dense_rms, _ = _pixel_metrics(binding, dense_poses, dense)
    stored = _evaluate(
        result, np.asarray([float(_fraction(frame)) for frame in fit["frames"]])
    )
    if not np.allclose(stored, fit["q"], rtol=1e-8, atol=1e-10):
        raise ValueError("Saved dense poses disagree with preserved spline")
    options = ImageFitConfig.from_record(original["config"]).constraint_options
    if options is None:
        raise ValueError("Audit requires an explicitly declared ground plane")
    ik = cast(NativeAuditIK, binding.plant.create_ik(attachments))
    report = audit_trajectory(
        result, binding.authored_coordinate_bounds(), frames, ik, options.ground
    )
    capture = library.load_asset(fit["capture_id"])
    with ZipFile(capture.path) as archive:
        source = json.loads(archive.read("receipt.json"))["source"]
    _check_parents(library, binding)
    after = fit_execution_stamp()
    _verify_audit_stamp(before, after, script_hash, compute_file_sha256(Path(__file__)))
    return {
        "schema": "necromatcher/independent-bounded-trial-audit/1",
        "fit_id": fit_id,
        "fit_hash": binding.fit_hash,
        "model_id": binding.model_id,
        "model_hash": binding.model_hash,
        "capture_id": fit["capture_id"],
        "capture_hash": fit["capture_hash"],
        "source_video_identity": source,
        "source_video_file_rehashed": False,
        "producer_execution_stamp": fit["provenance"].get("execution_stamp"),
        "audit_execution_stamp": before,
        "audit_source_runtime_unchanged": True,
        "audit_script_sha256": script_hash,
        "observed_original_rms_pixels": result.rms_pixels,
        "dense_original_rms_pixels": dense_rms,
        "dense_observed_point_count": int(np.count_nonzero(dense.confidence)),
        "added_image_objective_points": 0,
        **report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Audit explicit saved identities in a clean SDK process; publish exclusively."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", required=True, type=Path)
    parser.add_argument("--fit-id", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    for entry in (
        root,
        root / "src",
        root / "src/shared/python",
        root / "vendor/ud-tools/src",
    ):
        sys.path.insert(0, str(entry))
    if any(name.startswith("src.shared.python.workspace") for name in sys.modules):
        raise RuntimeError(
            "Audit CLI requires a clean interpreter before workspace import"
        )
    import mujoco  # noqa: F401 -- SDK must initialize before workspace/Qt imports
    import json
    from src.shared.python.workspace import NecromatcherLibrary
    from src.shared.python.workspace.artifact_handoff import compute_file_sha256

    library = NecromatcherLibrary(args.library_root)
    if (
        args.output.resolve().is_relative_to(library.root.resolve())
        or args.output.exists()
    ):
        raise ValueError(
            "Audit output must be a new path outside the immutable library"
        )
    report = {
        "schema": "necromatcher/independent-bounded-trial-audit-batch/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sdk_imported_before_workspace": True,
        "optimization_performed": False,
        "runs": [audit_saved_fit(library, identity) for identity in args.fit_id],
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    sys.stdout.write(
        json.dumps(
            {"output": str(args.output), "sha256": compute_file_sha256(args.output)}
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
