"""Validation of source-bound, unqualified native coordinate samples.

These records preserve research results, not calibrated motion or controls.
Coordinate units are declared by the producer; model compilation and scientific
acceptance are separate requirements. Source frame identities remain exact.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any
from fractions import Fraction

import numpy as np
from src.shared.python.estimation import SolverTelemetry

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary

FIT_SCHEMA = "necromatcher/kinematic-fit/1"
_FIELDS = frozenset(
    {
        "schema_version",
        "qualification",
        "physical_time_qualified",
        "dynamics_replayed",
        "model_id",
        "model_hash",
        "capture_id",
        "capture_hash",
        "coordinate_order",
        "coordinate_units",
        "frame_indices",
        "frames",
        "q",
        "provenance",
        "evidence",
    }
)


def _validate_solver_telemetry(evidence: dict[str, Any]) -> None:
    """New optional measurements are strict; untouched legacy evidence is absent."""
    original = evidence.get("original_fit")
    if isinstance(original, dict) and "solver_telemetry" in original:
        if original["solver_telemetry"] is None:
            raise ValueError("Present telemetry must contain its complete schema")
        telemetry = SolverTelemetry.from_record(original["solver_telemetry"])
        if original.get("optimizer_ran") is False and (
            telemetry.nfev is not None or telemetry.njev is not None
        ):
            raise ValueError("Unoptimized output cannot claim solver counts")


def read_kinematic_fit(
    source: Path, library: NecromatcherLibrary, swing_id: str
) -> dict[str, Any]:
    """Verify bytes, parent versions and exact source frames before publication."""
    from .necromatcher_review import CaptureReview

    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("Fit must contain exactly the kinematic-fit schema fields")
    if (
        payload["schema_version"] != FIT_SCHEMA
        or payload["qualification"] != "monocular_research_hypothesis"
        or payload["physical_time_qualified"] is not False
        or payload["dynamics_replayed"] is not False
    ):
        raise ValueError("Fit storage accepts only unqualified kinematic research")
    for name, kind in (("model", "native_model"), ("capture", "image_capture")):
        if not isinstance(payload[f"{name}_id"], str):
            raise ValueError("Fit parent identity must be a string")
        asset = library.load_asset(payload[f"{name}_id"])
        if asset.kind != kind or asset.session_id != swing_id:
            raise ValueError("Fit parents must belong to the same swing session")
        if payload[f"{name}_hash"] != asset.metadata["hash"]:
            raise ValueError("Fit parent hash mismatch")
        if name == "model" and payload["coordinate_order"] != asset.metadata["dofs"]:
            raise ValueError("Fit coordinate order must match its model version")
    order, units = payload["coordinate_order"], payload["coordinate_units"]
    if (
        not isinstance(units, list)
        or len(units) != len(order)
        or any(not isinstance(unit, str) or unit not in {"rad", "m"} for unit in units)
    ):
        raise ValueError("Fit coordinates require declared rad or m units")
    indices, frames = payload["frame_indices"], payload["frames"]
    if (
        not isinstance(indices, list)
        or len(indices) < 2
        or any(type(index) is not int or index < 0 for index in indices)
        or any(b <= a for a, b in zip(indices, indices[1:], strict=False))
        or not isinstance(frames, list)
        or len(frames) != len(indices)
    ):
        raise ValueError(
            "Fit frame indices must be increasing with matching identities"
        )
    try:
        coordinates = np.asarray(payload["q"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Fit coordinates must be a finite numeric matrix") from exc
    if (
        coordinates.dtype.kind not in "ifu"
        or coordinates.shape != (len(indices), len(order))
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("Fit coordinates must be finite in frame and model order")
    provenance = payload["provenance"]
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("description"), str)
        or not provenance["description"].strip()
        or not isinstance(payload["evidence"], dict)
    ):
        raise ValueError("Fit requires explicit provenance and research evidence")
    # Reject non-finite auxiliary evidence as well; retain the original JSON bytes.
    _validate_solver_telemetry(payload["evidence"])
    json.dumps(payload, allow_nan=False)
    with CaptureReview(library, payload["capture_id"]) as review:
        for index, identity in zip(indices, frames, strict=True):
            if index >= review.frame_count or identity != review.frame(index)["frame"]:
                raise ValueError("Fit source frame identity mismatch")
    from .necromatcher_placement import validate_placement_lineage

    validate_placement_lineage(payload, library, swing_id)
    from .necromatcher_hypothesis import validate_hypothesis_seed

    validate_hypothesis_seed(library, payload)
    parent = _scope_parent(library, payload)
    validate_scope_payload(library, payload, parent)
    return payload


def scope_record(source: dict[str, Any]) -> Any:
    """Return optional declared scope without treating provenance as admission."""
    provenance = source.get("provenance", {})
    if "source_fit_scope" in provenance and provenance["source_fit_scope"] is None:
        raise ValueError("Declared source scope cannot be null")
    return provenance.get("source_fit_scope")


def _scope_contact_domain(
    library: Any, source: dict[str, Any], config: Any, bound: Any, domain: Any
) -> None:
    from src.shared.python.motion_matching.historical_fit.contact_schedule import (
        ScheduledConstraintOptions,
    )
    from .necromatcher_contacts import contact_schedule_binding
    from .necromatcher_review import CaptureReview

    if config is None or not isinstance(
        config.constraint_options, ScheduledConstraintOptions
    ):
        return
    phases = config.constraint_options.schedule.phases
    if (Fraction(*phases[0].start_pts), Fraction(*phases[-1].end_pts)) != (
        domain.first_pts,
        domain.last_pts,
    ):
        raise ValueError(
            "Scoped contact schedule must cover exactly the selected source domain"
        )
    with CaptureReview(library, source["capture_id"]) as review:
        receipt = contact_schedule_binding(config, source, review)
    if receipt is None or any(
        i < domain.frame_indices[0] or i > domain.frame_indices[-1]
        for phase in receipt["phases"]
        for i in phase["boundary_frame_indices"] + phase["review_frame_indices"]
    ):
        raise ValueError("Contact schedule reviews lie outside selected source domain")


def _registered_scope(library: Any, scope: Any) -> None:
    """Stored and queued scope reviews must be portable, hash-bound library assets."""
    if library is None or scope is None:
        return  # Pure DTO/domain tests; production callers always supply a library.
    try:
        registered = library.load_source_scope_review(scope.review.artifact.artifact_id)
    except KeyError as exc:
        raise ValueError("Registered source scope review required") from exc
    if registered.to_record() != scope.to_record():
        raise ValueError("Source scope review differs from registered portable receipt")


def admit_refit_scope(
    library: Any,
    source: dict[str, Any],
    indices: tuple[int, ...],
    config: Any = None,
    shaft: Any = None,
    requested: Any = None,
) -> Any:
    """Authenticate inherited/explicit scope before scheduling or native compilation.

    Postcondition: selected body, shaft and contact rows lie inside reviewed scope
    and the actual selected domain. Absence retains untouched legacy behavior.
    """
    from .necromatcher_source_scope import (
        SourceFitScope,
        resolve_source_fit_scope,
        validate_scope_selection,
    )

    parent = scope_record(source)
    if parent is None and requested is None:
        return None
    parent_scope = SourceFitScope.from_record(parent) if parent is not None else None
    if requested is not None and not isinstance(requested, SourceFitScope):
        requested = SourceFitScope.from_record(requested)
    _registered_scope(library, parent_scope)
    _registered_scope(library, requested)
    identity = capture_identity(library, source["capture_id"])
    if identity.capture_hash != source["capture_hash"]:
        raise ValueError("Source scope capture hash differs from canonical fit")
    bound = resolve_source_fit_scope(
        identity, parent_scope, requested, artifact_root=getattr(library, "root", None)
    )
    if bound is None:
        raise ValueError("Source scope resolution unexpectedly absent")
    validate_scope_selection(bound, indices, shaft)
    domain = bound.selected_domain(indices)
    if shaft is not None and any(
        f.frame_index < indices[0] or f.frame_index > indices[-1] for f in shaft.frames
    ):
        raise ValueError("Shaft rows lie outside selected source domain")
    _scope_contact_domain(library, source, config, bound, domain)
    return bound


def scope_binding_record(bound: Any, indices: tuple[int, ...]) -> dict[str, Any]:
    """Keep exact selected support distinct from the wider reviewed window."""
    domain = bound.selected_domain(indices)
    return {
        "frame_indices": list(indices),
        "first_pts": [domain.first_pts.numerator, domain.first_pts.denominator],
        "last_pts": [domain.last_pts.numerator, domain.last_pts.denominator],
        "source_clock_sha256": bound.scope.source_clock_sha256,
    }


def validate_scope_payload(
    library: Any, payload: dict[str, Any], parent: dict[str, Any] | None = None
) -> None:
    """Rebind persisted scope, exact source samples and verified parent lineage."""
    from src.shared.python.motion_matching.historical_fit.contracts import (
        ImageFitConfig,
    )

    declared = scope_record(payload)
    if declared is None and (parent is None or scope_record(parent) is None):
        return
    if declared is None:
        raise ValueError("Descendant cannot erase parent source scope")
    indices = tuple(payload["frame_indices"])
    original = payload.get("evidence", {}).get("original_fit", {})
    training = tuple(original.get("frame_indices", indices))
    if any(i not in indices for i in training) or indices != tuple(
        range(indices[0], indices[-1] + 1)
    ):
        raise ValueError("Scoped dense/training frame domain is inconsistent")
    source = parent or payload
    config_record = (
        payload.get("provenance", {}).get("request_options", {}).get("config")
    )
    original_config = original.get("config")
    if not isinstance(config_record, dict) or not isinstance(original_config, dict):
        raise ValueError("Scoped config snapshots are required")
    required = {field.name for field in fields(ImageFitConfig)}
    if set(config_record) != required or set(original_config) != required:
        raise ValueError("Scoped config snapshots must be complete")
    if json.dumps(config_record, sort_keys=True, allow_nan=False) != json.dumps(
        original_config, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Scoped config snapshots differ")
    config = ImageFitConfig.from_record(config_record)
    shaft_record = payload.get("evidence", {}).get("shaft_axis", {}).get("recipe")
    shaft = None
    if shaft_record is not None:
        from .necromatcher_fit_jobs import _parse_shaft_recipe

        shaft = _parse_shaft_recipe(shaft_record)[0].evidence
    bound = admit_refit_scope(library, source, training, config, shaft, declared)
    expected_times = [
        float(bound.identity.frames[i].presentation_time) for i in training
    ]
    if original.get("source_times") != expected_times:
        raise ValueError("Scoped training source clock differs from exact frames")
    domain = scope_binding_record(bound, training)
    if indices[0] != training[0] or indices[-1] != training[-1]:
        raise ValueError("Scoped dense output must match selected training support")
    receipt = payload["provenance"].get("source_fit_scope_binding")
    validate_scope_binding_record(receipt, domain)
    for i, frame in zip(indices, payload["frames"], strict=True):
        if frame != bound.identity.frames[i].to_dict():
            raise ValueError("Scoped output source frame identity mismatch")


def capture_identity(library: Any, capture_id: str) -> Any:
    """Load the public capture owner lazily to retain SDK-free storage imports."""
    from .necromatcher_capture_identity import capture_identity as load_identity

    return load_identity(library, capture_id)


def _scope_parent(library: Any, payload: dict[str, Any]) -> dict[str, Any] | None:
    provenance = payload["provenance"]
    parent_id = provenance.get("warm_start_fit_id")
    if parent_id is None:
        return None
    try:
        asset = library.load_asset(parent_id)
    except KeyError:
        if scope_record(payload) is not None:
            raise ValueError("Scoped parent lineage missing") from None
        return None  # Historical unscoped records did not authenticate this field.
    raw_parent = json.loads(Path(asset.path).read_text(encoding="utf-8"))
    if scope_record(payload) is None and scope_record(raw_parent) is None:
        return None
    parent = library.load_fit(parent_id)
    if scope_record(payload) is not None or scope_record(parent) is not None:
        if provenance.get("warm_start_fit_hash") != asset.metadata["hash"]:
            raise ValueError("Scoped parent lineage hash differs")
    return parent


def validate_scope_binding_record(record: Any, expected: dict[str, Any]) -> None:
    """Strict JSON identity excludes bool/int and float/int receipt coercion."""
    if not isinstance(record, dict) or json.dumps(
        record, sort_keys=True, allow_nan=False
    ) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Source scope binding differs from selected domain")
