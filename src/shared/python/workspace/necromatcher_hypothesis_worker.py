"""Whitelisted clean native initializer for authenticated authored hypotheses."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

if __name__ == "__main__":
    import mujoco  # noqa: F401 -- SDK must precede workspace/native imports

import numpy as np

from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ImageFitInputs,
    initialize_image_trajectory,
    read_capture_evidence,
)
from .necromatcher import NecromatcherLibrary
from .necromatcher_contacts import contact_schedule_binding
from .necromatcher_fit_jobs import fit_execution_stamp
from .necromatcher_fit_metrics import dense_reprojection_metrics
from .necromatcher_fit_records import build_native_fit_payload
from .necromatcher_hypothesis import (
    bind_native_hypothesis,
    hypothesis_lineage,
    hypothesis_seed_options,
)
from .necromatcher_hypothesis_contracts import NativeHypothesisRequest
from .necromatcher_native import NativeModelBinding, load_native_model_binding
from .necromatcher_ranges import extract_authored_bounds
from .necromatcher_review import CaptureReview

logger = logging.getLogger(__name__)


def _check_stamp(expected: dict[str, Any]) -> dict[str, Any]:
    stamp = fit_execution_stamp()
    if any(
        stamp[key] != expected[key]
        for key in ("source_commit", "source_sha256", "runtime_sha256")
    ):
        raise ValueError(
            "Hypothesis source/runtime differs from authenticated execution request"
        )
    return stamp


def _ranges(model: NativeModelBinding, config: ImageFitConfig) -> dict[str, Any]:
    configured = {
        name: (lower, upper) for name, lower, upper in config.coordinate_bounds
    }
    authored = extract_authored_bounds(
        model.definition_bytes,
        model.model_hash,
        tuple(model.plant.coordinate_order),
        model.coordinate_units,
    )
    if configured and configured == dict(authored.named_bounds):
        return authored.to_record()
    return {"range_source": "custom_coordinate_bounds" if configured else "none"}


def compute_native_hypothesis(request: dict[str, Any]) -> dict[str, Any]:
    """Validate one native candidate and evaluate its explicit unoptimized rebound."""
    if not isinstance(request, dict) or set(request) != {
        "library_root",
        "source_fit_id",
        "new_fit_id",
        "hypothesis",
        "execution_stamp",
    }:
        raise ValueError("Unknown native hypothesis worker request")
    stamp = _check_stamp(request["execution_stamp"])
    library = NecromatcherLibrary(request["library_root"])
    recipe = NativeHypothesisRequest.from_record(request["hypothesis"])
    bound = bind_native_hypothesis(library, request["source_fit_id"], recipe)
    model = load_native_model_binding(
        library,
        recipe.model.model_id,
        recipe.model.definition_bytes,
        recipe.mapping.coordinate_units,
    )
    if (
        model.model_hash != recipe.model.model_hash
        or model.plant.plant_sha != bound.rebound_start.model_sha
    ):
        raise ValueError(
            "Hypothesis compiled model differs from rebound start identity"
        )
    source = bound.candidate_record()
    options = hypothesis_seed_options(bound)
    config = ImageFitConfig.from_record(options["config"])
    attachments = dict(recipe.model.attachments)
    seed = np.asarray(recipe.mapping.reference_pose)
    if not np.isfinite(model.plant.marker_positions(seed, attachments)).all():
        raise ValueError("Hypothesis native marker projection must be finite")
    indices = tuple(options["frame_indices"])
    with CaptureReview(library, recipe.parents.capture_id) as review:
        contacts = contact_schedule_binding(config, source, review)
        evidence = read_capture_evidence(
            review,
            tuple(attachments),
            indices,
            unknown_visibility_weight=options["unknown_visibility_weight"],
        )
        inputs = ImageFitInputs(
            evidence.source_times,
            evidence.observed_pixels,
            evidence.confidence,
            seed,
            np.asarray(options["coordinate_scales"]),
            recipe.mapping.free_coordinates,
            np.asarray(bound.rebound_start.knot_times),
        )
        result = initialize_image_trajectory(
            model.plant, attachments, recipe.camera, inputs, config, bound.rebound_start
        )
        dense_indices = tuple(source["frame_indices"])
        dense = read_capture_evidence(
            review,
            tuple(attachments),
            dense_indices,
            unknown_visibility_weight=options["unknown_visibility_weight"],
        )
        q = result.evaluate_source_times(dense.source_times)
        frames = [review.frame(index)["frame"] for index in dense_indices]
    if result.optimizer_ran or result.converged or result.initialization is not None:
        raise ValueError(
            "Hypothesis seed cannot run optimization or project the preserved start"
        )
    closure = np.array([model.plant.closure_residuals(pose) for pose in q])
    ranges = _ranges(model, config)
    build_request = {
        **request,
        "source_fit_hash": recipe.parents.source_fit_hash,
        "options": options,
    }
    output = build_native_fit_payload(
        build_request,
        source,
        result,
        (dense_indices, frames, q),
        stamp,
        float(np.max(np.linalg.norm(closure, axis=1))),
        ranges,
    )
    output["provenance"]["native_hypothesis"] = hypothesis_lineage(bound)
    output["provenance"]["coordinate_bounds_provenance"] = ranges
    if contacts is not None:
        output["provenance"]["contact_schedule_binding"] = contacts
    output["evidence"]["original_fit"].update(
        dense_reprojection_metrics(
            model.plant, recipe.camera, attachments, dense, q, indices
        )
    )
    bind_native_hypothesis(library, request["source_fit_id"], recipe)
    _check_stamp(request["execution_stamp"])
    return output


def main() -> None:
    """Serve the fixed reviewed hypothesis request; never accept module callbacks."""
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        result = compute_native_hypothesis(request)
        sys.stdout.write(json.dumps({"fit": result}, allow_nan=False))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        ImportError,
    ):
        logger.exception("Native hypothesis admission failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
