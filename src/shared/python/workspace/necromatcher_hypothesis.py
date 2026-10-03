"""Authenticated authored model/camera rebinding; no optimization or acceptance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ImageSplineStart,
)
from .necromatcher import NecromatcherLibrary
from .necromatcher_capture_identity import CaptureIdentity, capture_identity
from .necromatcher_hypothesis_contracts import NativeHypothesisRequest
from .necromatcher_spline import preserved_fit_spline, verify_preserved_fit_samples
from .project_store import DatasetMetadata, validate_workspace_id
from src.shared.python.shadow_tracker.source_records import FrameIdentity


@dataclass(frozen=True)
class BoundNativeHypothesis:
    """Library-authenticated input snapshot; native checks remain worker-owned."""

    request: NativeHypothesisRequest
    parent_bytes: bytes
    capture: CaptureIdentity
    parent_start: ImageSplineStart
    rebound_start: ImageSplineStart

    def parent_record(self) -> dict[str, Any]:
        return dict(json.loads(self.parent_bytes))

    def candidate_record(self) -> dict[str, Any]:
        """Adapt only verified authored identities for the canonical fit builder."""
        source = self.parent_record()
        source["model_id"] = self.request.model.model_id
        source["model_hash"] = self.request.model.model_hash
        source["provenance"]["native_definition"] = json.loads(
            self.request.model.definition_bytes
        )
        original = source["evidence"]["original_fit"]
        original["camera"] = self.request.to_record()["camera"]
        original["attachments"] = self.request.model.to_record()["attachments"]
        return source


def hypothesis_seed_options(bound: BoundNativeHypothesis) -> dict[str, Any]:
    """Preserve the authenticated parent recipe with explicit strict rebinding."""
    source = bound.parent_record()
    previous = source["provenance"].get("request_options")
    original = source["evidence"]["original_fit"]
    if not isinstance(previous, dict) or "coordinate_scales" not in previous:
        raise ValueError(
            "Hypothesis admission requires explicitly recorded native coordinate scales"
        )
    config = ImageFitConfig.from_record(
        original.get("config", previous.get("config", {}))
    )
    record = {
        **previous,
        "frame_indices": list(original["frame_indices"]),
        "knot_count": len(bound.rebound_start.knot_times),
        "config": asdict(replace(config, initialization_policy="strict")),
        "operation": "hypothesis_rebinding",
        "initialization_source": "explicit_model_camera_rebinding",
        "unknown_visibility_weight": previous.get("unknown_visibility_weight", 0.5),
    }
    return dict(json.loads(json.dumps(record, allow_nan=False)))


def _validate_seed_recipe(fit: dict[str, Any], bound: BoundNativeHypothesis) -> None:
    options = hypothesis_seed_options(bound)
    original = fit["evidence"]["original_fit"]
    actual = json.loads(
        json.dumps(
            {
                "options": fit["provenance"].get("request_options"),
                "config": original.get("config"),
                "frame_indices": original.get("frame_indices"),
            },
            allow_nan=False,
        )
    )
    if actual != {
        "options": options,
        "config": options["config"],
        "frame_indices": options["frame_indices"],
    }:
        raise ValueError("Hypothesis seed differs from the authenticated parent recipe")


def _mapped_parent_samples(
    source: dict[str, Any], start: ImageSplineStart
) -> np.ndarray:
    times = np.array(
        [float(frame.presentation_time) for frame in _source_frames(source)]
    )
    if times[0] != start.knot_times[0] or times[-1] != start.knot_times[-1]:
        raise ValueError("Hypothesis requires the complete preserved source interval")
    canonical = verify_preserved_fit_samples(source, start)
    saved = np.asarray(source["q"], dtype=float)
    if not np.allclose(saved, canonical, rtol=0.0, atol=1e-12):
        raise ValueError("Hypothesis saved poses exceed canonical roundoff tolerance")
    return saved


def _source_frames(source: dict[str, Any]) -> tuple[FrameIdentity, ...]:
    return tuple(FrameIdentity.from_dict(frame) for frame in source["frames"])


def bind_native_hypothesis(
    library: NecromatcherLibrary, source_fit_id: str, request: NativeHypothesisRequest
) -> BoundNativeHypothesis:
    """Authenticate current assets, ownership, source pixels/clock and explicit mapping."""
    if (
        not isinstance(request, NativeHypothesisRequest)
        or request.parents.source_fit_id != source_fit_id
    ):
        raise ValueError("Hypothesis source must match the typed parent request")
    source = library.load_fit(source_fit_id)
    source_asset = library.load_asset(source_fit_id)
    model = library.load_asset(request.model.model_id)
    capture = capture_identity(library, source["capture_id"])
    pins = request.parents
    if (
        pins.source_fit_hash != source_asset.metadata["hash"]
        or pins.capture_id != capture.capture_id
        or pins.capture_hash != capture.capture_hash
        or pins.source_sha256 != capture.source_sha256
        or pins.source_clock_sha256 != capture.source_clock_sha256
    ):
        raise ValueError("Hypothesis parent/source hashes or clock changed")
    if (
        model.kind != "native_model"
        or model.session_id != source_asset.session_id
        or model.metadata["hash"] != request.model.model_hash
    ):
        raise ValueError("Hypothesis model must be hash-bound to the same swing")
    mapping = request.mapping
    if (
        mapping.coordinate_order != tuple(source["coordinate_order"])
        or mapping.coordinate_order != tuple(model.metadata["dofs"])
        or mapping.coordinate_units != tuple(source["coordinate_units"])
    ):
        raise ValueError(
            "Hypothesis coordinate order/units must preserve explicit parent mapping"
        )
    start = preserved_fit_spline(source)
    if start is None or mapping.free_coordinates != start.free_coordinates:
        raise ValueError(
            "Hypothesis requires the exact saved final spline/free mapping"
        )
    if (
        tuple(source["frame_indices"]) != tuple(range(len(capture.frames)))
        or _source_frames(source) != capture.frames
    ):
        raise ValueError("Hypothesis must preserve complete original frame identities")
    samples = _mapped_parent_samples(source, start)
    if not np.array_equal(np.asarray(mapping.reference_pose), samples[0]):
        raise ValueError(
            "Hypothesis reference pose must equal the saved final first pose"
        )
    original = source["evidence"]["original_fit"]
    if tuple(request.model.attachments) != tuple(original["attachments"]):
        raise ValueError("Hypothesis must retain every parent marker identity/order")
    rebound = ImageSplineStart.from_coefficients(
        start.knot_times,
        start.spline_coefficients,
        mapping.coordinate_order,
        mapping.free_coordinates,
        request.model.definition_sha256.removeprefix("sha256:"),
    )
    return BoundNativeHypothesis(
        request,
        json.dumps(source, allow_nan=False).encode("utf-8"),
        capture,
        start,
        rebound,
    )


def author_native_hypothesis(
    library: NecromatcherLibrary,
    source_fit_id: str,
    new_fit_id: str,
    request: NativeHypothesisRequest,
) -> DatasetMetadata:
    """Clean-worker validation followed by exclusive, unoptimized seed persistence."""
    from .necromatcher_fit_jobs import fit_execution_stamp
    from .necromatcher_native_worker import execute_native_research_worker

    validate_workspace_id(new_fit_id, "New hypothesis fit ID")
    bound = bind_native_hypothesis(library, source_fit_id, request)
    if any(
        asset.dataset_id == new_fit_id
        for swing in library.swings()
        for asset in library.assets(swing.session_id)
    ):
        raise ValueError("New hypothesis fit ID already exists")
    stamp = fit_execution_stamp()
    with TemporaryDirectory(prefix="necromatcher-hypothesis-") as directory:
        path = Path(directory) / "request.json"
        payload = {
            "library_root": str(library.root),
            "source_fit_id": source_fit_id,
            "new_fit_id": new_fit_id,
            "hypothesis": request.to_record(),
            "execution_stamp": stamp,
        }
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        result = execute_native_research_worker(
            path, 600.0, lambda: False, operation="hypothesis"
        )
        fit = result["fit"]
        if fit["provenance"].get("execution_stamp") != stamp or any(
            fit["provenance"].get("worker_stamp", {}).get(key) != stamp[key]
            for key in ("source_sha256", "runtime_sha256")
        ):
            raise ValueError(
                "Hypothesis worker source/runtime receipt differs from request"
            )
        validate_hypothesis_seed(library, fit, bound)
        current = fit_execution_stamp()
        if (
            current["source_sha256"] != stamp["source_sha256"]
            or current["runtime_sha256"] != stamp["runtime_sha256"]
        ):
            raise ValueError(
                "Hypothesis implementation/runtime changed during validation"
            )
        bind_native_hypothesis(library, source_fit_id, request)
        path = Path(directory) / "fit.json"
        path.write_text(json.dumps(fit, allow_nan=False), encoding="utf-8")
        return library.add_fit(
            new_fit_id, library.load_asset(source_fit_id).session_id, path
        )


def validate_hypothesis_seed(
    library: NecromatcherLibrary,
    fit: dict[str, Any],
    bound: BoundNativeHypothesis | None = None,
) -> None:
    """Recalled seeds cannot silently change authored lineage or restart identity."""
    lineage = fit.get("provenance", {}).get("native_hypothesis")
    if lineage is None:
        if (
            bound is not None
            or fit.get("provenance", {}).get("operation") == "hypothesis_rebinding"
        ):
            raise ValueError("Hypothesis seed requires its authenticated lineage")
        return
    request = NativeHypothesisRequest.from_record(lineage["request"])
    current = bind_native_hypothesis(library, request.parents.source_fit_id, request)
    if bound is not None and hypothesis_lineage(bound) != hypothesis_lineage(current):
        raise ValueError("Hypothesis worker returned a different authored request")
    bound = current
    _validate_seed_recipe(fit, bound)
    expected = bound.candidate_record()
    original = fit["evidence"]["original_fit"]
    if (
        fit["model_id"] != request.model.model_id
        or fit["model_hash"] != request.model.model_hash
        or fit["capture_id"] != expected["capture_id"]
        or fit["capture_hash"] != expected["capture_hash"]
        or fit["frames"] != expected["frames"]
        or not np.allclose(
            np.asarray(fit["q"], dtype=float),
            np.asarray(expected["q"], dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Hypothesis seed model/source/poses differ from authenticated parents"
        )
    if (
        fit["qualification"] != "monocular_research_hypothesis"
        or fit["physical_time_qualified"] is not False
        or fit["dynamics_replayed"] is not False
        or original["optimizer_ran"] is not False
        or original["converged"] is not False
    ):
        raise ValueError(
            "Hypothesis seed cannot promote scientific or optimizer qualification"
        )
    if (
        preserved_fit_spline(fit) != bound.rebound_start
        or ImageSplineStart.from_record(original["initial_spline"])
        != bound.rebound_start
    ):
        raise ValueError("Hypothesis seed differs from the explicit rebound start")
    if (
        original["camera"] != request.to_record()["camera"]
        or original["attachments"] != request.model.to_record()["attachments"]
    ):
        raise ValueError(
            "Hypothesis seed camera/marker mapping differs from authored recipe"
        )
    if lineage != hypothesis_lineage(bound):
        raise ValueError(
            "Hypothesis lineage differs from authenticated source identities"
        )


def hypothesis_lineage(bound: BoundNativeHypothesis) -> dict[str, Any]:
    """Detached lineage explicitly distinguishes rebinding from same-model restart."""
    return {
        "schema_version": "necromatcher/native-hypothesis-lineage/1",
        "request": bound.request.to_record(),
        "capture_identity": bound.capture.to_record(),
        "parent_start": bound.parent_start.to_record(),
        "rebound_start": bound.rebound_start.to_record(),
        "initialization_source": "explicit_model_camera_rebinding",
        "optimizer_ran": False,
        "physical_time_qualified": False,
        "camera_qualified": False,
        "anatomy_qualified": False,
        "continuous_certified": False,
    }
