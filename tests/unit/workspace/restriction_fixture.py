"""Prospective SDK-free public-owner fixture; not an execution release."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from hypothesis_fixture import imported_capture
from test_scope_fixtures import fixture_review_artifact
from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ImageFitResult,
    ImageSplineStart,
    restrict_image_spline_interval,
)
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity
from src.shared.python.workspace import necromatcher_fit as fit


def restricted_case(fit_case: Any, tmp_path: Path) -> dict[str, Any]:
    library, source, payload = fit_case
    capture = imported_capture(library, tmp_path / "frames")
    identity = capture_identity(library, capture.dataset_id)
    review = fixture_review_artifact(tmp_path, identity, 0, 2)
    library.add_source_scope_review("restriction-review", "practice", Path(review.path))
    scope = library.load_source_scope_review("restriction-review")
    model_path = tmp_path / "model-two-coordinates.xml"
    model_path.write_text("<mujoco/>", encoding="utf-8")
    model = library.add_model(
        "restriction-model",
        "practice",
        model_path,
        engine="mujoco",
        dofs=("hip", "locked"),
    )
    definition = {"coordinate_order": ["hip", "locked"]}
    model_sha = hashlib.sha256(
        json.dumps(definition, allow_nan=False).encode()
    ).hexdigest()
    trajectory = CubicHermiteSplineTrajectory(np.array([0.0, 0.2]), 1)
    start = ImageSplineStart.from_coefficients(
        trajectory.knot_times,
        trajectory.pack(np.array([[0.1], [0.2]]), np.zeros((2, 1))),
        ("hip", "locked"),
        ("hip",),
        model_sha,
    )
    times = np.array([float(f.presentation_time) for f in identity.frames])
    full_q = np.column_stack(
        (
            trajectory.evaluate(np.array(start.spline_coefficients), times).q[:, 0],
            np.full(3, 0.37),
        )
    )
    config = ImageFitConfig(
        coordinate_bounds=(("hip", -1.0, 1.0), ("locked", -1.0, 1.0))
    )
    payload = deepcopy(payload)
    payload.update(
        model_id=model.dataset_id,
        model_hash=model.metadata["hash"],
        capture_id=capture.dataset_id,
        capture_hash=capture.metadata["hash"],
        coordinate_order=["hip", "locked"],
        coordinate_units=["rad", "rad"],
        frame_indices=[0, 1, 2],
        frames=[f.to_dict() for f in identity.frames],
        q=full_q.tolist(),
    )
    payload["provenance"]["native_definition"] = definition
    payload["evidence"]["original_fit"] = {
        **start.to_record(),
        "spline_start": start.to_record(),
        "camera": {},
        "attachments": {},
        "config": asdict(config),
        "frame_indices": [0, 1, 2],
        "source_times": times.tolist(),
        "q": full_q.tolist(),
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    library.add_fit("restriction-parent", "practice", source)
    parent = library.load_fit("restriction-parent")
    bound = fit.admit_refit_scope(library, parent, (0, 1), config, requested=scope)
    receipt = restrict_image_spline_interval(start, times[0], times[1])
    return _case_outputs(library, parent, bound, receipt, times, scope)


def _case_outputs(
    library: Any,
    parent: dict[str, Any],
    bound: Any,
    receipt: Any,
    times: np.ndarray,
    scope: Any,
) -> dict[str, Any]:
    new = receipt.restricted_start
    selected = times[:2]
    actual_q = np.column_stack(
        (
            CubicHermiteSplineTrajectory(np.array(new.knot_times), 1)
            .evaluate(np.array(new.spline_coefficients), selected)
            .q[:, 0],
            np.full(2, 0.37),
        )
    )
    result = ImageFitResult(
        selected,
        actual_q,
        2.0,
        2.0,
        np.zeros((2, 1)),
        2,
        new.model_sha,
        new.coordinate_order,
        False,
        "Synthetic unoptimized restriction",
        np.array(new.knot_times),
        np.array(new.spline_coefficients),
        new.free_coordinates,
        optimizer_ran=False,
        initial_spline=new,
    )
    options = {
        "frame_indices": [0, 1],
        "knot_count": 2,
        "coordinate_scales": [1.0, 1.0],
        "config": parent["evidence"]["original_fit"]["config"],
        "unknown_visibility_weight": 0.5,
        "budget_wall_s": 300.0,
        "operation": "restrict_initialization",
        "initialization_source": "restricted_spline",
    }
    request = {
        "source_fit_id": "restriction-parent",
        "source_fit_hash": library.load_asset("restriction-parent").metadata["hash"],
        "new_fit_id": "restriction-seed",
        "execution_stamp": {},
        "options": options,
        "source_scope": scope.to_record(),
        "source_scope_binding": fit.scope_binding_record(bound, (0, 1)),
        "spline_interval_restriction": receipt.to_record(),
        "spline_restriction_prior": {
            "policy": "selected_first_parent_pose",
            "frame_index": 0,
        },
    }
    return {
        "library": library,
        "source": parent,
        "scope": scope,
        "start": receipt.original_start,
        "receipt": receipt,
        "result": result,
        "request": request,
        "dense": ((0, 1), [f.to_dict() for f in bound.identity.frames[:2]], actual_q),
        "stamp": dict.fromkeys(
            ("started_at_utc", "source_sha256", "runtime_sha256"), "synthetic"
        ),
    }
