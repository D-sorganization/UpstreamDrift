"""Authenticated fixture parents are required even before native execution."""

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_bound_request_rebinds_model_without_claiming_strict_parent_identity(
    hypothesis_case,
):
    from src.shared.python.workspace.necromatcher_hypothesis import (
        bind_native_hypothesis,
    )

    library, request, _ = hypothesis_case
    bound = bind_native_hypothesis(library, "parent", request)
    assert (
        bound.rebound_start.model_sha
        == request.model.definition_sha256.removeprefix("sha256:")
    )
    assert bound.parent_start.model_sha != bound.rebound_start.model_sha
    assert (
        bound.parent_start.spline_coefficients
        == bound.rebound_start.spline_coefficients
    )
    assert bound.parent_start.knot_times == bound.rebound_start.knot_times
    import numpy as np
    from src.shared.python.estimation import CubicHermiteSplineTrajectory

    times = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    trajectory = CubicHermiteSplineTrajectory(
        np.asarray(bound.parent_start.knot_times),
        len(bound.parent_start.free_coordinates),
    )
    before = trajectory.evaluate(
        np.asarray(bound.parent_start.spline_coefficients), times
    )
    after = trajectory.evaluate(
        np.asarray(bound.rebound_start.spline_coefficients), times
    )
    for name in ("q", "v", "a"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


@pytest.mark.parametrize(
    "fault",
    [
        "fit_hash",
        "source_hash",
        "clock",
        "foreign_model",
        "locked",
        "missing_marker",
        "different_parent",
    ],
)
def test_authentication_rejects_stale_or_foreign_inputs_without_writes(
    hypothesis_case, fault
):
    from src.shared.python.workspace.necromatcher_hypothesis import (
        bind_native_hypothesis,
    )
    from src.shared.python.workspace.necromatcher_hypothesis_contracts import (
        NativeHypothesisRequest,
    )

    library, request, parent = hypothesis_case
    record = request.to_record()
    if fault == "fit_hash":
        record["parents"]["source_fit_hash"] = "sha256:" + "0" * 64
    elif fault == "source_hash":
        record["parents"]["source_sha256"] = "sha256:" + "0" * 64
    elif fault == "clock":
        record["parents"]["source_clock_sha256"] = "sha256:" + "0" * 64
    elif fault == "locked":
        record["mapping"]["reference_pose"][0] = 1
    elif fault == "missing_marker":
        record["model"]["attachments"] = {"other": ["world", [0, 0, 0]]}
    elif fault == "foreign_model":
        library.add_swing("other", "hogan", "Other")
        asset = library.load_asset("candidate")
        foreign = library.add_model(
            "foreign",
            "other",
            Path(asset.path),
            engine="mujoco",
            dofs=tuple(parent["coordinate_order"]),
        )
        record["model"]["model_id"] = foreign.dataset_id
    else:
        record["parents"]["source_fit_id"] = "other-parent"
    before = {
        path: path.read_bytes() for path in library.root.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError):
        bind_native_hypothesis(
            library, "parent", NativeHypothesisRequest.from_record(record)
        )
    assert {
        path: path.read_bytes() for path in library.root.rglob("*") if path.is_file()
    } == before
