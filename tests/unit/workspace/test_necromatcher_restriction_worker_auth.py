"""Restriction admission and final authentication use bounded real contexts."""

from pathlib import Path
from typing import Any

import pytest

from tests.unit.workspace.test_necromatcher_authenticated_worker import (
    mutate_same_stat,
)
from tests.unit.workspace.test_necromatcher_restriction_worker import start
from tests.unit.workspace.test_necromatcher_source_scope import make_scope
from src.shared.python.motion_matching.historical_fit import (
    restrict_image_spline_interval,
)
from src.shared.python.workspace import necromatcher_fit_worker as worker

pytestmark = pytest.mark.unit
pytest_plugins = ["tests.unit.workspace.test_necromatcher_authenticated_worker"]


@pytest.fixture
def restriction_case(case: Any, tmp_path: Path, monkeypatch: Any) -> Any:
    from src.shared.python.workspace import necromatcher_spline_restriction as owner

    library, request, state = case
    identity, scope = make_scope(tmp_path / "receipt")
    frames = [identity.frames[i].to_dict() for i in (0, 2)]
    first, last = (identity.frames[i].presentation_time for i in (0, 2))
    receipt = restrict_image_spline_interval(start(), 0.0, 0.8)
    request.update(
        source_scope=scope.to_record(),
        source_scope_binding={
            "frame_indices": [0, 2],
            "first_pts": [first.numerator, first.denominator],
            "last_pts": [last.numerator, last.denominator],
            "source_clock_sha256": scope.source_clock_sha256,
        },
        spline_interval_restriction=receipt.to_record(),
        spline_restriction_prior={
            "policy": "selected_first_parent_pose",
            "frame_index": 0,
        },
    )
    request["options"].update(
        operation="restrict_initialization", initialization_source="restricted_spline"
    )
    state.derivations = 0
    state.bindings = 0

    def derive(*args: Any) -> Any:
        state.derivations += 1
        return receipt

    original = worker._load_refit_binding

    def binding(*args: Any) -> Any:
        state.bindings += 1
        return original(*args)

    monkeypatch.setattr(owner, "derive_spline_restriction", derive)
    monkeypatch.setattr(owner, "_pose_samples", lambda *args: (None, None, frames))
    monkeypatch.setattr(library, "load_fit", lambda *args: {})
    monkeypatch.setattr(worker, "_load_refit_binding", binding)
    return library, request, state


@pytest.mark.parametrize(
    "mutation", ["receipt", "prior", "bool_prior", "mode", "missing_scope"]
)
def test_queued_restriction_fault_rejects_before_compiled_binding(
    restriction_case: Any, mutation: str
) -> None:
    library, request, state = restriction_case
    if mutation == "receipt":
        request["spline_interval_restriction"]["restricted_start"]["model_sha"] = (
            "other"
        )
    elif mutation == "prior":
        request["spline_restriction_prior"]["frame_index"] = 2
    elif mutation == "bool_prior":
        request["spline_restriction_prior"]["frame_index"] = False
    elif mutation == "mode":
        request["options"]["initialization_source"] = "sampled_parent"
    else:
        request.pop("source_scope_binding")
    with pytest.raises(ValueError):
        worker.compute_native_refit(request)
    assert state.bindings == state.computations == state.reviews == 0
    with library.authenticated_read():
        pass


def test_restriction_rederived_in_fresh_final_context(restriction_case: Any) -> None:
    library, request, state = restriction_case
    worker.compute_native_refit(request)
    assert state.derivations == 2 and state.computations == 1 and state.reviews == 0
    with library.authenticated_read():
        pass


def test_restriction_setup_mutation_prevents_initializer(restriction_case: Any) -> None:
    library, request, state = restriction_case
    state.mutate = 1
    with pytest.raises(ValueError, match="hash|changed"):
        worker.compute_native_refit(request)
    assert state.computations == state.reviews == 0
    with library.authenticated_read():
        pass


def test_restriction_compute_mutation_prevents_response(
    restriction_case: Any, monkeypatch: Any
) -> None:
    library, request, state = restriction_case
    original = worker._compute_operation

    def compute(*args: Any) -> Any:
        result = original(*args)
        mutate_same_stat(Path(library.load_asset(request["source_fit_id"]).path))
        return result

    monkeypatch.setattr(worker, "_compute_operation", compute)
    with pytest.raises(ValueError, match="hash|changed"):
        worker.compute_native_refit(request)
    assert state.computations == 1 and state.reviews == 0
    with library.authenticated_read():
        pass
