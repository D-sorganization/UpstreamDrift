"""Authored starts are separate research versions, not optimizer successes."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_author_operation_uses_public_initializer_and_never_optimizer(monkeypatch):
    from src.shared.python.motion_matching import historical_fit
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    result = SimpleNamespace(
        optimizer_ran=False, converged=False, initialization=_Initialization()
    )
    calls = []

    def initialize(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(
        historical_fit, "initialize_image_trajectory", initialize, raising=False
    )
    monkeypatch.setattr(
        worker, "fit_image_trajectory", lambda *args: pytest.fail("optimizer called")
    )
    config = SimpleNamespace(initialization_policy="authored_range_project_zero_slopes")
    assert (
        worker._compute_operation(
            "author_initialization", "native", {}, "camera", "inputs", config
        )
        is result
    )
    assert len(calls) == 1 and calls[0][-1] is config


def test_author_operation_rejects_contradictory_optimizer_receipt(monkeypatch):
    from src.shared.python.motion_matching import historical_fit
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    monkeypatch.setattr(
        historical_fit,
        "initialize_image_trajectory",
        lambda *args: SimpleNamespace(
            optimizer_ran=True, converged=True, initialization=None
        ),
        raising=False,
    )
    config = SimpleNamespace(initialization_policy="authored_range_project_zero_slopes")
    with pytest.raises(ValueError, match="receipt"):
        worker._compute_operation("author_initialization", None, {}, None, None, config)


@pytest.mark.parametrize(
    "operation,policy",
    [("unknown", "strict"), (True, "strict"), ("author_initialization", "strict")],
)
def test_invalid_worker_operation_or_policy_fails(operation, policy):
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    with pytest.raises(ValueError, match="operation|policy"):
        worker._compute_operation(
            operation,
            None,
            {},
            None,
            None,
            SimpleNamespace(initialization_policy=policy),
        )


def test_authored_ranges_reject_custom_limits_without_mislabeling():
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    authored = SimpleNamespace(
        named_bounds={"joint": (-1.0, 1.0)}, to_record=lambda: {"range_source": "exact"}
    )
    binding = SimpleNamespace(authored_coordinate_bounds=lambda: authored)
    config = SimpleNamespace(coordinate_bounds=(("joint", -2.0, 2.0),))
    with pytest.raises(ValueError, match="authored"):
        worker._range_provenance(binding, config, "author_initialization")
    assert worker._range_provenance(binding, config, "fit") == {
        "range_source": "custom_coordinate_bounds"
    }
    config.coordinate_bounds = (("joint", -1.0, 1.0),)
    assert worker._range_provenance(binding, config, "author_initialization") == {
        "range_source": "exact"
    }


@dataclass(frozen=True)
class _Initialization:
    policy: str = "authored_range_project_zero_slopes"
    original_coefficient_sha256: str = "sha256:old"
    initialized_coefficient_sha256: str = "sha256:new"


def test_saved_initialization_receipt_is_not_optimizer_convergence():
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    result = SimpleNamespace(
        source_times=np.array([1.0, 2.0]),
        q=np.zeros((2, 1)),
        free_coordinates=("joint",),
        coordinate_order=("joint",),
        knot_times=np.array([1.0, 2.0]),
        spline_coefficients=np.zeros(4),
        initial_rms_pixels=7.0,
        rms_pixels=7.0,
        converged=False,
        optimizer_message="Authored initialization only",
        optimizer_ran=False,
        initialization=_Initialization(),
        constraint_times=np.array([]),
        constraint_row_labels=(),
        constraint_residuals=np.empty((0, 0)),
        maximum_constraint_residual=0.0,
    )
    original = {"camera": {}, "attachments": {"marker": ["body", [0.0, 0.0, 0.0]]}}
    source = {
        "provenance": {"native_definition": {}},
        "evidence": {"original_fit": original},
        "q": [[9.0], [9.0]],
    }
    config = {
        "coordinate_bounds": [["joint", -1.0, 1.0]],
        "initialization_policy": "authored_range_project_zero_slopes",
    }
    request = {
        "source_fit_id": "old",
        "source_fit_hash": "sha256:old",
        "execution_stamp": {},
        "options": {
            "frame_indices": [0, 1],
            "config": config,
            "operation": "author_initialization",
        },
    }
    stamp = {
        "started_at_utc": "now",
        "source_sha256": "source",
        "runtime_sha256": "runtime",
    }
    payload = worker._build_fit_payload(
        request, source, result, ((0, 1), [{}, {}], result.q), stamp, 0.0
    )
    evidence = payload["evidence"]["original_fit"]
    assert evidence["optimizer_ran"] is False and evidence["converged"] is False
    assert evidence["initialization"]["policy"] == _Initialization().policy
    assert evidence["rms_pixels"] == 7.0 and source["q"] == [[9.0], [9.0]]
    reasons = payload["evidence"]["rejection_reasons"]
    assert "authored_initialization_only" in reasons
    assert "optimizer_not_converged" not in reasons
    assert "anatomical_ranges_not_enforced" in reasons
    assert "historical_anatomy_unqualified" in reasons
    assert evidence["constraint_assessment"]["continuous_certified"] is False


def test_exact_authored_ranges_keep_anatomy_and_unbounded_coordinates_unqualified():
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    result = SimpleNamespace(optimizer_ran=True, converged=True)
    config = SimpleNamespace(coordinate_bounds=(("joint", -1.0, 1.0),))
    ranges = {
        "range_source": "bound_native_definition.coordinate_ranges_deg",
        "named_bounds": {"joint": [-1.0, 1.0]},
        "unbounded_names": ["root"],
    }
    blockers = worker._research_blockers(result, config, ranges)
    assert "historical_anatomy_unqualified" in blockers
    assert "native_coordinates_without_authored_ranges" in blockers
    assert "native_authored_ranges_not_fully_enforced" not in blockers
    custom = worker._research_blockers(
        result, config, {"range_source": "custom_coordinate_bounds"}
    )
    assert "native_authored_ranges_not_fully_enforced" in custom
