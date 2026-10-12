"""Restriction seeds retain canonical coefficients without running an optimizer."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching import historical_fit as fitting
from src.shared.python.workspace import necromatcher_fit_worker as worker

pytestmark = pytest.mark.unit


def start() -> fitting.ImageSplineStart:
    return fitting.ImageSplineStart.from_coefficients(
        np.array([0.0, 0.4, 1.0]),
        np.array([0.1, 0.3, 0.2, 0.1, -0.2, 0.05]),
        ("free", "locked"),
        ("free",),
        "model",
    )


@pytest.mark.parametrize("additional", [None, object()])
def test_restricted_operation_passes_exact_start_to_initializer_only(
    monkeypatch: Any, additional: Any
) -> None:
    saved = start()
    calls = []
    result = SimpleNamespace(optimizer_ran=False, converged=False, initialization=None)
    monkeypatch.setattr(
        fitting,
        "initialize_image_trajectory",
        lambda *args: calls.append(args) or result,
    )
    monkeypatch.setattr(
        worker, "fit_image_trajectory", lambda *args: pytest.fail("Optimizer invoked")
    )
    config = fitting.ImageFitConfig()
    actual = worker._compute_operation(
        "restrict_initialization",
        "native",
        {},
        "camera",
        "inputs",
        config,
        saved,
        additional,
    )
    assert actual is result
    expected = ("native", {}, "camera", "inputs", config, saved)
    assert calls == [expected if additional is None else (*expected, additional)]


@pytest.mark.parametrize("change", ["missing", "policy", "optimizer", "authored"])
def test_restricted_operation_rejects_contradictory_execution(
    monkeypatch: Any, change: str
) -> None:
    result = SimpleNamespace(
        optimizer_ran=change == "optimizer",
        converged=False,
        initialization=object() if change == "authored" else None,
    )
    monkeypatch.setattr(fitting, "initialize_image_trajectory", lambda *args: result)
    config = fitting.ImageFitConfig()
    if change == "policy":
        config = replace(
            config,
            coordinate_bounds=(("free", -1.0, 1.0),),
            initialization_policy="authored_range_project_zero_slopes",
        )
    with pytest.raises(ValueError):
        worker._compute_operation(
            "restrict_initialization",
            "native",
            {},
            "camera",
            "inputs",
            config,
            None if change == "missing" else start(),
        )


def test_restricted_inputs_use_selected_prior_and_nonuniform_knots() -> None:
    saved = start()
    binding = SimpleNamespace(
        plant=SimpleNamespace(plant_sha="model", coordinate_order=("free", "locked")),
        fit={"evidence": {"original_fit": {"free_coordinates": ["free"]}}},
    )
    options = {
        "initialization_source": "restricted_spline",
        "operation": "restrict_initialization",
        "config": asdict(fitting.ImageFitConfig()),
        "knot_count": 3,
        "coordinate_scales": [1.0, 1.0],
    }
    evidence = SimpleNamespace(
        source_times=np.array([0.0, 0.5, 1.0]),
        observed_pixels=np.zeros((3, 1, 2)),
        confidence=np.ones((3, 1)),
    )
    samples = np.array([[0.1, 0.37], [0.2, 0.37], [0.3, 0.37]])
    inputs, actual = worker._worker_inputs(binding, options, evidence, samples, saved)
    assert actual is saved and inputs.initial_samples is None
    np.testing.assert_array_equal(inputs.seed, samples[0])
    np.testing.assert_array_equal(inputs.knot_times, saved.knot_times)


@pytest.mark.parametrize("field", ["model", "order", "free", "count", "clock"])
def test_restricted_input_identity_rejected(field: str) -> None:
    saved = start()
    binding = SimpleNamespace(
        plant=SimpleNamespace(
            plant_sha="other" if field == "model" else "model",
            coordinate_order=("locked", "free")
            if field == "order"
            else ("free", "locked"),
        ),
        fit={
            "evidence": {
                "original_fit": {
                    "free_coordinates": ["locked"] if field == "free" else ["free"]
                }
            }
        },
    )
    options = {
        "initialization_source": "restricted_spline",
        "operation": "restrict_initialization",
        "config": asdict(fitting.ImageFitConfig()),
        "knot_count": 4 if field == "count" else 3,
        "coordinate_scales": [1.0, 1.0],
    }
    evidence = SimpleNamespace(
        source_times=np.array([0.1 if field == "clock" else 0.0, 1.0])
    )
    with pytest.raises(ValueError):
        worker._worker_inputs(binding, options, evidence, np.zeros((2, 2)), saved)


def test_ordinary_preserved_restart_still_rejects_different_interval(
    monkeypatch: Any,
) -> None:
    saved = start()
    binding = SimpleNamespace(
        plant=SimpleNamespace(plant_sha="model", coordinate_order=("free", "locked")),
        fit={"evidence": {"original_fit": {"free_coordinates": ["free"]}}},
    )
    monkeypatch.setattr(worker, "preserved_fit_spline", lambda _: saved)
    with pytest.raises(ValueError, match="source clock interval"):
        worker._preserved_start(binding, {"knot_count": 3}, np.array([0.0, 0.8]))
