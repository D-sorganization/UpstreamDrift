"""Proxy validation boundaries, distinct from native mechanical equivalence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from src.engines.physics_engines.opensim.python import (
    native_abdominal_reduction as reduction,
)


def _model(names: tuple[str, ...]) -> MagicMock:
    model = MagicMock()
    model.state_names = names
    state = MagicMock()
    state.getNU.return_value = 1
    model.initSystem.return_value = state
    model.getStateVariableValue.return_value = 0.0
    muscle = MagicMock()
    muscle.getName.return_value = "observed_muscle"
    muscle.getLengtheningSpeed.return_value = -2.0
    muscle.getActuation.return_value = 3.0
    muscles = model.getMuscles.return_value
    muscles.__iter__.side_effect = lambda: iter((muscle,))
    muscles.get.return_value = muscle
    return model


@pytest.fixture
def boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[MagicMock, MagicMock, reduction.AbdominalReductionRequest]:
    retained = "/jointset/Abdjnt/Abs_r3/value"
    removed = tuple(
        f"/jointset/Abdjnt/{name}/{kind}"
        for name in ("Abs_t1", "Abs_t2")
        for kind in ("value", "speed")
    )
    source, derived = _model((retained, *removed)), _model((retained,))
    monkeypatch.setattr(reduction, "_verify_inventory", lambda *_: ("Abs_r3",))
    monkeypatch.setattr(reduction, "_names", lambda model: model.state_names)
    monkeypatch.setattr(reduction, "_native_admission", lambda *_: None)
    monkeypatch.setattr(reduction, "_pose_error", lambda *_: (*([0.0] * 8), 1, 1.0))
    request = reduction.AbdominalReductionRequest(
        tmp_path / "source.osim",
        "0" * 64,
        tmp_path / "derived.osim",
        (tuple((name, 0.0) for name in source.state_names),),
    )
    return source, derived, request


@pytest.mark.parametrize("side", [0, 1], ids=["source", "derived"])
@pytest.mark.parametrize("accessor", ["getLengtheningSpeed", "getActuation"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_native_scalar_cannot_disappear_in_maximum(
    boundary: tuple[Any, Any, reduction.AbdominalReductionRequest],
    side: int,
    accessor: str,
    value: float,
) -> None:
    muscle = boundary[side].getMuscles().get("observed_muscle")
    getattr(muscle, accessor).return_value = value
    with pytest.raises(ValueError, match="nonfinite"):
        reduction._measure_states(*boundary)


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_comparison_helper_observation_rejects(
    boundary: tuple[Any, Any, reduction.AbdominalReductionRequest],
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    value: float,
) -> None:
    observations = [*([0.0] * 8), 1, 1.0]
    observations[index] = value
    monkeypatch.setattr(reduction, "_pose_error", lambda *_: tuple(observations))
    with pytest.raises(ValueError, match="nonfinite or negative"):
        reduction._measure_states(*boundary)


def test_equal_finite_nonzero_signed_observations_are_retained(
    boundary: tuple[Any, Any, reduction.AbdominalReductionRequest],
) -> None:
    errors, rank, condition = reduction._measure_states(*boundary)
    assert errors == (0.0,) * 10
    assert rank == 1
    assert condition == 1.0


@pytest.mark.parametrize("accessor", ["getLengtheningSpeed", "getActuation"])
def test_finite_native_scalar_difference_remains_a_failure(
    boundary: tuple[Any, Any, reduction.AbdominalReductionRequest], accessor: str
) -> None:
    muscle = boundary[1].getMuscles().get("observed_muscle")
    getattr(muscle, accessor).return_value = 10.0
    with pytest.raises(ValueError, match="sampled mechanics differ"):
        reduction._measure_states(*boundary)


@pytest.mark.integration
def test_native_output_mutation_after_measurement_cannot_receive_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("opensim")
    source_text = os.environ.get("BUET_REDUCTION_SOURCE")
    prepared_text = os.environ.get("BUET_REDUCTION_PREPARED")
    if not source_text or not prepared_text:
        pytest.skip("owned BUET source and prepared receipt opt-in absent")
    source = Path(source_text)
    prepared = json.loads(Path(prepared_text).read_text(encoding="utf-8"))["prepared"][
        "named_state"
    ]
    output = tmp_path / "derived.osim"
    request = reduction.AbdominalReductionRequest(
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        output,
        (tuple(prepared.items()),),
    )
    measure = reduction._measure_states

    def mutate_after_measurement(*args: Any) -> Any:
        result = measure(*args)
        with output.open("ab") as stream:
            stream.write(b"\n<!-- changed after native measurement -->\n")
        return result

    monkeypatch.setattr(reduction, "_measure_states", mutate_after_measurement)
    with pytest.raises(ValueError, match="changed during verification"):
        reduction.derive_abdominal_zero_translations(request)
    assert output.read_bytes().endswith(b"<!-- changed after native measurement -->\n")
