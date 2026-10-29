"""Receipt contract for the unified cross-engine perturbation study (#13795)."""

from __future__ import annotations

import pytest

from src.shared.python.analysis.cross_engine import BACKEND_REAL, BACKEND_STUB
from src.shared.python.analysis.cross_engine_receipt import (
    SETUP_FIELDS,
    ReceiptContractError,
    SharedSetup,
    build_receipt,
    validate_receipt,
)
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEngineRunResult,
)
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
)

pytestmark = pytest.mark.unit

SETUP_ARGS = {
    "dt_s": 0.001,
    "integrator": "rk4",
    "seed": 42,
    "interpolation_mode": "zoh",
}


def _setup() -> SharedSetup:
    return SharedSetup.from_spec(TRIPLE_PENDULUM_SPEC, **SETUP_ARGS)


def _result(name: str, energy: float) -> CrossEngineRunResult:
    return CrossEngineRunResult(
        engine_name=name,
        mean_total_energy_final=energy,
        mean_end_effector_speed_final=5.0,
        mean_peak_end_effector_speed=6.0,
    )


def _build(backends: dict[str, str], energies: dict[str, float]) -> dict:
    results = {n: _result(n, e) for n, e in energies.items()}
    return build_receipt(_setup(), backends, results, n_trials=10, t_end_s=1.5)


def _measured(energies: tuple[float, float] = (10.0, 12.0)) -> dict:
    backends = {"mujoco": BACKEND_REAL, "drake": BACKEND_REAL}
    return _build(backends, dict(zip(backends, energies, strict=True)))


def test_measured_receipt_reports_cv_per_metric() -> None:
    receipt = _measured()
    validate_receipt(receipt)
    assert receipt["cv_summary"]["total_energy_final"]["cv"] == pytest.approx(1 / 11)


def test_near_zero_mean_cv_is_undefined_not_perfect() -> None:
    receipt = _measured(energies=(1.0, -1.0))
    validate_receipt(receipt)
    entry = receipt["cv_summary"]["total_energy_final"]
    assert entry["cv"] is None
    assert "undefined" in entry["reason"]


def test_stub_engines_are_named_and_never_counted() -> None:
    backends = {
        "mujoco": BACKEND_REAL,
        "drake": BACKEND_REAL,
        "pinocchio": BACKEND_STUB,
    }
    receipt = _build(backends, {"mujoco": 10.0, "drake": 12.0, "pinocchio": 1e6})
    validate_receipt(receipt)
    assert receipt["stubbed_engines"] == ["pinocchio"]
    assert receipt["engines"]["pinocchio"]["counted"] is False
    assert receipt["cv_summary"]["total_energy_final"]["cv"] == pytest.approx(1 / 11)


def test_fewer_than_two_real_engines_is_deferred_without_numbers() -> None:
    receipt = _build({"mujoco": BACKEND_REAL, "drake": BACKEND_STUB}, {"mujoco": 10.0})
    validate_receipt(receipt)
    assert receipt["status"] == "deferred_insufficient_real_engines"
    assert receipt["cv_summary"] is None


@pytest.mark.parametrize("field", SETUP_FIELDS)
@pytest.mark.parametrize("change", ["missing", "disagrees"])
def test_incomplete_or_disagreeing_setup_is_rejected(field: str, change: str) -> None:
    receipt = _measured()
    if change == "missing":
        del receipt["engines"]["drake"][field]
    else:
        receipt["engines"]["drake"][field] = "other"
    with pytest.raises(ReceiptContractError, match=field):
        validate_receipt(receipt)


def test_measured_receipt_cannot_count_a_stub_backend() -> None:
    receipt = _measured()
    receipt["engines"]["drake"]["backend"] = BACKEND_STUB
    with pytest.raises(ReceiptContractError, match="stub"):
        validate_receipt(receipt)


@pytest.mark.parametrize(
    "bad",
    [{"dt_s": 0.0}, {"dt_s": float("nan")}, {"integrator": ""}, {"seed": True}]
    + [{"interpolation_mode": "cubic"}],
)
def test_invalid_setup_fails_closed(bad: dict) -> None:
    with pytest.raises(ValueError):
        SharedSetup.from_spec(TRIPLE_PENDULUM_SPEC, **{**SETUP_ARGS, **bad})
