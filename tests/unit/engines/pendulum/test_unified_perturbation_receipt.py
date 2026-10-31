"""Committed unified cross-engine perturbation receipt (SAP-6, #13795)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.analysis.cross_engine import METRIC_KEYS
from src.shared.python.analysis.cross_engine_receipt import (
    SETUP_FIELDS,
    STATUS_MEASURED,
    triple_pendulum_spec_hash,
    validate_receipt,
)
from src.shared.python.pendulum_simulator.step_convergence import (
    InHouseTripleAdapter,
    MuJoCoTripleAdapter,
)
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
)
from src.shared.python.pendulum_simulator.unified_perturbation_study import (
    STUDY,
    SteppableTripleEngine,
    UnifiedStudy,
    run_unified_study,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
RECEIPT = ROOT / "docs/validation/cross_engine_perturbation_receipt.json"
DOC = ROOT / "docs/validation/cross_engine_perturbation.md"


def _receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_receipt_records_one_shared_setup_for_every_engine() -> None:
    receipt = _receipt()
    validate_receipt(receipt)
    assert receipt["status"] == STATUS_MEASURED
    setup = receipt["setup"]
    assert setup["spec_hash"] == triple_pendulum_spec_hash(TRIPLE_PENDULUM_SPEC)
    assert setup["gravity_m_s2"] == TRIPLE_PENDULUM_SPEC.gravity_m_s2
    assert (setup["dt_s"], setup["integrator"]) == (STUDY.dt_s, STUDY.integrator)
    assert (setup["seed"], setup["interpolation_mode"]) == (STUDY.seed, "zoh")
    assert len(receipt["engines"]) >= 2
    band = receipt["sensitivity_context"]["within_engine_trial_cv"]
    for name, entry in receipt["engines"].items():
        assert entry["counted"] and entry["backend"] == "real"
        assert {field: entry[field] for field in SETUP_FIELDS} == setup
        assert set(band[name]) == set(METRIC_KEYS)


def test_doc_states_every_measured_cv() -> None:
    doc = DOC.read_text(encoding="utf-8")
    for metric, cv in _receipt()["cv_summary"].items():
        shown = "undefined" if cv["cv"] is None else f"{cv['cv']:.3e}"
        assert f"| {metric} | {shown} |" in doc


@pytest.mark.slow
def test_regeneration_reproduces_the_committed_receipt() -> None:
    pytest.importorskip("mujoco")
    fresh, committed = run_unified_study(), _receipt()
    assert fresh["setup"] == committed["setup"]
    for metric, cv in committed["cv_summary"].items():
        got = fresh["cv_summary"][metric]["cv"]
        assert got == pytest.approx(cv["cv"], rel=1e-6, abs=1e-15)


def test_short_study_is_measured_with_one_setup() -> None:
    pytest.importorskip("mujoco")
    receipt = run_unified_study(study=UnifiedStudy(t_end_s=0.05, n_trials=2))
    assert receipt["status"] == STATUS_MEASURED
    assert {e["dt_s"] for e in receipt["engines"].values()} == {STUDY.dt_s}


def test_independent_adapters_agree_on_energy_and_tip_velocity() -> None:
    pytest.importorskip("mujoco")
    engines = [
        SteppableTripleEngine(a)
        for a in (InHouseTripleAdapter(), MuJoCoTripleAdapter())
    ]
    for engine in engines:
        engine.reset()
    assert engines[0].get_total_energy() == pytest.approx(
        engines[1].get_total_energy(), rel=1e-9
    )
    for engine in engines:
        for _ in range(50):
            engine.set_control(np.zeros(3))
            engine.step(STUDY.dt_s)
    np.testing.assert_allclose(
        engines[0].get_end_effector_velocity(),
        engines[1].get_end_effector_velocity(),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda e: e.step(0.0), "dt"),
        (lambda e: e.step(math.nan), "dt"),
        (lambda e: e.set_control(np.zeros(2)), "control"),
    ],
)
def test_wrapper_fails_closed(call, match: str) -> None:
    engine = SteppableTripleEngine(InHouseTripleAdapter())
    engine.reset()
    with pytest.raises(ValueError, match=match):
        call(engine)
