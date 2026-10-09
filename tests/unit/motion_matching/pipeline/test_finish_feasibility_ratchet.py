"""Regression ratchet on the canonical finish feasibility (#11668).

``finish_feasibility_baseline.json`` beside the canonical receipts records the
finish-window feasibility of the driver and 7-iron static-seed runs when the
metrics were introduced. A later change may only improve the ratcheted
fractions: the baseline must validate against the receipt schema, and any
canonical receipt that carries ``dynamics.finish_feasibility`` must not fall
below it. Balance-2 (#11669) raises the inside fractions and the baseline moves
with it, in the pull request that proves the improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.pipeline.finish_feasibility import (
    ratcheted_metrics,
    regressions,
)
from src.shared.python.motion_matching.pipeline.receipt_dynamics import (
    FinishFeasibilityReceipt,
)

pytestmark = pytest.mark.unit

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
BASELINE = EVIDENCE / "finish_feasibility_baseline.json"
CANONICAL = ("anthro_driver_seeds", "anthro_iron_seeds_zmp")


def _baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", CANONICAL)
def test_baseline_validates_against_the_receipt_schema(name: str) -> None:
    block = FinishFeasibilityReceipt.model_validate(_baseline()[name])
    assert block.window_s[0] == 1.0
    assert (
        block.simulation.vertical_force_bw_min <= block.simulation.vertical_force_bw_max
    )


@pytest.mark.parametrize("name", CANONICAL)
def test_baseline_pins_the_ratcheted_fractions(name: str) -> None:
    for side in ("reference", "simulation"):
        metrics = _baseline()[name][side]
        assert 0.0 <= metrics["zmp_inside_fraction_1_0_to_1_5s"] <= 1.0
        assert 0.0 <= metrics["friction_saturated_fraction"] <= 1.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#12040: main no longer reproduces the #11668 baseline, and the OSV-6 "
        "receipts (#11737) lower the 7-iron reference fractions further; the "
        "baseline change is an owner decision"
    ),
)
@pytest.mark.parametrize("name", CANONICAL)
def test_receipt_finish_feasibility_has_not_regressed(name: str) -> None:
    receipt = json.loads((EVIDENCE / name / "receipt.json").read_text("utf-8"))
    block = receipt["dynamics"].get("finish_feasibility")
    if block is None:
        pytest.skip("canonical receipt predates finish-feasibility metrics")
    assert regressions(block, ratcheted_metrics(_baseline()[name])) == []
