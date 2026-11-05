"""R11 (#12691) evidence gate: the chain-shaft receipt covers every chain engine.

The parity tests in ``test_flex_shaft_parity.py`` run only where an engine is
installed.  This gate is engine-free: it reads the committed receipt and fails
closed when an engine in ``CHAIN_ENGINES`` has no impact-window deflection
entry, when an entry differs from the MuJoCo chain by more than the tolerance,
or when the overlay figure is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.club_shaft_chain import CHAIN_ENGINES

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/flex_shaft/r11"
RECEIPT = EVIDENCE / "r11_chain_receipt.json"
FIGURE = EVIDENCE / "r11_chain_deflection_impact_window.png"
#: Evidence-gate bound (not a parity tolerance) on the Clubhead deflection
#: difference to the MuJoCo chain from the window start to contact, metres.
#: Recorded differences are at most 1.3e-10 m against deflections of 0.1 m or
#: more; the replay parity tolerances stay in ``test_flex_shaft_parity.py``.
DEFLECTION_ATOL_M = 1e-8
#: The same bundle must have been replayed by every engine.
REFERENCE_ENGINE = "mujoco"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_receipt_has_an_impact_window_entry_for_every_chain_engine(
    receipt: dict,
) -> None:
    recorded = set(receipt["impact_window"]["engines"])
    assert recorded == set(CHAIN_ENGINES), (
        f"missing {sorted(set(CHAIN_ENGINES) - recorded)}, "
        f"unexpected {sorted(recorded - set(CHAIN_ENGINES))}"
    )


def test_each_engine_matches_the_mujoco_deflection(receipt: dict) -> None:
    engines = receipt["impact_window"]["engines"]
    assert engines[REFERENCE_ENGINE]["max_deflection_difference_vs_mujoco_m"] == 0.0
    for name, entry in engines.items():
        assert entry["max_deflection_difference_vs_mujoco_m"] <= DEFLECTION_ATOL_M, name


def test_receipt_does_not_claim_a_supported_engine_is_refused(receipt: dict) -> None:
    refusal = receipt["refusal"].lower()
    for name in CHAIN_ENGINES:
        assert name not in refusal, (
            f"{name} supports the chain but the receipt says refused"
        )
    assert "simscape" in refusal


def test_overlay_figure_is_committed() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 10_000
