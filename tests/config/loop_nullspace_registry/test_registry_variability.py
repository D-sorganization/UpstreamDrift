"""NSX-80 (#12709) variability hypotheses and citations in the shipped registry."""

from __future__ import annotations

import copy

import pytest

from src.shared.python.loop_nullspace import registry as reg_mod

pytestmark = pytest.mark.unit

NEW_HYPOTHESES = (
    "H-NOISE-ADDITIVE",
    "H-NOISE-SIGNAL-DEPENDENT",
    "H-GEOMETRIC-PRELOAD",
    "H-DECLARED-JOINT-IMPEDANCE",
    "H-SENSORIMOTOR-DELAY",
    "H-TIMING-JITTER",
    "H-MUSCLE-SHORT-RANGE-STIFFNESS",
)
NEW_CITATIONS = (
    "CIT-HARRIS-WOLPERT-1998",
    "CIT-JONES-HAMILTON-WOLPERT-2002",
    "CIT-OPTIMAL-FEEDBACK-2002",
    "CIT-SCHOLZ-SCHONER-1999",
    "CIT-MULLER-STERNAD-2004",
    "CIT-BURDET-2001",
    "CIT-CUSUMANO-DINGWELL-2013",
    "CIT-CUI-2008",
)


@pytest.fixture(scope="module")
def shipped() -> reg_mod.Registry:
    return reg_mod.load_registry()


def test_all_new_hypotheses_and_citations_present(shipped: reg_mod.Registry) -> None:
    for hid in NEW_HYPOTHESES:
        assert shipped.hypothesis(hid).statement.strip()
    for cid in NEW_CITATIONS:
        assert shipped.citation_status(cid) == "VERIFIED", cid


def test_no_statement_cites_a_source_id(shipped: reg_mod.Registry) -> None:
    for hid in NEW_HYPOTHESES:
        text = shipped.hypothesis(hid).statement
        assert not any(cid in text for cid in NEW_CITATIONS), hid


def test_new_hypotheses_may_be_displayed(shipped: reg_mod.Registry) -> None:
    assert shipped.require_verified_for_display(NEW_HYPOTHESES) == NEW_HYPOTHESES


def test_cited_hypotheses_rest_on_verified_sources(shipped: reg_mod.Registry) -> None:
    expected = {
        "H-NOISE-SIGNAL-DEPENDENT": "CIT-HARRIS-WOLPERT-1998",
        "H-DECLARED-JOINT-IMPEDANCE": "CIT-BURDET-2001",
        "H-MUSCLE-SHORT-RANGE-STIFFNESS": "CIT-CUI-2008",
    }
    for hid, cid in expected.items():
        hyp = shipped.hypothesis(hid)
        assert hyp.source.citation_id == cid
        assert hyp.verification_status == "VERIFIED"


def test_planted_unverified_id_is_rejected_by_the_gate() -> None:
    raw = copy.deepcopy(reg_mod.load_raw_registry())
    raw["citations"].append(
        {
            "id": "CIT-PLANTED-2099",
            "reference": "Planted",
            "verification_status": "UNVERIFIED",
        }
    )
    raw["hypotheses"].append(
        {
            "id": "H-PLANTED",
            "statement": "A planted claim.",
            "parameters": {},
            "source": {"citation_id": "CIT-PLANTED-2099", "note": "planted"},
            "verification_status": "UNVERIFIED",
        }
    )
    loaded = reg_mod.parse_registry(raw)
    with pytest.raises(ValueError, match="H-PLANTED"):
        loaded.require_verified_for_display(["H-PLANTED"])
    # Quoting the unverified id in user-facing text fails registry validation.
    raw["hypotheses"][-1]["statement"] = "Shown as CIT-PLANTED-2099 says."
    with pytest.raises(ValueError, match="CIT-PLANTED-2099"):
        reg_mod.parse_registry(raw)
    # A preset may not include it.
    raw["hypotheses"][-1]["statement"] = "A planted claim."
    raw["presets"].append(
        {
            "id": "planted",
            "label": "Planted",
            "hypotheses": ["H-PLANTED"],
            "conventions": [],
        }
    )
    with pytest.raises(ValueError, match="H-PLANTED"):
        reg_mod.parse_registry(raw)
