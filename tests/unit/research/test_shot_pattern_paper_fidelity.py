"""Unit tests asserting paper fidelity against saved scoring inputs and bound summaries (#12159)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re

import pytest

from src.tools.shot_pattern_analysis.scenario_scoring import score_course_endpoints

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "docs" / "research" / "shot_pattern_analysis"
TEX_PATH = DOCS_DIR / "shot_pattern_analysis.tex"
TRACEABILITY_PATH = (
    DOCS_DIR
    / "phase2"
    / "validation"
    / "selected_headline_traceability_v1"
    / "review.json"
)
DATASET_PATH = DOCS_DIR / "phase2" / "statistics_v2" / "headline_dataset.json"


@pytest.fixture(scope="module")
def paper_text() -> str:
    assert TEX_PATH.exists(), f"Article LaTeX source missing: {TEX_PATH}"
    return TEX_PATH.read_text(encoding="utf-8")


def test_course_scoring_inputs_match_saved_config_and_code(paper_text: str) -> None:
    """Issue #12159: Driver hole is 400 m, corridor half-width is 15 m (30 m total),

    and approach green radius is 15 m. Erroneous values (380 m hole, 10 m green) must be absent.
    """
    sig = inspect.signature(score_course_endpoints)
    assert sig.parameters["hole_distance_m"].default == 400.0
    assert sig.parameters["fairway_half_width_m"].default == 15.0
    assert sig.parameters["green_radius_m"].default == 15.0

    # Ensure actual course context values are present in paper prose
    assert "400" in paper_text
    assert re.search(
        r"400\\,m\s+hole\s+with\s+a\s+30\\,m\s+fairway\s+corridor", paper_text
    )
    assert re.search(r"centered\s+15\\,m-radius\s+green", paper_text)
    assert re.search(
        r"400\s*m\s+tee\s+opportunity\s+for\s+Driver.*15\s*m\s+fairway\s+half-width",
        paper_text,
    )
    assert re.search(r"centered\s+circular\s+green\s+of\s+radius\s+15\s*m", paper_text)

    # Discredited candidate values from #12159 must NOT appear in paper prose
    assert "380 m" not in paper_text
    assert "380\\,m" not in paper_text
    assert "10.0 m green" not in paper_text
    assert "10\\,m-radius green" not in paper_text
    assert "radius 10.0" not in paper_text


def test_nominal_face_search_bounds_match_bound_summaries(paper_text: str) -> None:
    """Issue #12159: Original face search bounds are reference face +/- 3 deg,

    not +/- 2 deg. Supplemental PW bounds are [-6, 6] deg and Driver control speed
    factors are [0.75, 1.30].
    """
    # Original search bounds must be reference face +/- 3 deg
    assert (
        re.search(r"reference\s+face\s+f_{\\rm\s+ref}\\pm3\^\\circ", paper_text)
        is not None
        or r"reference face $\pm3^\circ$" in paper_text
    )
    assert re.search(r"speed\s+\$\[0\.75,\s*1\.25\]", paper_text) is not None or (
        "speed factors $[0.75,1.25]$" in paper_text
    )

    # Supplemental bounds: PW uses absolute [-6, 6] deg and speed [0.75, 1.25]
    assert re.search(
        r"WP\s+and\s+WC\s+use\s+absolute\s+face\s+bounds\s+\$\[-6,\s*6\]\^\\circ\$\s+with\s+speed\s+factors\s+\$\[0\.75,\s*1\.25\]\$",
        paper_text,
    )

    # Driver control DC uses speed factor [0.75, 1.30]
    assert re.search(
        r"DC\s+uses\s+absolute\s+face\s+bounds\s+\$\[-6,\s*6\]\^\\circ\$\s+and\s+speed\s+factors\s+\$\[0\.75,\s*1\.30\]\$",
        paper_text,
    )

    # Erroneous candidate claim of +/- 2 deg bounds must NOT be present
    assert "plus/minus 2" not in paper_text.lower()
    assert r"\pm 2^\circ" not in paper_text
    assert r"\pm2^\circ" not in paper_text


def test_maximum_nominal_residual_matches_saved_traceability(paper_text: str) -> None:
    """Issue #12159: Maximum nominal residual among the 72 selected solutions is

    ~1.999733e-13 m, not 1.6e-13 m. Acceptance threshold is 0.05 m.
    """
    assert TRACEABILITY_PATH.exists()
    trace = json.loads(TRACEABILITY_PATH.read_text(encoding="utf-8"))
    residuals = [
        nom["residual_m"]
        for cell in trace["cells"]
        for nom in cell.get("nominals", {}).values()
    ]
    max_res = max(residuals)
    assert abs(max_res - 1.9997332019487617e-13) < 1e-18
    assert max_res < 0.05

    # Paper must state verified residual and 0.05 m criterion
    assert r"\numtext{+1.999733e-13}" in paper_text
    assert re.search(
        r"below\s+the\s+predeclared\s+\$?0\.05\$?\s*m\s+criterion", paper_text
    )

    # Contradicted candidate claim 1.6e-13 must be absent
    assert "1.6e-13" not in paper_text
    assert "1.6\\times 10^{-13}" not in paper_text


def test_protocol_identities_and_population_separation(paper_text: str) -> None:
    """Issue #12159: Primary C-keys and control E-keys are distinct. Ordered

    source protocols and distinct populations (individual endpoints vs common-three
    vs common-six) must be explicitly retained.
    """
    # Protocols must be documented with escaped underscores in LaTeX
    for proto in [
        r"\texttt{original\_primary/1}",
        r"\texttt{original\_control/1}",
        r"\texttt{pw-primary-face-bounds/1}",
        r"\texttt{pw-control-face-bounds/1}",
        r"\texttt{driver-control-speed-bounds/1}",
    ]:
        assert proto in paper_text

    # Distinct populations must be explicitly defined
    assert "individual supported endpoints" in paper_text.lower()
    assert "common-three scored population" in paper_text.lower()
    assert "common supported indices of all six populations" in paper_text.lower()

    # Verify C-keys and E-keys separation
    assert r"\input{generated/conditions}" in paper_text
    assert r"\input{generated/contrasts}" in paper_text
    assert "Most pairs come from the OC control matrix" in paper_text


def test_noncausal_scientific_claims_and_neutrality_disclaimers(
    paper_text: str,
) -> None:
    """Issue #12159: Unsupported causal claims (attributing exclusions to high kinetic energy,

    treating zero-spanning intervals as neutral/equivalent, or claiming universal shapes)
    must be disclaimed.
    """
    # High kinetic energy is explicitly disclaimed as cause of unsupported shots
    assert re.search(
        r"not\s+evidence\s+that\s+increased\s+kinetic\s+energy\s+caused\s+unsupported\s+shots",
        paper_text,
    )

    # Zero-spanning intervals are explicitly NOT proof of equivalence
    assert re.search(
        r"interactions\s+span\s+zero;\s+that\s+is\s+not\s+proof\s+of\s+equivalence\s+or\s+negligible\s+effects",
        paper_text,
    )

    # Universal best shape claim is refuted
    assert re.search(
        r"do\s+not\s+support\s+a\s+general\s+recommendation\s+to\s+prefer\s+one\s+shape",
        paper_text,
    )


def test_honest_review_and_compiler_claims(paper_text: str) -> None:
    """Issue #12159: Paper must not claim unperformed Astra/Root review approvals,

    and compiler status must be truthful.
    """
    # Fabricated editor/review claims must not be present
    assert "Astra review passed" not in paper_text
    assert "Root review passed" not in paper_text
    assert "publication accepted" not in paper_text.lower()
    assert "empirical qualification remains deferred" in paper_text.lower()

    # Truthful compiler statement
    assert "Tectonic 0.15.0" in paper_text
    assert "compilation is separate from empirical qualification" in paper_text
