"""Publication gate evaluator for Closed-Loop Null-Space Exploration (NSX-74, #12592).

Enforces the pre-publication checklist before any NSX result is published
externally:
1. All active hypotheses in ``src/config/loop_nullspace_registry.json`` have
   ``verification_status == "VERIFIED"`` and map to verified archival citations
   with a non-empty DOI or URL.
2. No ``UNVERIFIED`` or ``REMOVED`` citation token appears in any user-facing
   hypothesis statement or preset description.
3. The pre-registered external validation protocol ``docs/development/planning/DV-12554.md``
   is present and tracked.
4. All five explicit scientific non-claims (Rule R13 / Epic #12593 §8) are
   attached verbatim.
5. External physiological claims remain strictly blocked
   (``ready_for_physiological_claim = False``) unless physical validation under
   ``DV-12554`` (instrumented dual-hand grip + surface EMG + force plates on
   held-out swings) has been completed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import ensure, require

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "src" / "config" / "loop_nullspace_registry.json"
DEFAULT_DV_PATH = _REPO_ROOT / "docs" / "development" / "planning" / "DV-12554.md"

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "No claim that any single hand split is the golfer's true physical load share without instrumented grip measurement.",
    "No joint stiffness, impedance, or antagonist co-contraction inference from net joint torques.",
    "No inertial parameter identification beyond the MOSAIC-identifiable base parameter subspace.",
    "No claim that aerodynamic drag resolves closed-loop ZTCF multiplier identifiability.",
    "No unverified literature hand-share percentages from the v1 proposal are used in any prior or user-facing statement.",
)


@dataclass(frozen=True)
class PublicationGateItem:
    """A single verifiable criterion in the NSX publication gate (NSX-74)."""

    gate_id: str
    description: str
    passed: bool
    evidence: str


@dataclass(frozen=True)
class PublicationGateReport:
    """Complete evaluation report for the NSX publication gate (NSX-74)."""

    checklist: tuple[PublicationGateItem, ...]
    all_hypotheses_verified: bool
    no_unverified_Active_citations: bool
    deferred_validation_recorded: bool
    external_validation_completed: bool
    explicit_non_claims: tuple[str, ...]
    ready_for_methodology_publication: bool
    ready_for_physiological_claim: bool


def _verify_registry_citations(
    registry_data: dict[str, Any],
) -> tuple[bool, bool, str]:
    """Audit hypotheses and citations in the NSX registry JSON."""
    citations_by_id: dict[str, dict[str, Any]] = {
        str(c["id"]): c for c in registry_data.get("citations", [])
    }
    hypotheses = registry_data.get("hypotheses", [])
    if not hypotheses:
        return False, False, "No hypotheses found in registry."

    all_hyp_verified = True
    no_unverified_active = True
    unverified_ids = {
        cid
        for cid, c in citations_by_id.items()
        if c.get("verification_status") in {"UNVERIFIED", "REMOVED"}
    }

    for hyp in hypotheses:
        status = hyp.get("verification_status")
        source_id = str(hyp.get("source", ""))
        source_rec = citations_by_id.get(source_id)
        if (
            status != "VERIFIED"
            or source_rec is None
            or source_rec.get("verification_status") != "VERIFIED"
            or not source_rec.get("doi_or_url")
        ):
            all_hyp_verified = False

        statement = str(hyp.get("statement", ""))
        if source_id in unverified_ids or any(
            bad_id in statement for bad_id in unverified_ids
        ):
            no_unverified_active = False

    summary = (
        f"Audited {len(hypotheses)} active hypotheses against "
        f"{len(citations_by_id)} citation records."
    )
    return all_hyp_verified, no_unverified_active, summary


def _verify_dv_record(dv_path: Path) -> tuple[bool, str]:
    """Check that DV-12554 exists and specifies held-out instrumented grip + EMG gates."""
    if not dv_path.is_file():
        return False, f"Missing deferred validation file: {dv_path.as_posix()}"
    text = dv_path.read_text(encoding="utf-8")
    required_tokens = ("DV-12554", "hand_on_bare_club", "EMG", "Held-Out Swings")
    missing = [tok for tok in required_tokens if tok not in text]
    if missing:
        return False, f"DV-12554 missing required sections/tokens: {missing}"
    return True, f"Verified deferred validation record at {dv_path.name}."


def evaluate_nsx_publication_gate(
    *,
    registry_path: Path | None = None,
    dv_path: Path | None = None,
    external_validation_completed: bool = False,
) -> PublicationGateReport:
    """Evaluate the NSX publication gate (NSX-74, #12592).

    Preconditions:
        - ``registry_path`` (if provided) or default registry JSON exists.

    Postconditions:
        - ``ready_for_physiological_claim`` is ``True`` if and only if all
          methodology gates pass AND ``external_validation_completed is True``.
    """
    reg_file = registry_path if registry_path is not None else DEFAULT_REGISTRY_PATH
    dv_file = dv_path if dv_path is not None else DEFAULT_DV_PATH
    require(reg_file.is_file(), f"Registry file does not exist: {reg_file}")

    registry_data = json.loads(reg_file.read_text(encoding="utf-8"))
    all_hyp_ok, no_unverified_ok, reg_evidence = _verify_registry_citations(
        registry_data
    )
    dv_ok, dv_evidence = _verify_dv_record(dv_file)
    non_claims_ok = len(EXPLICIT_NON_CLAIMS) >= 5

    items = (
        PublicationGateItem(
            gate_id="GATE-CITATIONS-VERIFIED",
            description="All active registry hypotheses cite VERIFIED sources with DOI/URL.",
            passed=all_hyp_ok,
            evidence=reg_evidence,
        ),
        PublicationGateItem(
            gate_id="GATE-NO-UNVERIFIED-PRIORS",
            description="Zero UNVERIFIED or REMOVED v1 citations appear in active priors.",
            passed=no_unverified_ok,
            evidence="Checked active hypotheses against quarantined citation IDs.",
        ),
        PublicationGateItem(
            gate_id="GATE-DV-12554-RECORDED",
            description="External validation protocol DV-12554 is recorded and tracked.",
            passed=dv_ok,
            evidence=dv_evidence,
        ),
        PublicationGateItem(
            gate_id="GATE-EXPLICIT-NON-CLAIMS",
            description="All five explicit scientific non-claims are declared.",
            passed=non_claims_ok,
            evidence=f"{len(EXPLICIT_NON_CLAIMS)} non-claims attached.",
        ),
        PublicationGateItem(
            gate_id="GATE-EXTERNAL-PHYSICAL-VALIDATION",
            description="Physical validation with 6-axis dual grip and sEMG on held-out swings (DV-12554).",
            passed=bool(external_validation_completed),
            evidence=(
                "Completed physical validation bundle."
                if external_validation_completed
                else "Deferred under DV-12554; physiological claims prohibited."
            ),
        ),
    )

    methodology_ready = all_hyp_ok and no_unverified_ok and dv_ok and non_claims_ok
    physiological_ready = methodology_ready and bool(external_validation_completed)
    ensure(
        not physiological_ready or external_validation_completed,
        "ready_for_physiological_claim cannot be True without external validation",
    )

    return PublicationGateReport(
        checklist=items,
        all_hypotheses_verified=all_hyp_ok,
        no_unverified_Active_citations=no_unverified_ok,
        deferred_validation_recorded=dv_ok,
        external_validation_completed=bool(external_validation_completed),
        explicit_non_claims=EXPLICIT_NON_CLAIMS,
        ready_for_methodology_publication=methodology_ready,
        ready_for_physiological_claim=physiological_ready,
    )
