"""Reconciliation of historical closed issues and Tools repository revisions (TB-00 #10585).

Audits closed epics #9914, #9921, #10003 against current evidence receipts,
demonstrating why closure does not constitute full-swing G3 qualification.
Records exact Tools submodule pins, gitlink history, and links full-body governing epics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReconciliationRecord:
    """Historical reconciliation entry for a closed issue."""

    issue_number: int
    title: str
    state: str
    scope: str
    is_full_swing_qualified: bool
    receipt_path: str | None
    notes: str
    governing_continuation: str


@dataclass(frozen=True)
class ToolsRevisionStatus:
    """Status and provenance of the external Tools repository dependency."""

    vendor_submodule_path: str
    pinned_commit_sha: str
    historical_review_gitlink: str
    owned_package_name: str
    source_owner: str
    update_policy: str


@dataclass(frozen=True)
class GoverningEpic:
    """Governing epic for ongoing full-body and matched swing work."""

    issue_number: int
    title: str
    owner_or_role: str
    mandate: str


_RECONCILIATION_RECORDS: tuple[ReconciliationRecord, ...] = (
    ReconciliationRecord(
        issue_number=9914,
        title="C3D Reference Fitting Epic",
        state="closed",
        scope="Kinematic reference overlay fitting on 9 catalog models",
        is_full_swing_qualified=False,
        receipt_path="docs/development/reference_fitting_epic.md",
        notes=(
            "Fitted 9 articulated/URDF catalog models to driver and iron C3D markers. "
            "Reconstruction double and triple pendulums attained low RMS (33.1 mm and 9.6 mm) "
            "solely because they omitted the club and tracked only wrists/elbows. "
            "Delivered reference playback assets; does not represent forward-dynamics or torque qualification."
        ),
        governing_continuation="#10584 (Tour Baselines Epic)",
    ),
    ReconciliationRecord(
        issue_number=9921,
        title="Simscape Tour Matching",
        state="closed",
        scope="MATLAB R2025b Simscape 3D Golf Model matching on driver tour average",
        is_full_swing_qualified=False,
        receipt_path="docs/development/simscape_tour_matching/native_evidence/run-102.json",
        notes=(
            "Simscape forward dynamics match in MATLAB R2025b (run-102: 0-0.85 s window, "
            "whole swing 20.3 mm, but terminal error 40.3 mm exceeded the 35 mm gate). "
            "Simscape serves as historical tour authority and cross-validation lane; "
            "remains strictly pinned to MATLAB R2025b."
        ),
        governing_continuation="#10363 / #10378",
    ),
    ReconciliationRecord(
        issue_number=10003,
        title="OpenSim Tour Matching",
        state="closed",
        scope="OpenSim Moco musculoskeletal tracking of tour capture",
        is_full_swing_qualified=False,
        receipt_path="docs/development/opensim_tour_matching/evidence/receipt.json",
        notes=(
            "Moco optimization rungs passed 0.10s and 0.30s calibration (41-42 mm); "
            "0.60s converged but open-loop replay diverged to 81 mm whole / 204 mm terminal. "
            "Model architecture merged under #10414 (OG-01..09); full-swing tracking remains active under MS-102."
        ),
        governing_continuation="#10376 (MS-102) / #10378",
    ),
)

_TOOLS_REVISION = ToolsRevisionStatus(
    vendor_submodule_path="vendor/ud-tools",
    pinned_commit_sha="a9ed0e7c5c6905b1164082659051d6381068052d",
    historical_review_gitlink="62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1",
    owned_package_name="double_pendulum_golf (Tools repo)",
    source_owner="Tools",
    update_policy=(
        "Tools repo owns double_pendulum_golf. Changes must be committed in Tools first, "
        "and UpstreamDrift pin updated via a reviewed pull request. Never edit vendor/ud-tools directly."
    ),
)

_GOVERNING_EPICS: tuple[GoverningEpic, ...] = (
    GoverningEpic(
        issue_number=10363,
        title="Matched Swing Program — Unified Cross-Engine Golf Swing Biomechanics",
        owner_or_role="Program Lead (`agent:local`)",
        mandate="Authoritative program for physical gates G1/G2/G3 across all 6 physics engines.",
    ),
    GoverningEpic(
        issue_number=10378,
        title="MS-104: Driver and 7-Iron Dual-Club G3 Coverage Across All Engines",
        owner_or_role="Full-Body Lead",
        mandate="Dual-club coverage requirement across MuJoCo, Pinocchio, Drake, OpenSim, Simscape, MyoSuite.",
    ),
    GoverningEpic(
        issue_number=10430,
        title="Multi-Engine Trajectory Service & Replay Qualification",
        owner_or_role="Verification Lead",
        mandate="Independent forward simulation replay verification for all candidate control trajectories.",
    ),
)


def get_historical_reconciliation() -> list[ReconciliationRecord]:
    """Return reconciliation records for closed issues #9914, #9921, #10003."""
    return list(_RECONCILIATION_RECORDS)


def get_tools_revision_status() -> ToolsRevisionStatus:
    """Return verified Tools submodule pin, historical gitlink divergence, and ownership policy."""
    return _TOOLS_REVISION


def get_governing_epics() -> list[GoverningEpic]:
    """Return the governing full-body epics linking this work to the Matched Swing Program."""
    return list(_GOVERNING_EPICS)


def render_reconciliation_markdown() -> str:
    """Render historical reconciliation and Tools revision tracking as Markdown."""
    records = get_historical_reconciliation()
    tools = get_tools_revision_status()
    epics = get_governing_epics()

    lines = [
        "# Historical Closed-Issue and Tools Revision Reconciliation",
        "",
        "Reconciles closed issues against current evidence receipts and establishes the Tools ownership boundary.",
        "Governed by Matched Swing Program ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363), "
        "[#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584), "
        "[#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585)).",
        "",
        "## 1. Closed Issue Reconciliation",
        "",
        "| Issue | Title | Scope | Full-Swing Qualified | Receipt / Evidence | Reconciliation Finding |",
        "|---|---|---|---|---|---|",
    ]

    for r in records:
        qualified_str = "✅ Yes" if r.is_full_swing_qualified else "❌ No"
        receipt_str = f"`{r.receipt_path}`" if r.receipt_path else "None"
        lines.append(
            f"| [#{r.issue_number}](https://github.com/D-sorganization/UpstreamDrift/issues/{r.issue_number}) | "
            f"**{r.title}** | {r.scope} | {qualified_str} | {receipt_str} | {r.notes} |"
        )

    lines.extend(
        [
            "",
            "## 2. Tools Submodule Revision Status & Ownership",
            "",
            f"- **Submodule Path:** `{tools.vendor_submodule_path}`",
            f"- **Current Verified HEAD Commit:** `{tools.pinned_commit_sha}`",
            f"- **Historical Review Gitlink:** `{tools.historical_review_gitlink}`",
            f"- **Owned Package:** `{tools.owned_package_name}`",
            f"- **Source Owner:** `{tools.source_owner}`",
            f"- **Update Policy:** {tools.update_policy}",
            "",
            "## 3. Governing Full-Body Issues",
            "",
            "| Issue | Title | Accountable Role | Mandate |",
            "|---|---|---|---|",
        ]
    )

    for e in epics:
        lines.append(
            f"| [#{e.issue_number}](https://github.com/D-sorganization/UpstreamDrift/issues/{e.issue_number}) | "
            f"**{e.title}** | `{e.owner_or_role}` | {e.mandate} |"
        )

    lines.append("")
    return "\n".join(lines)
