"""Two-capture coverage matrix generation and non-golf tool exclusions (TB-00 #10585).

Provides a programmatic, auditable coverage table covering every registered model
across both Driver (360 Hz, 654 frames) and 7-Iron (359 Hz, 657 frames) captures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import EvidenceStatus, GolfModelIdentity
from .registry import list_golf_models


@dataclass(frozen=True)
class CoverageCell:
    """Coverage status for a single (model, capture) pair."""

    model_id: str
    capture: str  # "driver" or "iron"
    supported: bool
    observation_set: str
    existing_artifact: str | None
    ownership: str
    missing_adapter: str | None
    blocked_reason: str | None
    governing_issue: str
    evidence_status: EvidenceStatus


@dataclass(frozen=True)
class ToolExclusion:
    """Explicit exclusion of a non-golf launcher tool."""

    tool_id: str
    name: str
    reason: str
    category: str


_EXCLUDED_TOOLS: tuple[ToolExclusion, ...] = (
    ToolExclusion(
        tool_id="bunkershot",
        name="Bunker Shot Simulation",
        reason="Sand-trap particle/contact game and visualization; not a full tour swing trajectory.",
        category="special_app",
    ),
    ToolExclusion(
        tool_id="shot_tracer",
        name="Shot Tracer",
        reason="Aerodynamic post-impact ball flight model (Waterloo, MacDonald, Nathan); not swing biomechanics.",
        category="simulation",
    ),
    ToolExclusion(
        tool_id="cross_engine_dashboard",
        name="Cross-Engine Dashboard",
        reason="Engine-level perturbation and numerical robustness analyzer; not a biomechanical baseline model.",
        category="simulation",
    ),
    ToolExclusion(
        tool_id="pose_studio",
        name="Pose Studio",
        reason="Interactive cross-engine kinematic pose authoring editor; not a dynamic golf swing fitter.",
        category="tool",
    ),
    ToolExclusion(
        tool_id="starting_pose_matcher",
        name="Starting-Pose Matcher",
        reason="Static address-frame alignment solver; does not track or simulate continuous swing dynamics.",
        category="tool",
    ),
    ToolExclusion(
        tool_id="force_plate_lab",
        name="Force Plate Lab",
        reason="Ground reaction force visualizer and COP diagnostics; consumes GRF traces without solving swing dynamics.",
        category="tool",
    ),
    ToolExclusion(
        tool_id="swing_plane_analyzer",
        name="Swing Plane Analyzer",
        reason="Pure geometric plane fitting and club-shaft inclination utility; does not simulate body or club physics.",
        category="tool",
    ),
    ToolExclusion(
        tool_id="putting_green",
        name="Putting Green Simulator",
        reason="Short-game pendulum putting surface and ball roll physics; excluded from full-swing tour baselines.",
        category="simulation",
    ),
    ToolExclusion(
        tool_id="camera_setup",
        name="Camera Setup Wizard",
        reason="Multi-camera extrinsics and optical calibration tool; does not model golfer dynamics.",
        category="tool",
    ),
    ToolExclusion(
        tool_id="coaching_drawings",
        name="Coaching Drawings Tool",
        reason="Video 2D overlay and angle measurement annotations for golf instruction; not a physics model.",
        category="tool",
    ),
    ToolExclusion(
        tool_id="model_explorer",
        name="Model Explorer",
        reason="3D asset hierarchy and mesh inspector; does not simulate biomechanics.",
        category="tool",
    ),
    ToolExclusion(
        tool_id="calibration_wizard",
        name="Sensor Calibration Wizard",
        reason="IMU and mocap sensor calibration utility; not a fittable biomechanical model.",
        category="tool",
    ),
)


_DRIVEN_DOUBLE_RECEIPTS: dict[str, str] = {
    "driver": "docs/plans/tour_baselines/evidence/tb04_driver_qualification_receipt.json",
    "iron": "docs/plans/tour_baselines/evidence/tb04_iron_qualification_receipt.json",
}

_DRIVEN_DOUBLE_REJECTION = (
    "TB-04 receipt is DISQUALIFIED: solver reached max iterations and marker "
    "accuracy exceeds its declared threshold."
)


def list_excluded_tools() -> list[ToolExclusion]:
    """Return all non-golf launcher tools with explicit exclusion rationales."""
    return list(_EXCLUDED_TOOLS)


def _cell_reconstruction(m: GolfModelIdentity, capture: str) -> CoverageCell:
    obs_set = (
        "Shoulder and hands markers (wrists midpoint)"
        if "double" in m.model_id
        else (
            "Shoulder, elbow, and hands markers"
            if "triple" in m.model_id
            else "Upper/lower body anatomical markers (no club)"
        )
    )
    artifact = f"docs/development/reference_fitting_epic.md (Epic #9914, {capture})"
    return CoverageCell(
        model_id=m.model_id,
        capture=capture,
        supported=True,
        observation_set=obs_set,
        existing_artifact=artifact,
        ownership=m.source_owner.value,
        missing_adapter=None,
        blocked_reason=None,
        governing_issue="#9914",
        evidence_status=EvidenceStatus.HISTORICAL_REFERENCE,
    )


def _cell_driven(m: GolfModelIdentity, capture: str) -> CoverageCell:
    if m.model_id == "driven_double_pendulum":
        return CoverageCell(
            model_id=m.model_id,
            capture=capture,
            supported=True,
            observation_set="Projected 2D swing plane (shoulder pivot + clubhead/grip)",
            existing_artifact=(f"{_DRIVEN_DOUBLE_RECEIPTS[capture]} (DISQUALIFIED)"),
            ownership="Tools",
            missing_adapter=None,
            blocked_reason=_DRIVEN_DOUBLE_REJECTION,
            governing_issue="#10589",
            evidence_status=EvidenceStatus.REJECTED,
        )
    return CoverageCell(
        model_id=m.model_id,
        capture=capture,
        supported=True,
        observation_set="Projected 2D swing plane (shoulder pivot + clubhead/grip)",
        existing_artifact=None,
        ownership="Tools",
        missing_adapter="Pending TB-04 / TB-05 bounded dynamic fitting campaign",
        blocked_reason=None,
        governing_issue="#10589" if "double" in m.model_id else "#10590",
        evidence_status=EvidenceStatus.UNQUALIFIED,
    )


def _cell_upper_body(m: GolfModelIdentity, capture: str) -> CoverageCell:
    return CoverageCell(
        model_id=m.model_id,
        capture=capture,
        supported=True,
        observation_set="3D upper torso, bilateral arms, and clubhead trajectory",
        existing_artifact=None,
        ownership="Tools",
        missing_adapter="Pending TB-06 upper-body constrained dynamic fitter",
        blocked_reason=None,
        governing_issue="#10591",
        evidence_status=EvidenceStatus.UNQUALIFIED,
    )


def _cell_full_body(m: GolfModelIdentity, capture: str) -> CoverageCell:
    engine_name = m.backend.value
    artifact = (
        f"evidence/matched/{capture}_g1_crocoddyl_rk45_b100/ (REJECTED)"
        if engine_name == "pinocchio" and capture == "driver"
        else (
            f"docs/development/full_body_models/evidence/ground_support/anthro_{capture}_shoot_g025/receipt.json"
            if engine_name == "mujoco"
            else (
                "docs/development/simscape_tour_matching/native_evidence/run-102.json"
                if engine_name == "simscape" and capture == "driver"
                else None
            )
        )
    )
    blocked = (
        "Requires MyoSuite environment & retargeting (fail-closed per MS-50)"
        if engine_name == "myosuite"
        else (
            "Pending Moco full-swing tracking convergence under MS-102"
            if engine_name == "opensim"
            else (
                "Terminal error 40.3 mm > 35 mm gate; pinned to MATLAB R2025b"
                if engine_name == "simscape"
                else (
                    "Pending dual-club G3 cross-engine qualification (#10378)"
                    if capture == "iron"
                    else None
                )
            )
        )
    )
    supported = engine_name != "myosuite"
    return CoverageCell(
        model_id=m.model_id,
        capture=capture,
        supported=supported,
        observation_set="Full 38 C3D markers + dual ground reaction force plates",
        existing_artifact=artifact,
        ownership="UpstreamDrift",
        missing_adapter="MyoSuite adapter" if not supported else None,
        blocked_reason=blocked,
        governing_issue="#10378",
        evidence_status=(
            EvidenceStatus.UNAVAILABLE
            if not supported
            else (
                EvidenceStatus.G1_KINEMATIC_PASSED
                if engine_name == "mujoco" and capture == "driver"
                else EvidenceStatus.NATIVE_CANDIDATE
            )
        ),
    )


def _cell_reference_or_placeholder(
    m: GolfModelIdentity, capture: str
) -> CoverageCell | None:
    if m.model_id.startswith("reference_"):
        artifact = f"docs/development/reference_fitting_epic.md (Epic #9914, {capture})"
        return CoverageCell(
            model_id=m.model_id,
            capture=capture,
            supported=True,
            observation_set="Mapped subset of anatomical markers (per URDF preset)",
            existing_artifact=artifact,
            ownership="UpstreamDrift",
            missing_adapter=None,
            blocked_reason=None,
            governing_issue="#9914",
            evidence_status=EvidenceStatus.HISTORICAL_REFERENCE,
        )
    if m.model_id in ("myosuite_body", "opensim_golfer"):
        reason = (
            "Bundled myobody assets are placeholders, not MyoSuite anatomy"
            if m.model_id == "myosuite_body"
            else "Native OpenSim custom-joint/muscle constraints require OpenSim adapter"
        )
        return CoverageCell(
            model_id=m.model_id,
            capture=capture,
            supported=False,
            observation_set="None (placeholder)",
            existing_artifact=None,
            ownership="UpstreamDrift",
            missing_adapter=f"{m.model_id} native adapter",
            blocked_reason=reason,
            governing_issue="#9914",
            evidence_status=EvidenceStatus.UNAVAILABLE,
        )
    return None


def _cell_for_model(m: GolfModelIdentity, capture: str) -> CoverageCell | None:
    if m.model_id.startswith("reconstruction_"):
        return _cell_reconstruction(m, capture)
    if m.model_id.startswith("driven_"):
        return _cell_driven(m, capture)
    if m.model_id == "constrained_upper_body_golfer":
        return _cell_upper_body(m, capture)
    if m.model_id.startswith("full_body_"):
        return _cell_full_body(m, capture)
    return _cell_reference_or_placeholder(m, capture)


def generate_coverage_matrix() -> list[CoverageCell]:
    """Generate the authoritative Model x {Driver, Iron} coverage matrix.

    Evaluates every registered model across both captures, reporting:
    - Supported observation set
    - Existing artifacts
    - Source ownership
    - Missing adapters / blockers
    - Governing issue
    - Fail-closed evidence status
    """
    cells: list[CoverageCell] = []
    for m in list_golf_models():
        for capture in ("driver", "iron"):
            cell = _cell_for_model(m, capture)
            if cell is not None:
                cells.append(cell)
    return cells


def render_coverage_markdown() -> str:
    """Render the authoritative two-capture coverage matrix and tool exclusions as Markdown."""
    cells = generate_coverage_matrix()
    exclusions = list_excluded_tools()

    lines = [
        "# Tour Baselines Coverage Matrix",
        "",
        "Authoritative mapping of all registered golf models across Driver (360 Hz) and 7-Iron (359 Hz) captures.",
        "Governed by Matched Swing Program ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363), "
        "[#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584), "
        "[#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585)).",
        "",
        "## 1. Two-Capture Model Coverage",
        "",
        "| Model ID | Capture | Supported | Observation Set | Ownership | Status | Existing Artifact / Blocker | Governing Issue |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for c in cells:
        status_icon = (
            "✅"
            if c.evidence_status
            in (
                EvidenceStatus.G1_KINEMATIC_PASSED,
                EvidenceStatus.G2_DYNAMIC_PASSED,
                EvidenceStatus.G3_RELEASED,
            )
            else (
                "🏛️"
                if c.evidence_status == EvidenceStatus.HISTORICAL_REFERENCE
                else (
                    "⚙️"
                    if c.evidence_status == EvidenceStatus.NATIVE_CANDIDATE
                    else (
                        "❌"
                        if c.evidence_status
                        in (EvidenceStatus.UNAVAILABLE, EvidenceStatus.REJECTED)
                        else "⏳"
                    )
                )
            )
        )
        artifact_or_blocker = c.existing_artifact or (
            c.blocked_reason or c.missing_adapter or "None"
        )
        lines.append(
            f"| `{c.model_id}` | **{c.capture.capitalize()}** | {'Yes' if c.supported else 'No'} | "
            f"{c.observation_set} | {c.ownership} | {status_icon} {c.evidence_status.value} | "
            f"{artifact_or_blocker} | {c.governing_issue} |"
        )

    lines.extend(
        [
            "",
            "## 2. Non-Golf Tool Exclusions",
            "",
            "The following tools from `src/config/models.yaml` are intentionally excluded from the Tour Baselines matrix:",
            "",
            "| Tool ID | Display Name | Category | Exclusion Rationale |",
            "|---|---|---|---|",
        ]
    )

    for e in exclusions:
        lines.append(f"| `{e.tool_id}` | {e.name} | `{e.category}` | {e.reason} |")

    lines.append("")
    return "\n".join(lines)
