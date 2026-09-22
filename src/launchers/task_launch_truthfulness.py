"""Task Launch Truthfulness and Audit Contracts (ORG-03, Issue #10512).

Enforces launch truthfulness across UpstreamDrift:
1. Audits capabilities to distinguish production solvers, prototype demos,
   library-only algorithms, parametric CLIs, provider-required tools, and service previews.
2. Replaces misleading launches with honest explanations and next actions.
3. Requires FreeMoCap input/output selection before scheduling CLI; never launches
   argparse with zero arguments.
4. Prevents placeholder/example execution from reporting success.
5. Verifies process lifecycle truthfulness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LaunchDisposition(str, Enum):
    """Authoritative disposition defining how a capability may be launched."""

    PRODUCTION_SOLVER = "production_solver"
    PROTOTYPE_DEMO = "prototype_demo"
    LIBRARY_ONLY = "library_only"
    PROVIDER_REQUIRED = "provider_required"
    PARAMETRIC_CLI = "parametric_cli"
    SERVICE_PREVIEW = "service_preview"


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """Audit metadata describing launch requirements and constraints for a capability."""

    capability_id: str
    disposition: LaunchDisposition
    label: str
    explanation: str
    next_action: str | None = None
    required_parameters: tuple[str, ...] = ()
    tracking_issue: str | None = None


# ---------------------------------------------------------------------------
# Canonical Audit Table for Problematic Capabilities (#10512)
# ---------------------------------------------------------------------------

AUDITED_CAPABILITIES: dict[str, AuditEntry] = {
    "golf_simulation_suite": AuditEntry(
        capability_id="golf_simulation_suite",
        disposition=LaunchDisposition.PROTOTYPE_DEMO,
        label="Golf Simulation Suite (Prototype Demo)",
        explanation=(
            "Prototype visualization and demo inspection suite; not a qualified production "
            "physics solver. Built for preliminary inspection and UI testing."
        ),
        next_action=(
            "Use qualified production physics engines (MuJoCo, Drake, Pinocchio, OpenSim) "
            "for verified forward/inverse dynamic simulation."
        ),
        tracking_issue="#10512",
    ),
    "motion_capture": AuditEntry(
        capability_id="motion_capture",
        disposition=LaunchDisposition.PARAMETRIC_CLI,
        label="FreeMoCap Motion Capture (Parametric CLI)",
        explanation=(
            "FreeMoCap sidecar runner requires explicit --input video path and --output directory "
            "parameters; cannot launch headless with zero arguments."
        ),
        next_action=(
            "Provide input session videos and output directory via CLI or interactive modal (ORG-10)."
        ),
        required_parameters=("--input", "--output"),
        tracking_issue="#9478, #10512",
    ),
    "swing_optimizer": AuditEntry(
        capability_id="swing_optimizer",
        disposition=LaunchDisposition.LIBRARY_ONLY,
        label="Swing Optimizer (Library Only)",
        explanation=(
            "Multi-objective trajectory optimization algorithm is available as a library only "
            "(src/shared/python/optimization/); no interactive GUI has been built."
        ),
        next_action=(
            "Import and use programmatic API or await dedicated solver workflow form under ORG-16."
        ),
        tracking_issue="#10512, #8390",
    ),
    "injury_analysis": AuditEntry(
        capability_id="injury_analysis",
        disposition=LaunchDisposition.LIBRARY_ONLY,
        label="Injury Risk Analysis (Library Only)",
        explanation=(
            "Joint stress and spinal load scoring algorithms are available as a library only "
            "(src/shared/python/injury/); no interactive GUI has been built."
        ),
        next_action=(
            "Use programmatic scoring API or await dedicated biomechanics analysis form under ORG-17."
        ),
        tracking_issue="#10512",
    ),
    "canonical_core_estimation": AuditEntry(
        capability_id="canonical_core_estimation",
        disposition=LaunchDisposition.SERVICE_PREVIEW,
        label="Canonical-Core Estimation (Service Boundary Preview)",
        explanation=(
            "Estimation services are accessed via web/React route (/tools/canonical-core/estimation) "
            "and core service APIs; desktop shell is an inspection surface."
        ),
        next_action="Launch web catalog / React surface for estimation services.",
        tracking_issue="#10512",
    ),
    "canonical_core_comparison": AuditEntry(
        capability_id="canonical_core_comparison",
        disposition=LaunchDisposition.SERVICE_PREVIEW,
        label="Canonical-Core Comparison (Service Boundary Preview)",
        explanation=(
            "Comparison services are accessed via web/React route (/tools/canonical-core/comparison) "
            "and core service APIs; desktop shell is an inspection surface."
        ),
        next_action="Launch web catalog / React surface for comparison services.",
        tracking_issue="#10512",
    ),
    "pinn_pure_rigid": AuditEntry(
        capability_id="pinn_pure_rigid",
        disposition=LaunchDisposition.LIBRARY_ONLY,
        label="Physics-Informed Pure Rigid (Library Only)",
        explanation=(
            "Pinocchio inverse dynamics PINN mode is a library-only component "
            "(src/shared/python/physics_informed/); no interactive UI has been built."
        ),
        next_action="Import PhysicsMode.PURE_RIGID from library; track UI under #7984 and epic #5419.",
        tracking_issue="#7984, #5419",
    ),
    "pinn_hybrid": AuditEntry(
        capability_id="pinn_hybrid",
        disposition=LaunchDisposition.LIBRARY_ONLY,
        label="PINN Hybrid (Library Only)",
        explanation=(
            "Rigid body + JAX residual torque MLP is a library-only component "
            "(src/shared/python/physics_informed/); no interactive UI has been built."
        ),
        next_action="Import PhysicsMode.PINN_HYBRID from library; track UI under #7984 and epic #5419.",
        tracking_issue="#7984, #5419",
    ),
    "video_analyzer": AuditEntry(
        capability_id="video_analyzer",
        disposition=LaunchDisposition.PRODUCTION_SOLVER,
        label="Video Analyzer",
        explanation=(
            "File picker -> SwingAnalyzer.analyze_video() -> head-stability report, all "
            "implemented in this repo (src/tools/video_analyzer/). No external Tools "
            "repository provider is needed any more (issue #8883 replaced it). MediaPipe "
            "is an optional runtime dependency; if it is not installed, Analyze surfaces "
            "an explicit error in the report pane rather than a blank or placeholder window."
        ),
        next_action=None,
        tracking_issue="#8883",
    ),
}

AUDITED_CAPABILITY_IDS: tuple[str, ...] = tuple(AUDITED_CAPABILITIES.keys())


def get_launch_truthfulness_audit(capability_id: str) -> AuditEntry | None:
    """Retrieve the launch truthfulness audit entry for a capability, if audited."""
    return AUDITED_CAPABILITIES.get(capability_id)


def audit_capability_launch(capability_id: str) -> AuditEntry:
    """Retrieve audit entry or default production solver entry for unaudited capabilities."""
    entry = get_launch_truthfulness_audit(capability_id)
    if entry is not None:
        return entry
    return AuditEntry(
        capability_id=capability_id,
        disposition=LaunchDisposition.PRODUCTION_SOLVER,
        label=capability_id.replace("_", " ").title(),
        explanation="Production capability with standard launch lifecycle.",
    )


# ---------------------------------------------------------------------------
# FreeMoCap Pre-Flight Parameter Validation Seam
# ---------------------------------------------------------------------------


def launch_freemocap_with_validation(
    input_path: str | Path | None,
    output_path: str | Path | None,
    repo_path: Path,
    process_manager: Any,
) -> bool:
    """Validate parameters before scheduling FreeMoCap sidecar execution.

    Design by Contract:
    - Precondition: Both input_path and output_path must be non-empty strings/Paths.
    - Postcondition: If valid, schedules subprocess with exact CLI parameters;
                     if invalid or cancelled, performs NO subprocess spawn.
    """
    if repo_path is None:
        raise ValueError("repo_path must be provided")

    input_str = str(input_path).strip() if input_path is not None else ""
    output_str = str(output_path).strip() if output_path is not None else ""

    if not input_str or not output_str:
        logger.warning(
            "FreeMoCap launch rejected: both --input and --output are required. "
            "Zero-argument or cancelled invocation performs no subprocess spawn."
        )
        return False

    script_path = repo_path / "src" / "tools" / "freemocap_sidecar" / "run_freemocap.py"
    cli_args = ["--input", input_str, "--output", output_str]

    if hasattr(process_manager, "launch_script_with_args"):
        process = process_manager.launch_script_with_args(
            name="FreeMoCap Sidecar",
            script_path=script_path,
            args=cli_args,
            cwd=repo_path,
            keep_terminal_open=True,
        )
    elif hasattr(process_manager, "launch_script"):
        process = process_manager.launch_script(
            name="FreeMoCap Sidecar",
            script_path=script_path,
            cwd=repo_path,
            args=cli_args,
            keep_terminal_open=True,
        )
    else:
        logger.error("Process manager has no launch_script method")
        return False

    return process is not None
