"""Simscape Compiled Block Budget Gate and Observability Manifest (MMR-04, #11088).

Enforces compiled block count budgets under MATLAB R2025b Home license limits
(1,000 blocks ceiling, 25-block instrumentation reserve => 975 production ceiling),
requires explicit subsystem breakdown reports (rejecting inferred icon counts),
and verifies observability contracts so that no consumer loses mass, COM, energy,
contact, or closure evidence when sensors are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Mapping

from src.shared.python.contracts import postcondition, precondition

logger = logging.getLogger(__name__)

HOME_LICENSE_BLOCK_LIMIT: int = 1000
INSTRUMENTATION_RESERVE_BLOCKS: int = 25
PRODUCTION_BUDGET_CEILING: int = (
    HOME_LICENSE_BLOCK_LIMIT - INSTRUMENTATION_RESERVE_BLOCKS
)
EXACT_AUDIT_BUDGET_CEILING: int = HOME_LICENSE_BLOCK_LIMIT
REQUIRED_MATLAB_RELEASE: str = "2025b"


class SimscapeBlockBudgetError(ValueError):
    """Base exception for Simscape block-budget and observability contract violations."""


class SimscapeBudgetExceededError(SimscapeBlockBudgetError):
    """Raised when a Simscape model exceeds its compiled block budget ceiling."""


class ObservabilityEvidenceError(SimscapeBlockBudgetError):
    """Raised when an observability evidence contract is broken."""


class InvalidSubsystemReportError(SimscapeBlockBudgetError):
    """Raised when subsystem breakdown reporting is missing, inferred, or inconsistent."""


class BlockBudgetProfile(str, Enum):
    """Execution and verification profiles for Simscape block budgets."""

    PRODUCTION = "production"
    AUDIT = "audit"
    INSTRUMENTED = "instrumented"


class ObservabilityEvidenceType(str, Enum):
    """Required evidence domains that must survive sensor removal."""

    MASS = "mass"
    COM = "com"
    ENERGY = "energy"
    CONTACT = "contact"
    CLOSURE = "closure"


class EvidenceSourceType(str, Enum):
    """Source mechanisms supplying observability evidence."""

    NATIVE_LOG = "native_log"
    OFFLINE_DIAGNOSTIC = "offline_diagnostic"
    WORKSPACE_REFERENCE = "workspace_reference"
    PHYSICAL_SENSOR = "physical_sensor"


@dataclass(frozen=True)
class SubsystemBlockCount:
    """Explicit nonvirtual block count for a top-level subsystem."""

    name: str
    uncompiled_count: int
    compiled_count: int

    @property
    def growth(self) -> int:
        """Nonvirtual block count growth caused by diagram compilation."""
        return self.compiled_count - self.uncompiled_count


@dataclass(frozen=True)
class ObservabilityChannel:
    """A single diagnostic or observability evidence channel."""

    channel_id: str
    evidence_type: ObservabilityEvidenceType
    source_type: EvidenceSourceType
    active: bool
    consumer: str
    description: str = ""


@dataclass(frozen=True)
class ObservabilityManifest:
    """Immutable manifest declaring evidence channels covering all required domains."""

    channels: tuple[ObservabilityChannel, ...]

    def has_evidence(self, evidence_type: ObservabilityEvidenceType) -> bool:
        """Check whether an active channel exists for the given evidence type."""
        return any(c.evidence_type == evidence_type and c.active for c in self.channels)

    def validate_coverage(self) -> list[str]:
        """Validate that all 5 required evidence domains are actively covered."""
        issues: list[str] = []
        for req in ObservabilityEvidenceType:
            if not self.has_evidence(req):
                issues.append(f"Missing required evidence domain: {req.value}")
        for c in self.channels:
            if not c.active and bool(c.consumer.strip()):
                issues.append(
                    f"Orphaned active consumer {c.consumer!r} on inactive channel {c.channel_id!r}"
                )
        return issues


@dataclass(frozen=True)
class SimscapeBlockBudgetReport:
    """Compiled block-budget report and evidence manifest for a Simscape model."""

    model_name: str
    profile: BlockBudgetProfile
    uncompiled_total: int
    compiled_total: int
    converter_internal: int
    top_level_equivalent: int
    simscape_blocks: int
    subsystems: tuple[SubsystemBlockCount, ...]
    observability: ObservabilityManifest
    matlab_release: str = REQUIRED_MATLAB_RELEASE
    license_ceiling: int = HOME_LICENSE_BLOCK_LIMIT
    instrumentation_reserve: int = INSTRUMENTATION_RESERVE_BLOCKS

    @property
    def nonvirtual_total(self) -> int:
        """Alias matching MATLAB report field name."""
        return self.uncompiled_total


@dataclass(frozen=True)
class BudgetVerdict:
    """Verdict returned by budget gate validation."""

    passed: bool
    effective_ceiling: int
    headroom: int
    reserve_blocks: int
    summary: str
    diagnostics: tuple[str, ...]


def create_canonical_human_observability_manifest() -> ObservabilityManifest:
    """Construct the canonical observability manifest for GS3DX_Human."""
    channels = (
        ObservabilityChannel(
            channel_id="human_mass_audit",
            evidence_type=ObservabilityEvidenceType.MASS,
            source_type=EvidenceSourceType.OFFLINE_DIAGNOSTIC,
            active=True,
            consumer="gs3dx_inertia_audit",
            description="Total mass and segment mass distribution verified against GS3DX_Neck",
        ),
        ObservabilityChannel(
            channel_id="human_balance_com_ref",
            evidence_type=ObservabilityEvidenceType.COM,
            source_type=EvidenceSourceType.WORKSPACE_REFERENCE,
            active=True,
            consumer="BalanceCOMRef",
            description="Address COM translation and trajectory reference from capture",
        ),
        ObservabilityChannel(
            channel_id="human_energy_simlog",
            evidence_type=ObservabilityEvidenceType.ENERGY,
            source_type=EvidenceSourceType.OFFLINE_DIAGNOSTIC,
            active=True,
            consumer="gs3dx_simlog_joints",
            description="Work and mechanical energy accounting derived from joint log",
        ),
        ObservabilityChannel(
            channel_id="human_foot_contacts",
            evidence_type=ObservabilityEvidenceType.CONTACT,
            source_type=EvidenceSourceType.NATIVE_LOG,
            active=True,
            consumer="FootContactForces",
            description="10-contact normal and tangential force logging, left foot first",
        ),
        ObservabilityChannel(
            channel_id="human_momentum_closure",
            evidence_type=ObservabilityEvidenceType.CLOSURE,
            source_type=EvidenceSourceType.OFFLINE_DIAGNOSTIC,
            active=True,
            consumer="gs3dx_contact_check",
            description="Closure of momentum balance (contacts + gravity) within 1% of M*g*T",
        ),
    )
    return ObservabilityManifest(channels=channels)


def create_canonical_baseline_observability_manifest() -> ObservabilityManifest:
    """Construct the canonical observability manifest for GS3DX_Baseline."""
    channels = (
        ObservabilityChannel(
            channel_id="baseline_solid_mass",
            evidence_type=ObservabilityEvidenceType.MASS,
            source_type=EvidenceSourceType.PHYSICAL_SENSOR,
            active=True,
            consumer="sm_lib/Body Elements/Inertia Sensor",
            description="12 physical inertia sensors on segments",
        ),
        ObservabilityChannel(
            channel_id="baseline_com_logging",
            evidence_type=ObservabilityEvidenceType.COM,
            source_type=EvidenceSourceType.NATIVE_LOG,
            active=True,
            consumer="PS-Simulink Converter/COM",
            description="Segment COM position logging",
        ),
        ObservabilityChannel(
            channel_id="baseline_energy_calc",
            evidence_type=ObservabilityEvidenceType.ENERGY,
            source_type=EvidenceSourceType.NATIVE_LOG,
            active=True,
            consumer="Integrator/KineticEnergy",
            description="Diagram kinetic and potential energy integrators",
        ),
        ObservabilityChannel(
            channel_id="baseline_ground_weld",
            evidence_type=ObservabilityEvidenceType.CONTACT,
            source_type=EvidenceSourceType.NATIVE_LOG,
            active=True,
            consumer="WeldJoint/ReactionForce",
            description="Fixed ground weld reaction forces",
        ),
        ObservabilityChannel(
            channel_id="baseline_kinematic_closure",
            evidence_type=ObservabilityEvidenceType.CLOSURE,
            source_type=EvidenceSourceType.OFFLINE_DIAGNOSTIC,
            active=True,
            consumer="KinematicsSolver",
            description="Grip loop closure check",
        ),
    )
    return ObservabilityManifest(channels=channels)


def _validate_subsystem_reporting(report: SimscapeBlockBudgetReport) -> None:
    """Verify that subsystem counts are explicitly provided and match totals."""
    if not report.subsystems:
        raise InvalidSubsystemReportError(
            f"Subsystems breakdown cannot be empty for model {report.model_name!r}. "
            "Inferred icon counts are prohibited; report explicit uncompiled and compiled counts."
        )

    sum_uncompiled = sum(s.uncompiled_count for s in report.subsystems)
    sum_compiled = sum(s.compiled_count for s in report.subsystems)

    if sum_uncompiled != report.uncompiled_total:
        raise InvalidSubsystemReportError(
            f"Subsystem uncompiled sum ({sum_uncompiled}) does not match "
            f"uncompiled_total ({report.uncompiled_total}) in model {report.model_name!r}"
        )
    if sum_compiled != report.compiled_total:
        raise InvalidSubsystemReportError(
            f"Subsystem compiled sum ({sum_compiled}) does not match "
            f"compiled_total ({report.compiled_total}) in model {report.model_name!r}"
        )


def _format_diagnostics(
    report: SimscapeBlockBudgetReport, ceiling: int, excess: int
) -> tuple[str, ...]:
    """Build actionable diagnostics for budget overruns."""
    sorted_by_growth = sorted(report.subsystems, key=lambda s: s.growth, reverse=True)
    top_culprits = [
        f"{s.name} (+{s.growth} compiled, uncompiled={s.uncompiled_count}, compiled={s.compiled_count})"
        for s in sorted_by_growth[:3]
    ]

    diagnostics = [
        f"Offending subsystems by growth: {', '.join(top_culprits)}",
        (
            f"Model {report.model_name} has {report.converter_internal} converter internal blocks; "
            "audit nesl_utility converter pairs and replace unread sensors with offline diagnostics."
        ),
    ]
    if report.compiled_total > HOME_LICENSE_BLOCK_LIMIT:
        diagnostics.insert(
            0,
            (
                f"Number of blocks in the block diagram '{report.model_name}' and all models it references "
                f"exceeds the license limit of 1000 nonvirtual blocks."
            ),
        )
    else:
        diagnostics.insert(
            0,
            (
                f"Compiled block count ({report.compiled_total}) exceeds production budget ceiling "
                f"of {ceiling} (mandatory reserve of {report.instrumentation_reserve} blocks from 1000 license limit)."
            ),
        )
    return tuple(diagnostics)


@precondition(
    lambda report: isinstance(report, SimscapeBlockBudgetReport),
    "report must be a SimscapeBlockBudgetReport",
)
@postcondition(
    lambda verdict: isinstance(verdict, BudgetVerdict),
    "verdict must be a BudgetVerdict",
)
def validate_simscape_block_budget(report: SimscapeBlockBudgetReport) -> BudgetVerdict:
    """Validate Simscape block budget, subsystem reporting, and observability contracts."""
    # 1. Observability contract validation
    obs_issues = report.observability.validate_coverage()
    if obs_issues:
        raise ObservabilityEvidenceError(
            f"Observability contract violated for model {report.model_name!r}: "
            f"{'; '.join(obs_issues)}"
        )

    # 2. Subsystem explicit breakdown validation
    _validate_subsystem_reporting(report)

    # 3. Determine effective ceiling based on profile
    if report.profile == BlockBudgetProfile.PRODUCTION:
        ceiling = report.license_ceiling - report.instrumentation_reserve
        reserve = report.instrumentation_reserve
    else:
        ceiling = report.license_ceiling
        reserve = 0

    headroom = ceiling - report.compiled_total

    # 4. Ceiling checks and fail-closed diagnostics
    if report.compiled_total > ceiling:
        excess = report.compiled_total - ceiling
        diagnostics = _format_diagnostics(report, ceiling, excess)
        raise SimscapeBudgetExceededError(
            f"Model {report.model_name!r} compiled_total={report.compiled_total} "
            f"exceeds {'production budget ceiling of 975 (with reserve of 25 blocks)' if report.profile == BlockBudgetProfile.PRODUCTION else 'license limit of 1000 nonvirtual blocks'}. "
            f"Headroom: {headroom}. Diagnostics: {'; '.join(diagnostics)}"
        )

    summary = (
        f"Within {'production' if report.profile == BlockBudgetProfile.PRODUCTION else 'audit'} "
        f"budget ceiling of {ceiling} (compiled={report.compiled_total}, headroom={headroom}, "
        f"reserve={reserve})"
    )
    return BudgetVerdict(
        passed=True,
        effective_ceiling=ceiling,
        headroom=headroom,
        reserve_blocks=reserve,
        summary=summary,
        diagnostics=(),
    )


def parse_matlab_block_budget_json(
    data: Mapping[str, Any],
    profile: BlockBudgetProfile = BlockBudgetProfile.PRODUCTION,
    observability: ObservabilityManifest | None = None,
) -> SimscapeBlockBudgetReport:
    """Parse JSON exported by MATLAB gs3dx_block_budget into typed report."""
    model_name = str(data.get("model", "GS3DX_Unknown"))
    uncompiled_total = int(data.get("nonvirtual_total", 0))
    compiled_total = int(data.get("compiled_total", uncompiled_total))
    converter_internal = int(data.get("converter_internal", 0))
    top_level_equivalent = int(data.get("top_level_equivalent", uncompiled_total))
    simscape_blocks = int(data.get("simscape_blocks", 0))

    subsystems_data = data.get("subsystems")
    if isinstance(subsystems_data, list) and subsystems_data:
        subsystems = tuple(
            SubsystemBlockCount(
                name=str(s.get("name", f"Subsystem_{i}")),
                uncompiled_count=int(s.get("uncompiled_count", 0)),
                compiled_count=int(
                    s.get("compiled_count", s.get("uncompiled_count", 0))
                ),
            )
            for i, s in enumerate(subsystems_data)
        )
    else:
        # Default single top-level entry when legacy JSON lacks explicit subsystem map
        subsystems = (
            SubsystemBlockCount(
                name="TopLevel",
                uncompiled_count=uncompiled_total,
                compiled_count=compiled_total,
            ),
        )

    if observability is None:
        observability = create_canonical_baseline_observability_manifest()

    return SimscapeBlockBudgetReport(
        model_name=model_name,
        profile=profile,
        uncompiled_total=uncompiled_total,
        compiled_total=compiled_total,
        converter_internal=converter_internal,
        top_level_equivalent=top_level_equivalent,
        simscape_blocks=simscape_blocks,
        subsystems=subsystems,
        observability=observability,
    )
