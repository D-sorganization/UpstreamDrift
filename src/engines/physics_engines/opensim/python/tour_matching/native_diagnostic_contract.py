"""Source-bound findings for native diagnostic execution, never qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
import math
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ExecutionEffect(StrEnum):
    """Whether a finding blocks execution or needs explicit diagnostic opt-in."""

    BLOCK = "block"
    PROCEED = "proceed"
    REQUIRES_OPT_IN = "requires-opt-in"


class QualificationEffect(StrEnum):
    """Unqualified evidence can never be promoted by a successful numerical run."""

    NONE = "none"
    OBSERVATION = "observation"
    UNQUALIFIED = "unqualified"


class DiagnosticCode(StrEnum):
    """Stable machine-readable outcomes from the native diagnostic audits."""

    PASSIVE_POLICY_UNAVAILABLE = "passive_policy_unavailable"
    GUIDE_WORK_BUDGET_UNAVAILABLE = "guide_work_budget_unavailable"
    GLOBAL_CHART_PROOF_UNAVAILABLE = "global_chart_proof_unavailable"
    WRAPPED_PATH_DECLARATION_REQUIRED = "wrapped_path_declaration_required"
    WRAPPED_PATH_SOURCE_CLOSURE_INVALID = "wrapped_path_source_closure_invalid"
    WRAPPED_PATH_SOURCE_CLOSED = "wrapped_path_source_closed"
    WRAPPED_PATH_NATIVE_ROUTE_OBSERVED = "wrapped_path_native_route_observed"
    WRAPPED_PATH_NATIVE_ROUTE_INVALID = "wrapped_path_native_route_invalid"
    WRAPPED_PATH_TRANSITION_UNQUALIFIED = "wrapped_path_transition_unqualified"
    PHYSICAL_BODY_INERTIA_INVALID = "physical_body_inertia_invalid"
    NATIVE_SOURCE_IDENTITY_INVALID = "native_source_identity_invalid"
    NATIVE_RUNTIME_IDENTITY_UNAVAILABLE = "native_runtime_identity_unavailable"
    NATIVE_MODEL_LOAD_FAILED = "native_model_load_failed"
    NATIVE_STATE_RECONSTRUCTION_FAILED = "native_state_reconstruction_failed"
    NATIVE_STATE_COVERAGE_INVALID = "native_state_coverage_invalid"
    NATIVE_STATE_NONFINITE = "native_state_nonfinite"
    NATIVE_CONSTRAINT_POLICY_UNAVAILABLE = "native_constraint_policy_unavailable"
    NATIVE_CONSTRAINT_REALIZATION_FAILED = "native_constraint_realization_failed"
    NATIVE_MARKER_BINDING_INVALID = "native_marker_binding_invalid"
    NATIVE_INPUT_COVERAGE_INVALID = "native_input_coverage_invalid"
    NATIVE_FUNCTION_EVALUATION_FAILED = "native_function_evaluation_failed"
    NATIVE_JOINT_EVALUATION_FAILED = "native_joint_evaluation_failed"
    NATIVE_SHORT_PROBE_FAILED = "native_short_probe_failed"
    NATIVE_SHORT_PROBE_NONFINITE = "native_short_probe_nonfinite"
    NATIVE_SOURCE_CONTROLLER_UNDECLARED = "native_source_controller_undeclared"
    NATIVE_EXTERNAL_FORCE_UNINVENTORIED = "native_external_force_uninventoried"
    NATIVE_EXTERNAL_FORCE_UNREPRODUCED = "native_external_force_unreproduced"
    NATIVE_PASSIVE_POLICY_INVALID = "native_passive_policy_invalid"
    NATIVE_PASSIVE_LIMIT_EXCEEDED = "native_passive_limit_exceeded"
    NATIVE_REPLAY_CAPABILITY_UNAVAILABLE = "native_replay_capability_unavailable"
    NATIVE_OBSERVATION_INVALID = "native_observation_invalid"
    NATIVE_CHART_PROOF_UNAVAILABLE = "native_chart_proof_unavailable"
    NATIVE_PATH_OPTIMIZER_DOMAIN_INVALID = "native_path_optimizer_domain_invalid"
    NATIVE_SOURCE_IDENTITY_VERIFIED = "native_source_identity_verified"
    NATIVE_STATE_RECONSTRUCTED = "native_state_reconstructed"
    NATIVE_CONSTRAINTS_RECONSTRUCTED = "native_constraints_reconstructed"
    NATIVE_INPUT_BINDINGS_VERIFIED = "native_input_bindings_verified"
    NATIVE_MARKER_BINDINGS_VERIFIED = "native_marker_bindings_verified"
    NATIVE_PHYSICAL_INERTIA_ADMITTED = "native_physical_inertia_admitted"
    NATIVE_MECHANICS_EVALUATED = "native_mechanics_evaluated"
    NATIVE_SHORT_PROBE_COMPLETED = "native_short_probe_completed"
    SOURCE_CONDITIONAL_ANATOMICAL_HYPOTHESIS = (
        "source_conditional_anatomical_hypothesis"
    )
    SUPPORT_CONTACT_GRIP_UNQUALIFIED = "support_contact_grip_unqualified"
    MARKER_REGISTRATION_PROVISIONAL = "marker_registration_provisional"


class DiagnosticSoftwareAdmission(StrEnum):
    """Input/software admission, independent from attempted runtime success."""

    ADMITTED = "admitted"
    BLOCKED = "blocked"


class RuntimeViability(StrEnum):
    """Outcome of one identified, bounded native execution probe/trajectory."""

    NOT_ATTEMPTED = "not-attempted"
    VIABLE = "viable"
    FAILED = "failed"


class RuntimePhase(StrEnum):
    """Scope of the exact trajectory whose runtime disposition is recorded."""

    SHORT_CAPABILITY_PROBE = "short-capability-probe"
    OPTIMIZATION = "optimization"
    FULL_REPLAY = "full-replay"


class PhysicalQualification(StrEnum):
    """The diagnostic contract is deliberately unable to certify physical fit."""

    NOT_ESTABLISHED = "not-established"
    UNQUALIFIED = "unqualified"


_EXPECTED_EFFECTS: dict[DiagnosticCode, tuple[ExecutionEffect, QualificationEffect]] = {
    DiagnosticCode.PASSIVE_POLICY_UNAVAILABLE: (
        ExecutionEffect.REQUIRES_OPT_IN,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE: (
        ExecutionEffect.REQUIRES_OPT_IN,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.GLOBAL_CHART_PROOF_UNAVAILABLE: (
        ExecutionEffect.REQUIRES_OPT_IN,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.WRAPPED_PATH_DECLARATION_REQUIRED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.WRAPPED_PATH_SOURCE_CLOSURE_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.WRAPPED_PATH_SOURCE_CLOSED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.WRAPPED_PATH_NATIVE_ROUTE_OBSERVED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.WRAPPED_PATH_NATIVE_ROUTE_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.WRAPPED_PATH_TRANSITION_UNQUALIFIED: (
        ExecutionEffect.REQUIRES_OPT_IN,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.PHYSICAL_BODY_INERTIA_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_RUNTIME_IDENTITY_UNAVAILABLE: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_MODEL_LOAD_FAILED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_STATE_RECONSTRUCTION_FAILED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_STATE_COVERAGE_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_STATE_NONFINITE: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_CONSTRAINT_POLICY_UNAVAILABLE: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_CONSTRAINT_REALIZATION_FAILED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_MARKER_BINDING_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_FUNCTION_EVALUATION_FAILED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_JOINT_EVALUATION_FAILED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_SHORT_PROBE_FAILED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_SHORT_PROBE_NONFINITE: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_SOURCE_CONTROLLER_UNDECLARED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_EXTERNAL_FORCE_UNINVENTORIED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_EXTERNAL_FORCE_UNREPRODUCED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_PASSIVE_POLICY_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.NATIVE_PASSIVE_LIMIT_EXCEEDED: (
        ExecutionEffect.BLOCK,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.NATIVE_REPLAY_CAPABILITY_UNAVAILABLE: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_OBSERVATION_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_CHART_PROOF_UNAVAILABLE: (
        ExecutionEffect.REQUIRES_OPT_IN,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.NATIVE_PATH_OPTIMIZER_DOMAIN_INVALID: (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    ),
    DiagnosticCode.NATIVE_SOURCE_IDENTITY_VERIFIED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.NATIVE_STATE_RECONSTRUCTED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.NATIVE_CONSTRAINTS_RECONSTRUCTED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.NATIVE_INPUT_BINDINGS_VERIFIED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.NATIVE_MARKER_BINDINGS_VERIFIED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.NATIVE_PHYSICAL_INERTIA_ADMITTED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.NATIVE_MECHANICS_EVALUATED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.NATIVE_SHORT_PROBE_COMPLETED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.OBSERVATION,
    ),
    DiagnosticCode.SOURCE_CONDITIONAL_ANATOMICAL_HYPOTHESIS: (
        ExecutionEffect.PROCEED,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.SUPPORT_CONTACT_GRIP_UNQUALIFIED: (
        ExecutionEffect.PROCEED,
        QualificationEffect.UNQUALIFIED,
    ),
    DiagnosticCode.MARKER_REGISTRATION_PROVISIONAL: (
        ExecutionEffect.PROCEED,
        QualificationEffect.UNQUALIFIED,
    ),
}

_REQUIRED_EXECUTION_AUDITS = frozenset(
    {
        DiagnosticCode.NATIVE_SOURCE_IDENTITY_VERIFIED,
        DiagnosticCode.NATIVE_STATE_RECONSTRUCTED,
        DiagnosticCode.NATIVE_CONSTRAINTS_RECONSTRUCTED,
        DiagnosticCode.NATIVE_INPUT_BINDINGS_VERIFIED,
        DiagnosticCode.NATIVE_MARKER_BINDINGS_VERIFIED,
        DiagnosticCode.NATIVE_PHYSICAL_INERTIA_ADMITTED,
        DiagnosticCode.NATIVE_MECHANICS_EVALUATED,
        DiagnosticCode.NATIVE_SHORT_PROBE_COMPLETED,
    }
)


def diagnostic_effects(
    code: DiagnosticCode,
) -> tuple[ExecutionEffect, QualificationEffect]:
    """Return the immutable registry disposition for a validated diagnostic code."""
    if not isinstance(code, DiagnosticCode):
        raise TypeError("diagnostic code must be a DiagnosticCode")
    return _EXPECTED_EFFECTS[code]


_OPT_IN_CODES = frozenset(
    code
    for code, (execution, _) in _EXPECTED_EFFECTS.items()
    if execution is ExecutionEffect.REQUIRES_OPT_IN
)


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True)
class NativeDiagnosticFinding:
    """A source-bound audit result with independent execution/qualification effects."""

    code: DiagnosticCode
    source_sha256: str
    component_path: str | None = None
    evidence_sha256: str | None = None
    execution_effect: ExecutionEffect = ExecutionEffect.BLOCK
    qualification_effect: QualificationEffect = QualificationEffect.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.code, DiagnosticCode):
            raise TypeError("diagnostic code must be a registered DiagnosticCode")
        _require_sha256(self.source_sha256, "finding source SHA-256")
        expected = _EXPECTED_EFFECTS[self.code]
        if (self.execution_effect, self.qualification_effect) != expected:
            raise ValueError("diagnostic effects differ from the registered code")
        if self.component_path is not None and (
            not isinstance(self.component_path, str)
            or not self.component_path.startswith("/")
        ):
            raise ValueError("diagnostic component path must be absolute")
        if self.evidence_sha256 is not None:
            _require_sha256(self.evidence_sha256, "finding evidence SHA-256")


@dataclass(frozen=True)
class NativeDiagnosticPolicy:
    """Explicit opt-in bound to one source and a finite set of qual-only findings."""

    source_sha256: str
    allowed_unqualified_findings: tuple[DiagnosticCode, ...]
    provenance: str

    def __post_init__(self) -> None:
        _require_sha256(self.source_sha256, "policy source SHA-256")
        if not isinstance(self.allowed_unqualified_findings, tuple):
            raise TypeError("allowed unqualified findings must be a tuple")
        if not self.allowed_unqualified_findings:
            raise ValueError("diagnostic opt-in must name its qualification findings")
        if len(set(self.allowed_unqualified_findings)) != len(
            self.allowed_unqualified_findings
        ):
            raise ValueError("diagnostic opt-in contains duplicate finding codes")
        if any(
            not isinstance(code, DiagnosticCode) or code not in _OPT_IN_CODES
            for code in self.allowed_unqualified_findings
        ):
            raise ValueError("policy may allow qualification-only findings only")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("diagnostic opt-in requires explicit provenance")

    @property
    def identity_sha256(self) -> str:
        payload = {
            "source_sha256": self.source_sha256,
            "allowed_unqualified_findings": sorted(
                code.value for code in self.allowed_unqualified_findings
            ),
            "provenance": self.provenance,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class NativeQualificationContext:
    """Optional source-specific physical gaps with exact evidence provenance."""

    source_sha256: str
    findings: tuple[NativeDiagnosticFinding, ...]
    provenance_references: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.source_sha256, "qualification context source SHA-256")
        if not isinstance(self.findings, tuple) or not self.findings:
            raise ValueError("qualification context requires typed findings")
        if (
            not isinstance(self.provenance_references, tuple)
            or not self.provenance_references
        ):
            raise ValueError("qualification context requires provenance references")
        provenance_by_evidence: dict[str, tuple[str, ...]] = {}
        for evidence_sha256, references in self.provenance_references:
            _require_sha256(evidence_sha256, "qualification evidence SHA-256")
            if (
                not isinstance(references, tuple)
                or not references
                or any(
                    not isinstance(reference, str) or not reference.strip()
                    for reference in references
                )
            ):
                raise ValueError(
                    "each qualification evidence digest requires provenance"
                )
            provenance_by_evidence[evidence_sha256] = references
        allowed = {
            DiagnosticCode.SOURCE_CONDITIONAL_ANATOMICAL_HYPOTHESIS,
            DiagnosticCode.SUPPORT_CONTACT_GRIP_UNQUALIFIED,
            DiagnosticCode.MARKER_REGISTRATION_PROVISIONAL,
        }
        for finding in self.findings:
            if not isinstance(finding, NativeDiagnosticFinding):
                raise TypeError("qualification context findings must be typed")
            if finding.source_sha256 != self.source_sha256:
                raise ValueError("qualification finding source differs from context")
            if finding.code not in allowed or finding.evidence_sha256 is None:
                raise ValueError(
                    "qualification context code or evidence is unsupported"
                )
            if finding.evidence_sha256 not in provenance_by_evidence:
                raise ValueError("qualification finding evidence lacks provenance")
            if _EXPECTED_EFFECTS[finding.code] != (
                finding.execution_effect,
                finding.qualification_effect,
            ):
                raise ValueError("qualification finding differs from registry effects")


@dataclass(frozen=True)
class NativeDiagnosticAssessment:
    """Three independent outcomes retained with the exact request and source."""

    request_sha256: str
    source_sha256: str
    findings: tuple[NativeDiagnosticFinding, ...]
    software_admission: DiagnosticSoftwareAdmission
    runtime_viability: RuntimeViability
    physical_qualification: PhysicalQualification
    runtime_phase: RuntimePhase | None = None
    runtime_window_seconds: tuple[float, float] | None = None
    allowed_unqualified_findings: tuple[DiagnosticCode, ...] = ()
    policy_sha256: str | None = None
    native_runtime_sha256: str | None = None
    runtime_evidence_sha256: str | None = None
    blocker_receipt_sha256: str | None = None
    qualification_provenance_references: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "request SHA-256")
        _require_sha256(self.source_sha256, "assessment source SHA-256")
        if not isinstance(self.runtime_viability, RuntimeViability):
            raise TypeError("runtime viability must be a RuntimeViability")
        if not isinstance(self.physical_qualification, PhysicalQualification):
            raise TypeError("physical qualification must be a PhysicalQualification")
        _validate_assessment_findings_and_provenance(self)
        _validate_assessment_runtime(self)
        _validate_assessment_dispositions(self)

    @property
    def execution_permitted(self) -> bool:
        """Permit optimization only after an identified short capability probe."""
        if self.software_admission is not DiagnosticSoftwareAdmission.ADMITTED:
            return False
        if self.runtime_viability is not RuntimeViability.VIABLE:
            return False
        if self.runtime_phase is not RuntimePhase.SHORT_CAPABILITY_PROBE:
            return False
        if not _REQUIRED_EXECUTION_AUDITS.issubset(
            {finding.code for finding in self.findings}
        ):
            return False
        allowed = set(self.allowed_unqualified_findings)
        return all(
            finding.execution_effect is not ExecutionEffect.REQUIRES_OPT_IN
            or finding.code in allowed
            for finding in self.findings
        )

    @property
    def explicit_opt_in(self) -> bool:
        return bool(self.allowed_unqualified_findings)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request_sha256": self.request_sha256,
            "source_sha256": self.source_sha256,
            "findings": [
                {
                    **asdict(finding),
                    "code": finding.code.value,
                    "execution_effect": finding.execution_effect.value,
                    "qualification_effect": finding.qualification_effect.value,
                }
                for finding in self.findings
            ],
            "software_admission": self.software_admission.value,
            "runtime_viability": self.runtime_viability.value,
            "runtime_phase": (
                None if self.runtime_phase is None else self.runtime_phase.value
            ),
            "runtime_window_seconds": self.runtime_window_seconds,
            "physical_qualification": self.physical_qualification.value,
            "explicit_opt_in": self.explicit_opt_in,
            "allowed_unqualified_findings": sorted(
                code.value for code in self.allowed_unqualified_findings
            ),
            "policy_sha256": self.policy_sha256,
            "execution_permitted": self.execution_permitted,
            "native_runtime_sha256": self.native_runtime_sha256,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "blocker_receipt_sha256": self.blocker_receipt_sha256,
            "qualification_provenance_references": [
                {"evidence_sha256": digest, "references": references}
                for digest, references in self.qualification_provenance_references
            ],
        }


@dataclass(frozen=True)
class NativeDiagnosticRuntimeEvidence:
    """One indivisible runtime attempt identity carried by an assessment."""

    viability: RuntimeViability = RuntimeViability.NOT_ATTEMPTED
    phase: RuntimePhase | None = None
    window_seconds: tuple[float, float] | None = None
    native_runtime_sha256: str | None = None
    evidence_sha256: str | None = None


def _validate_assessment_findings_and_provenance(
    assessment: NativeDiagnosticAssessment,
) -> None:
    if not isinstance(assessment.findings, tuple) or any(
        not isinstance(finding, NativeDiagnosticFinding)
        for finding in assessment.findings
    ):
        raise TypeError("assessment findings must be typed native findings")
    if any(f.source_sha256 != assessment.source_sha256 for f in assessment.findings):
        raise ValueError("diagnostic finding source differs from assessment source")
    allowed = assessment.allowed_unqualified_findings
    if not isinstance(allowed, tuple) or any(
        not isinstance(code, DiagnosticCode) or code not in _OPT_IN_CODES
        for code in allowed
    ):
        raise ValueError("assessment opt-in codes must be qualification-only")
    if len(set(allowed)) != len(allowed):
        raise ValueError("assessment opt-in contains duplicate finding codes")
    if bool(allowed) != (assessment.policy_sha256 is not None):
        raise ValueError("assessment opt-in codes and policy identity differ")
    if assessment.policy_sha256 is not None:
        _require_sha256(assessment.policy_sha256, "diagnostic policy SHA-256")
    if assessment.blocker_receipt_sha256 is not None:
        _require_sha256(assessment.blocker_receipt_sha256, "blocker receipt SHA-256")
    references = assessment.qualification_provenance_references
    if not isinstance(references, tuple):
        raise TypeError("qualification provenance references must be typed pairs")
    provenance = dict(references)
    for digest, values in references:
        _require_sha256(digest, "qualification provenance evidence SHA-256")
        if (
            not isinstance(values, tuple)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError("qualification provenance references must be nonblank")
    provenance_codes = {
        DiagnosticCode.SOURCE_CONDITIONAL_ANATOMICAL_HYPOTHESIS,
        DiagnosticCode.SUPPORT_CONTACT_GRIP_UNQUALIFIED,
        DiagnosticCode.MARKER_REGISTRATION_PROVISIONAL,
    }
    if any(
        finding.code in provenance_codes and finding.evidence_sha256 not in provenance
        for finding in assessment.findings
    ):
        raise ValueError("qualification finding is missing its provenance reference")


def _validate_assessment_runtime(assessment: NativeDiagnosticAssessment) -> None:
    if assessment.runtime_phase is not None and not isinstance(
        assessment.runtime_phase, RuntimePhase
    ):
        raise TypeError("runtime phase must be a RuntimePhase or unavailable")
    for name, digest in (
        ("native runtime SHA-256", assessment.native_runtime_sha256),
        ("runtime evidence SHA-256", assessment.runtime_evidence_sha256),
    ):
        if digest is not None:
            _require_sha256(digest, name)
    if assessment.runtime_viability is RuntimeViability.NOT_ATTEMPTED:
        if any(
            value is not None
            for value in (
                assessment.runtime_evidence_sha256,
                assessment.native_runtime_sha256,
                assessment.runtime_phase,
                assessment.runtime_window_seconds,
            )
        ):
            raise ValueError("unattempted runtime cannot carry execution evidence")
        return
    if assessment.runtime_evidence_sha256 is None:
        raise ValueError("attempted runtime requires source-bound probe evidence")
    if assessment.native_runtime_sha256 is None:
        raise ValueError("attempted runtime requires native runtime identity")
    if assessment.runtime_phase is None:
        raise ValueError("attempted runtime requires an explicit phase")
    window = assessment.runtime_window_seconds
    if window is None:
        if assessment.runtime_viability is not RuntimeViability.FAILED:
            raise ValueError("attempted runtime requires a valid observed time window")
        return
    if (
        not isinstance(window, tuple)
        or len(window) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            for value in window
        )
        or (
            assessment.runtime_viability is RuntimeViability.VIABLE
            and window[1] <= window[0]
        )
        or (
            assessment.runtime_viability is RuntimeViability.FAILED
            and window[1] < window[0]
        )
    ):
        raise ValueError("attempted runtime requires a valid observed time window")


def _validate_assessment_dispositions(
    assessment: NativeDiagnosticAssessment,
) -> None:
    expected_admission = (
        DiagnosticSoftwareAdmission.BLOCKED
        if any(f.execution_effect is ExecutionEffect.BLOCK for f in assessment.findings)
        else DiagnosticSoftwareAdmission.ADMITTED
    )
    if assessment.software_admission is not expected_admission:
        raise ValueError("software admission differs from typed findings")
    expected_qualification = (
        PhysicalQualification.UNQUALIFIED
        if any(
            finding.qualification_effect is QualificationEffect.UNQUALIFIED
            for finding in assessment.findings
        )
        else PhysicalQualification.NOT_ESTABLISHED
    )
    if assessment.physical_qualification is not expected_qualification:
        raise ValueError("diagnostics cannot promote physical qualification")


def _decode_assessment_findings(
    raw_findings: object,
) -> tuple[NativeDiagnosticFinding, ...]:
    if not isinstance(raw_findings, list):
        raise ValueError("serialized assessment findings must be an array")
    findings: list[NativeDiagnosticFinding] = []
    fields = {
        "code",
        "source_sha256",
        "component_path",
        "evidence_sha256",
        "execution_effect",
        "qualification_effect",
    }
    for raw in raw_findings:
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ValueError("serialized assessment finding fields are invalid")
        code = raw["code"]
        source = raw["source_sha256"]
        path = raw["component_path"]
        evidence = raw["evidence_sha256"]
        execution = raw["execution_effect"]
        qualification = raw["qualification_effect"]
        if (
            not isinstance(code, str)
            or not isinstance(source, str)
            or (path is not None and not isinstance(path, str))
            or not isinstance(evidence, str)
            or not isinstance(execution, str)
            or not isinstance(qualification, str)
        ):
            raise ValueError("serialized assessment finding types are invalid")
        findings.append(
            NativeDiagnosticFinding(
                DiagnosticCode(code),
                source,
                path,
                evidence,
                ExecutionEffect(execution),
                QualificationEffect(qualification),
            )
        )
    return tuple(findings)


def _decode_qualification_provenance(
    raw_refs: object,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(raw_refs, list):
        raise ValueError("serialized qualification provenance must be an array")
    items: list[tuple[str, tuple[str, ...]]] = []
    for item in raw_refs:
        if not isinstance(item, dict) or set(item) != {"evidence_sha256", "references"}:
            raise ValueError("serialized qualification provenance entry is malformed")
        digest, references = item["evidence_sha256"], item["references"]
        if (
            not isinstance(digest, str)
            or not isinstance(references, list)
            or any(not isinstance(reference, str) for reference in references)
        ):
            raise ValueError("serialized qualification provenance types are invalid")
        items.append((digest, tuple(references)))
    return tuple(items)


def _decode_runtime_window(raw_window: object) -> tuple[float, float] | None:
    if raw_window is None:
        return None
    if (
        not isinstance(raw_window, list)
        or len(raw_window) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_window
        )
    ):
        raise ValueError("serialized assessment runtime window is invalid")
    return float(raw_window[0]), float(raw_window[1])


def native_diagnostic_assessment_from_json_dict(
    payload: dict[str, object],
) -> NativeDiagnosticAssessment:
    """Revalidate a serialized assessment instead of trusting receipt JSON."""
    required = {
        "request_sha256",
        "source_sha256",
        "findings",
        "software_admission",
        "runtime_viability",
        "runtime_phase",
        "runtime_window_seconds",
        "physical_qualification",
        "allowed_unqualified_findings",
        "policy_sha256",
        "native_runtime_sha256",
        "runtime_evidence_sha256",
        "blocker_receipt_sha256",
        "qualification_provenance_references",
    }
    if not isinstance(payload, dict) or set(payload) != required | {
        "explicit_opt_in",
        "execution_permitted",
    }:
        raise ValueError(
            "serialized native assessment fields are incomplete or unknown"
        )

    def required_string(name: str) -> str:
        value = payload[name]
        if not isinstance(value, str):
            raise ValueError(f"serialized assessment {name} must be a string")
        return value

    def optional_string(name: str) -> str | None:
        value = payload[name]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"serialized assessment {name} must be a string")
        return value

    request_sha256 = required_string("request_sha256")
    source_sha256 = required_string("source_sha256")
    admission = required_string("software_admission")
    viability = required_string("runtime_viability")
    qualification = required_string("physical_qualification")
    runtime_phase = payload["runtime_phase"]
    if runtime_phase is not None and not isinstance(runtime_phase, str):
        raise ValueError("serialized assessment runtime phase must be a string")
    policy_sha256 = optional_string("policy_sha256")
    native_runtime_sha256 = optional_string("native_runtime_sha256")
    runtime_evidence_sha256 = optional_string("runtime_evidence_sha256")
    blocker_receipt_sha256 = optional_string("blocker_receipt_sha256")
    findings = _decode_assessment_findings(payload["findings"])
    provenance = _decode_qualification_provenance(
        payload["qualification_provenance_references"]
    )
    runtime_window = _decode_runtime_window(payload["runtime_window_seconds"])
    allowed = payload["allowed_unqualified_findings"]
    if not isinstance(allowed, list) or any(
        not isinstance(code, str) for code in allowed
    ):
        raise ValueError("serialized assessment opt-in codes must be strings")
    allowed_codes = tuple(DiagnosticCode(code) for code in allowed)
    assessment = NativeDiagnosticAssessment(
        request_sha256=request_sha256,
        source_sha256=source_sha256,
        findings=findings,
        software_admission=DiagnosticSoftwareAdmission(admission),
        runtime_viability=RuntimeViability(viability),
        physical_qualification=PhysicalQualification(qualification),
        runtime_phase=None if runtime_phase is None else RuntimePhase(runtime_phase),
        runtime_window_seconds=runtime_window,
        allowed_unqualified_findings=allowed_codes,
        policy_sha256=policy_sha256,
        native_runtime_sha256=native_runtime_sha256,
        runtime_evidence_sha256=runtime_evidence_sha256,
        blocker_receipt_sha256=blocker_receipt_sha256,
        qualification_provenance_references=provenance,
    )
    if (
        payload["explicit_opt_in"] is not assessment.explicit_opt_in
        or payload["execution_permitted"] is not assessment.execution_permitted
    ):
        raise ValueError("serialized assessment dispositions differ from its findings")
    return assessment


def assess_native_diagnostic(
    request_sha256: str,
    source_sha256: str,
    findings: tuple[NativeDiagnosticFinding, ...],
    *,
    policy: NativeDiagnosticPolicy | None = None,
    runtime: NativeDiagnosticRuntimeEvidence = NativeDiagnosticRuntimeEvidence(),
    blocker_receipt_sha256: str | None = None,
    qualification_provenance_references: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> NativeDiagnosticAssessment:
    """Bind generated findings to a request, exact source and diagnostic policy."""
    _require_sha256(source_sha256, "assessment source SHA-256")
    if not isinstance(findings, tuple) or any(
        not isinstance(finding, NativeDiagnosticFinding) for finding in findings
    ):
        raise TypeError("findings must be a tuple of typed native diagnostic findings")
    if any(finding.source_sha256 != source_sha256 for finding in findings):
        raise ValueError("finding source differs from assessment source")
    if not isinstance(policy, NativeDiagnosticPolicy) and policy is not None:
        raise TypeError("policy must be a NativeDiagnosticPolicy or unavailable")
    if policy is not None and policy.source_sha256 != source_sha256:
        raise ValueError("diagnostic policy source differs from assessment source")
    if not isinstance(runtime, NativeDiagnosticRuntimeEvidence):
        raise TypeError("runtime must be NativeDiagnosticRuntimeEvidence")
    allowed = policy.allowed_unqualified_findings if policy else ()
    software_admission = (
        DiagnosticSoftwareAdmission.BLOCKED
        if any(f.execution_effect is ExecutionEffect.BLOCK for f in findings)
        else DiagnosticSoftwareAdmission.ADMITTED
    )
    physical_qualification = (
        PhysicalQualification.UNQUALIFIED
        if any(
            f.qualification_effect is QualificationEffect.UNQUALIFIED for f in findings
        )
        else PhysicalQualification.NOT_ESTABLISHED
    )
    return NativeDiagnosticAssessment(
        request_sha256=request_sha256,
        source_sha256=source_sha256,
        findings=findings,
        software_admission=software_admission,
        runtime_viability=runtime.viability,
        physical_qualification=physical_qualification,
        runtime_phase=runtime.phase,
        runtime_window_seconds=runtime.window_seconds,
        allowed_unqualified_findings=allowed,
        policy_sha256=None if policy is None else policy.identity_sha256,
        native_runtime_sha256=runtime.native_runtime_sha256,
        runtime_evidence_sha256=runtime.evidence_sha256,
        blocker_receipt_sha256=blocker_receipt_sha256,
        qualification_provenance_references=qualification_provenance_references,
    )


__all__ = [
    "DiagnosticCode",
    "DiagnosticSoftwareAdmission",
    "ExecutionEffect",
    "NativeDiagnosticAssessment",
    "NativeDiagnosticFinding",
    "NativeDiagnosticPolicy",
    "NativeDiagnosticRuntimeEvidence",
    "NativeQualificationContext",
    "PhysicalQualification",
    "QualificationEffect",
    "RuntimePhase",
    "RuntimeViability",
    "assess_native_diagnostic",
    "diagnostic_effects",
    "native_diagnostic_assessment_from_json_dict",
]
