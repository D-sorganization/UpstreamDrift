from __future__ import annotations

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_diagnostic_contract import (
    DiagnosticCode,
    DiagnosticSoftwareAdmission,
    ExecutionEffect,
    NativeDiagnosticFinding,
    NativeDiagnosticPolicy,
    PhysicalQualification,
    RuntimePhase,
    RuntimeViability,
    QualificationEffect,
    assess_native_diagnostic,
)

pytestmark = pytest.mark.unit


SOURCE = "a" * 64
REQUEST = "b" * 64


def _finding(
    *,
    code: DiagnosticCode = DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE,
    execution: ExecutionEffect = ExecutionEffect.REQUIRES_OPT_IN,
    qualification: QualificationEffect = QualificationEffect.UNQUALIFIED,
) -> NativeDiagnosticFinding:
    return NativeDiagnosticFinding(
        code=code,
        source_sha256=SOURCE,
        component_path="/forceset/muscle/geometrypath",
        evidence_sha256="c" * 64,
        execution_effect=execution,
        qualification_effect=qualification,
    )


def test_qualification_finding_is_independent_and_requires_source_bound_opt_in() -> (
    None
):
    finding = _finding()
    strict = assess_native_diagnostic(REQUEST, SOURCE, (finding,))

    assert strict.software_admission is DiagnosticSoftwareAdmission.ADMITTED
    assert strict.runtime_viability is RuntimeViability.NOT_ATTEMPTED
    assert strict.physical_qualification is PhysicalQualification.UNQUALIFIED
    assert not strict.execution_permitted

    policy = NativeDiagnosticPolicy(
        source_sha256=SOURCE,
        allowed_unqualified_findings=(finding.code,),
        provenance="reviewed diagnostic assay",
    )
    opted_in = assess_native_diagnostic(
        REQUEST,
        SOURCE,
        (finding,),
        policy=policy,
        runtime_viability=RuntimeViability.VIABLE,
        runtime_phase=RuntimePhase.SHORT_CAPABILITY_PROBE,
        native_runtime_sha256="e" * 64,
        runtime_evidence_sha256="d" * 64,
    )

    assert opted_in.software_admission is DiagnosticSoftwareAdmission.ADMITTED
    assert opted_in.runtime_viability is RuntimeViability.VIABLE
    assert opted_in.execution_permitted
    assert opted_in.physical_qualification is PhysicalQualification.UNQUALIFIED


@pytest.mark.parametrize(
    "findings",
    [
        (),
        (
            NativeDiagnosticFinding(
                code=DiagnosticCode.WRAPPED_PATH_SOURCE_CLOSED,
                source_sha256=SOURCE,
                execution_effect=ExecutionEffect.PROCEED,
                qualification_effect=QualificationEffect.OBSERVATION,
            ),
        ),
    ],
)
def test_empty_or_observation_only_assessment_never_establishes_qualification(
    findings: tuple[NativeDiagnosticFinding, ...],
) -> None:
    assessment = assess_native_diagnostic(REQUEST, SOURCE, findings)

    assert assessment.physical_qualification is PhysicalQualification.NOT_ESTABLISHED
    assert not assessment.execution_permitted


def test_explicit_opt_in_never_waives_execution_blockers() -> None:
    blocked = _finding(
        code=DiagnosticCode.PHYSICAL_BODY_INERTIA_INVALID,
        execution=ExecutionEffect.BLOCK,
        qualification=QualificationEffect.UNQUALIFIED,
    )
    policy = NativeDiagnosticPolicy(
        source_sha256=SOURCE,
        allowed_unqualified_findings=(DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE,),
        provenance="reviewed diagnostic assay",
    )
    assessment = assess_native_diagnostic(
        REQUEST,
        SOURCE,
        (blocked,),
        policy=policy,
        runtime_viability=RuntimeViability.VIABLE,
        runtime_phase=RuntimePhase.SHORT_CAPABILITY_PROBE,
        native_runtime_sha256="e" * 64,
        runtime_evidence_sha256="d" * 64,
    )

    assert assessment.software_admission is DiagnosticSoftwareAdmission.BLOCKED
    assert not assessment.execution_permitted


def test_source_binding_is_required_for_policy_and_each_finding() -> None:
    policy = NativeDiagnosticPolicy(
        source_sha256="e" * 64,
        allowed_unqualified_findings=(DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE,),
        provenance="reviewed diagnostic assay",
    )
    with pytest.raises(ValueError, match="policy source"):
        assess_native_diagnostic(REQUEST, SOURCE, (), policy=policy)

    with pytest.raises(ValueError, match="finding source"):
        assess_native_diagnostic(
            REQUEST,
            SOURCE,
            (
                NativeDiagnosticFinding(
                    code=DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE,
                    source_sha256="e" * 64,
                    component_path="/forceset/muscle/geometrypath",
                    evidence_sha256="c" * 64,
                    execution_effect=ExecutionEffect.REQUIRES_OPT_IN,
                    qualification_effect=QualificationEffect.UNQUALIFIED,
                ),
            ),
        )


def test_policy_cannot_opt_in_to_a_blocking_finding() -> None:
    with pytest.raises(ValueError, match="qualification-only"):
        NativeDiagnosticPolicy(
            source_sha256=SOURCE,
            allowed_unqualified_findings=(
                DiagnosticCode.PHYSICAL_BODY_INERTIA_INVALID,
            ),
            provenance="reviewed diagnostic assay",
        )


def test_nonviable_or_unattempted_native_trajectory_cannot_run_optimizer() -> None:
    finding = _finding()
    policy = NativeDiagnosticPolicy(
        source_sha256=SOURCE,
        allowed_unqualified_findings=(finding.code,),
        provenance="reviewed diagnostic assay",
    )

    for viability in (RuntimeViability.NOT_ATTEMPTED, RuntimeViability.FAILED):
        assessment = assess_native_diagnostic(
            REQUEST,
            SOURCE,
            (finding,),
            policy=policy,
            runtime_viability=viability,
            runtime_phase=(
                None
                if viability is RuntimeViability.NOT_ATTEMPTED
                else RuntimePhase.SHORT_CAPABILITY_PROBE
            ),
            native_runtime_sha256=(
                None if viability is RuntimeViability.NOT_ATTEMPTED else "e" * 64
            ),
            runtime_evidence_sha256=(
                "d" * 64 if viability is RuntimeViability.FAILED else None
            ),
        )
        assert not assessment.execution_permitted


def test_viable_runtime_requires_phase_and_both_exact_evidence_identities() -> None:
    with pytest.raises(ValueError, match="native runtime identity"):
        assess_native_diagnostic(
            REQUEST,
            SOURCE,
            (),
            runtime_viability=RuntimeViability.VIABLE,
            runtime_phase=RuntimePhase.SHORT_CAPABILITY_PROBE,
            runtime_evidence_sha256="d" * 64,
        )

    completed_optimizer = assess_native_diagnostic(
        REQUEST,
        SOURCE,
        (),
        runtime_viability=RuntimeViability.VIABLE,
        runtime_phase=RuntimePhase.OPTIMIZATION,
        native_runtime_sha256="e" * 64,
        runtime_evidence_sha256="d" * 64,
    )
    assert not completed_optimizer.execution_permitted

    with pytest.raises(ValueError, match="explicit phase"):
        assess_native_diagnostic(
            REQUEST,
            SOURCE,
            (),
            runtime_viability=RuntimeViability.VIABLE,
            native_runtime_sha256="e" * 64,
            runtime_evidence_sha256="d" * 64,
        )
