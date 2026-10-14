"""MOSAIC: Model-Aware, Multi-Trial Forward-Dynamics Matching.

This package provides forward-dynamics matching on consistent inertial manifolds
with variable-projection inverse-dynamics collocation and planar oracle
fixture.  See ``docs/research/model_aware_matching/model_aware_matching.tex``.
"""

from __future__ import annotations

from src.shared.python.estimation.mosaic.benchmark_protocol import (
    BenchmarkProtocol,
    FixtureConfig,
    ToleranceConfig,
    load_benchmark_protocol,
    verify_protocol_integrity,
)
from src.shared.python.estimation.mosaic.contact_wrenches import (
    ConeViolationReport,
    ContactIdentifiabilityDiagnostics,
    ContactWrenchProblem,
    ContactWrenchSolution,
    compute_cone_violations,
    compute_identifiability_diagnostics,
    project_friction_cone,
    solve_contact_wrenches,
)
from src.shared.python.estimation.mosaic.ik_baseline import (
    IKBaselineResult,
    run_ik_smoothing_baseline,
)
from src.shared.python.estimation.mosaic.negative_results import (
    AccelerationNoiseReport,
    MasslessCollapseReport,
    ScaleCollapseReport,
    verify_acceleration_noise_without_input_smoothness,
    verify_massless_collapse_under_resolved_basis,
    verify_scale_collapse_without_anchor,
)
from src.shared.python.estimation.mosaic.planar_oracle import (
    ConformanceReport,
    PlanarConformanceOracle,
)
from src.shared.python.estimation.mosaic.qualification_gates import (
    GateName,
    GateResult,
    GateStatus,
    check_closed_loop_effort,
    check_cross_trial_consistency,
    check_declared_discrepancy,
    check_kinematic_whiteness,
    check_observability,
    check_open_loop_replay,
    check_physical_consistency,
    evaluate_qualification_gates,
)

__all__ = [
    "AccelerationNoiseReport",
    "BenchmarkProtocol",
    "ConeViolationReport",
    "ConformanceReport",
    "ContactIdentifiabilityDiagnostics",
    "ContactWrenchProblem",
    "ContactWrenchSolution",
    "FixtureConfig",
    "GateName",
    "GateResult",
    "GateStatus",
    "IKBaselineResult",
    "MasslessCollapseReport",
    "PlanarConformanceOracle",
    "ScaleCollapseReport",
    "ToleranceConfig",
    "check_closed_loop_effort",
    "check_cross_trial_consistency",
    "check_declared_discrepancy",
    "check_kinematic_whiteness",
    "check_observability",
    "check_open_loop_replay",
    "check_physical_consistency",
    "compute_cone_violations",
    "compute_identifiability_diagnostics",
    "evaluate_qualification_gates",
    "load_benchmark_protocol",
    "project_friction_cone",
    "run_ik_smoothing_baseline",
    "solve_contact_wrenches",
    "verify_acceleration_noise_without_input_smoothness",
    "verify_massless_collapse_under_resolved_basis",
    "verify_protocol_integrity",
    "verify_scale_collapse_without_anchor",
]
