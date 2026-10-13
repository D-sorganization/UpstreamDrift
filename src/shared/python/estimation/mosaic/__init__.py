"""MOSAIC: Model-aware, Observation-consistent, Simultaneous Anthropometry,
Inertia and Control estimation.

Engine-agnostic kernels for fitting a forward-dynamics model to measured
kinematics across many trials of one subject.  The kernels consume batched
inertial regressors ``Y(q, v, a)``, actuation and constraint maps, and
observation Jacobians; a planar analytic chain is provided as the reference
fixture.  See ``docs/research/model_aware_matching/`` for the methods.
"""

from __future__ import annotations

from src.shared.python.estimation.mosaic.benchmark_protocol import (
    BenchmarkProtocol,
    FixtureConfig,
    ToleranceConfig,
    load_benchmark_protocol,
    verify_protocol_integrity,
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
    "ConformanceReport",
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
    "evaluate_qualification_gates",
    "load_benchmark_protocol",
    "run_ik_smoothing_baseline",
    "verify_acceleration_noise_without_input_smoothness",
    "verify_massless_collapse_under_resolved_basis",
    "verify_protocol_integrity",
    "verify_scale_collapse_without_anchor",
]
