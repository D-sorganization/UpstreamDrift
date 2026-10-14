"""MOSAIC: Model-aware, Observation-consistent, Simultaneous Anthropometry,
Inertia and Control estimation.

Engine-agnostic kernels for fitting a forward-dynamics model to measured
kinematics across many trials of one subject.  The kernels consume batched
inertial regressors ``Y(q, v, a)``, actuation and constraint maps, and
observation Jacobians; a planar analytic chain is provided as the reference
fixture.  See ``docs/research/model_aware_matching/`` for the methods.
"""

from __future__ import annotations

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

__all__ = [
    "ConeViolationReport",
    "ContactIdentifiabilityDiagnostics",
    "ContactWrenchProblem",
    "ContactWrenchSolution",
    "compute_cone_violations",
    "compute_identifiability_diagnostics",
    "project_friction_cone",
    "solve_contact_wrenches",
]
