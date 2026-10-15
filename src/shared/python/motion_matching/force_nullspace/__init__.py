"""Force-only alternatives on an explicitly supplied dynamics equality (#10436, #12552).

The caller owns the native equation A x = b, coordinate order, units, contact
schedule and constitutive constraints. No root slack or motion correction is
introduced here. Feasibility is numerical evidence, not native replay approval.
"""

from __future__ import annotations

from scipy.optimize import minimize

from src.shared.python.motion_matching.force_nullspace._common import Array, _array
from src.shared.python.motion_matching.force_nullspace.constraints import (
    ForceConstraints,
    FrictionCone,
)
from src.shared.python.motion_matching.force_nullspace.redistribution import (
    RedistributionResult,
    _finalize_redistribution,
    _is_solution_feasible,
    _margin_violation,
    redistribute_forces,
    redistribute_trajectory,
)
from src.shared.python.motion_matching.force_nullspace.space import (
    ForceNullSpace,
    NullSpaceAnalysis,
)
from src.shared.python.motion_matching.force_nullspace.tradeoffs import (
    TorqueDistributionTradeoff,
    TradeoffAlternative,
    explore_torque_tradeoffs,
)

__all__ = [
    "ForceConstraints",
    "ForceNullSpace",
    "FrictionCone",
    "NullSpaceAnalysis",
    "RedistributionResult",
    "TorqueDistributionTradeoff",
    "TradeoffAlternative",
    "explore_torque_tradeoffs",
    "redistribute_forces",
    "redistribute_trajectory",
]
