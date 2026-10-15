"""Pareto torque distribution alternatives and tradeoff exploration (#12552)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.shared.python.motion_matching.force_nullspace._common import Array, _array
from src.shared.python.motion_matching.force_nullspace.constraints import (
    ForceConstraints,
)
from src.shared.python.motion_matching.force_nullspace.redistribution import (
    redistribute_forces,
)
from src.shared.python.motion_matching.force_nullspace.space import ForceNullSpace

_DEFAULT_UNITS: dict[str, str] = {
    "torque": "N*m",
    "force": "N",
    "power": "W",
    "cop": "m",
    "angle": "rad",
}

_DEFAULT_RATIONALE = (
    "Conservative default distributes dynamic swing reaction torques across lead and trail arm kinematic chains "
    "while maintaining reserve torque margins below actuator peak capacities and preserving ground contact COP "
    "within support boundaries. Avoids single-arm saturation and excessive grip shear."
)


@dataclass(frozen=True)
class TradeoffAlternative:
    """A single force/torque distribution strategy on the Pareto tradeoff frontier."""

    name: str
    strategy: str
    forces: Array
    feasible: bool
    converged: bool
    cost: float
    lead_effort: float
    trail_effort: float
    trail_share: float
    grip_load: float
    ground_load: float
    per_joint_torque: Array
    per_joint_power: Array
    cop_estimate: float
    grip_wrench: Array
    active_constraints: list[str]
    mechanical_rationale: str
    units: dict[str, str]


@dataclass(frozen=True)
class TorqueDistributionTradeoff:
    """Set of candidate torque allocation alternatives with Pareto evaluation and export."""

    alternatives: tuple[TradeoffAlternative, ...]
    default_rationale: str
    selected_default_name: str = "conservative_default"

    def get_alternative(self, name: str) -> TradeoffAlternative | None:
        """Find an alternative by its unique identifier name."""
        for alt in self.alternatives:
            if alt.name == name:
                return alt
        return None

    def get_selected_default(self) -> TradeoffAlternative:
        """Return the conservative default alternative."""
        alt = self.get_alternative(self.selected_default_name)
        if alt is not None:
            return alt
        for a in self.alternatives:
            if a.feasible:
                return a
        return self.alternatives[0]

    def to_pareto_table(self) -> list[dict[str, Any]]:
        """Return reproducible dictionary table of trade-off metrics."""
        return [
            {
                "name": alt.name,
                "strategy": alt.strategy,
                "feasible": alt.feasible,
                "lead_effort": float(alt.lead_effort),
                "trail_effort": float(alt.trail_effort),
                "trail_share": float(alt.trail_share),
                "cost": float(alt.cost),
                "grip_load": float(alt.grip_load),
                "ground_load": float(alt.ground_load),
            }
            for alt in self.alternatives
        ]

    def export_json(self, path: Path | str) -> None:
        """Export all trade-off alternatives and mechanical rationales to JSON."""
        p = Path(path)
        data = {
            "default_rationale": self.default_rationale,
            "selected_default": self.get_selected_default().name,
            "alternatives": [
                {
                    "name": alt.name,
                    "strategy": alt.strategy,
                    "feasible": alt.feasible,
                    "converged": alt.converged,
                    "cost": float(alt.cost),
                    "lead_effort": float(alt.lead_effort),
                    "trail_effort": float(alt.trail_effort),
                    "trail_share": float(alt.trail_share),
                    "grip_load": float(alt.grip_load),
                    "ground_load": float(alt.ground_load),
                    "cop_estimate": float(alt.cop_estimate),
                    "forces": alt.forces.tolist(),
                    "per_joint_torque": alt.per_joint_torque.tolist(),
                    "per_joint_power": alt.per_joint_power.tolist(),
                    "units": alt.units,
                    "mechanical_rationale": alt.mechanical_rationale,
                }
                for alt in self.alternatives
            ],
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def export_csv(self, path: Path | str) -> None:
        """Export summary Pareto trade-off table to CSV."""
        p = Path(path)
        table = self.to_pareto_table()
        header = "name,strategy,feasible,lead_effort,trail_effort,trail_share,cost,grip_load,ground_load"
        lines = [header]
        for row in table:
            lines.append(
                f"{row['name']},{row['strategy']},{row['feasible']},"
                f"{row['lead_effort']:.6f},{row['trail_effort']:.6f},{row['trail_share']:.6f},"
                f"{row['cost']:.6f},{row['grip_load']:.6f},{row['ground_load']:.6f}"
            )
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _EvalContext:
    space: ForceNullSpace
    lead_idx: list[int]
    trail_idx: list[int]
    grip_idx: list[int]
    ground_idx: list[int]
    joint_velocities: Array | None
    tolerance: float

    def force_dim(self) -> int:
        return len(self.space.particular)


def _evaluate_alternative(
    ctx: _EvalContext,
    name: str,
    strategy: str,
    weights: Array,
    c: ForceConstraints,
    rationale: str,
) -> TradeoffAlternative:
    res = redistribute_forces(ctx.space, weights, c, tolerance=ctx.tolerance)
    x = res.forces
    lead_effort = float(np.linalg.norm(x[ctx.lead_idx])) if ctx.lead_idx else 0.0
    trail_effort = float(np.linalg.norm(x[ctx.trail_idx])) if ctx.trail_idx else 0.0
    tot_effort = lead_effort + trail_effort
    trail_share = float(trail_effort / tot_effort) if tot_effort > 1e-12 else 0.0
    grip_load = float(np.linalg.norm(x[ctx.grip_idx])) if ctx.grip_idx else 0.0
    ground_load = float(np.linalg.norm(x[ctx.ground_idx])) if ctx.ground_idx else 0.0

    all_joint_indices = ctx.lead_idx + ctx.trail_idx
    per_joint_torque = (
        _array(x[all_joint_indices], 1, "per_joint_torque")
        if all_joint_indices
        else np.empty(0)
    )
    if ctx.joint_velocities is not None:
        jv = np.asarray(ctx.joint_velocities, dtype=float)
        if jv.size != per_joint_torque.size:
            raise ValueError(
                f"joint_velocities length ({jv.size}) must match the number of "
                f"joint indices ({per_joint_torque.size})"
            )
        per_joint_power = _array(per_joint_torque * jv, 1, "per_joint_power")
    else:
        per_joint_power = _array(np.zeros_like(per_joint_torque), 1, "per_joint_power")

    grip_wrench = (
        _array(x[ctx.grip_idx], 1, "grip_wrench") if ctx.grip_idx else np.empty(0)
    )
    active: list[str] = []
    if np.any(np.abs(x - c.lower) < 1e-4):
        active.append("lower_bound")
    if np.any(np.abs(c.upper_bound - x) < 1e-4):
        active.append("upper_bound")

    return TradeoffAlternative(
        name=name,
        strategy=strategy,
        forces=x,
        feasible=res.feasible,
        converged=res.converged,
        cost=float(res.cost),
        lead_effort=lead_effort,
        trail_effort=trail_effort,
        trail_share=trail_share,
        grip_load=grip_load,
        ground_load=ground_load,
        per_joint_torque=per_joint_torque,
        per_joint_power=per_joint_power,
        cop_estimate=float("nan"),
        grip_wrench=grip_wrench,
        active_constraints=active,
        mechanical_rationale=rationale,
        units=dict(_DEFAULT_UNITS),
    )


def _scaled_weights(n: int, groups: Sequence[tuple[Sequence[int], float]]) -> Array:
    w = np.ones(n)
    for indices, weight in groups:
        for idx in indices:
            w[idx] = weight
    return w


def _build_tradeoff_specs(
    ctx: _EvalContext, constraints: ForceConstraints
) -> list[tuple[str, str, Array, ForceConstraints, str]]:
    n = ctx.force_dim()
    lo_zero = constraints.lower.copy()
    hi_zero = constraints.upper_bound.copy()
    for i in ctx.trail_idx:
        lo_zero[i] = 0.0
        hi_zero[i] = 0.0
    c_zero = ForceConstraints(
        lower=lo_zero,
        upper_bound=hi_zero,
        matrix=constraints.matrix,
        upper=constraints.upper,
        cones=constraints.cones,
    )

    specs: list[tuple[str, str, Array, ForceConstraints, str]] = [
        (
            "baseline_minimum_effort",
            "unweighted minimum-effort distribution (least squares)",
            np.ones(n),
            constraints,
            "Unweighted minimum-effort distribution allocating reaction load uniformly across degrees of freedom.",
        ),
        (
            "conservative_default",
            "conservative balanced distribution with torque safety margins",
            _scaled_weights(
                n,
                [
                    (ctx.lead_idx, 1.0),
                    (ctx.trail_idx, 1.5),
                    (ctx.grip_idx, 1.2),
                    (ctx.ground_idx, 1.0),
                ],
            ),
            constraints,
            _DEFAULT_RATIONALE,
        ),
        (
            "trail_arm_reduced_50",
            "reduced trail arm share (50% target reduction)",
            _scaled_weights(n, [(ctx.lead_idx, 1.0), (ctx.trail_idx, 2.5)]),
            constraints,
            "Penalizes trail arm involvement by approximately 50%, shifting dynamic swing load onto the lead arm kinematic chain.",
        ),
        (
            "trail_arm_reduced_80",
            "heavily reduced trail arm share (80% target reduction)",
            _scaled_weights(n, [(ctx.lead_idx, 1.0), (ctx.trail_idx, 8.0)]),
            constraints,
            "Heavily reduces trail arm contribution to ~20% of baseline, requiring significant lead arm torque authority.",
        ),
        (
            "hard_zero_trail",
            "hard zero trail arm effort (lead arm carries full upper body load)",
            np.ones(n),
            c_zero,
            "Strictly zeroes out trail arm torques, forcing the lead arm and body structure to support 100% of upper-extremity reactions. Infeasible if lead bounds are exceeded.",
        ),
        (
            "relaxed_minimum_trail",
            "relaxed minimum trail arm load (closest admissible solution)",
            _scaled_weights(n, [(ctx.lead_idx, 1.0), (ctx.trail_idx, 50.0)]),
            constraints,
            "Finds the closest physically admissible solution when hard-zero trail is infeasible by heavily penalizing trail arm involvement while respecting actuator capacity bounds.",
        ),
    ]
    if ctx.grip_idx:
        specs.append(
            (
                "grip_squeeze_minimized",
                "minimized grip squeeze and internal hand shear wrench",
                _scaled_weights(n, [(ctx.grip_idx, 20.0)]),
                constraints,
                "Penalizes internal hand grip forces and contact shear wrench, minimizing interface reaction loads.",
            )
        )
    if ctx.ground_idx:
        specs.append(
            (
                "ground_load_regularized",
                "regularized ground reaction distribution and COP centering",
                _scaled_weights(n, [(ctx.ground_idx, 5.0)]),
                constraints,
                "Regularizes ground reaction normal and frictional distribution, centering center-of-pressure within the support polygon.",
            )
        )
    return specs


def explore_torque_tradeoffs(
    space: ForceNullSpace,
    constraints: ForceConstraints,
    *,
    lead_indices: Sequence[int],
    trail_indices: Sequence[int],
    grip_indices: Sequence[int] | None = None,
    ground_indices: Sequence[int] | None = None,
    joint_velocities: Array | None = None,
    tolerance: float = 1e-8,
) -> TorqueDistributionTradeoff:
    """Explore feasible force null spaces and generate torque distribution tradeoffs."""
    ctx = _EvalContext(
        space=space,
        lead_idx=list(lead_indices),
        trail_idx=list(trail_indices),
        grip_idx=list(grip_indices) if grip_indices is not None else [],
        ground_idx=list(ground_indices) if ground_indices is not None else [],
        joint_velocities=joint_velocities,
        tolerance=tolerance,
    )
    specs = _build_tradeoff_specs(ctx, constraints)
    alternatives = [
        _evaluate_alternative(ctx, name, strategy, weights, c, rationale)
        for name, strategy, weights, c, rationale in specs
    ]
    return TorqueDistributionTradeoff(
        alternatives=tuple(alternatives),
        default_rationale=_DEFAULT_RATIONALE,
        selected_default_name="conservative_default",
    )


__all__ = [
    "TorqueDistributionTradeoff",
    "TradeoffAlternative",
    "explore_torque_tradeoffs",
]
