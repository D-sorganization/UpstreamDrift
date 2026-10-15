"""Torque-distribution tradeoff exploration (NSX-03 split from force_nullspace)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._common import Array, _array
from .constraints import ForceConstraints, FrictionCone
from .redistribution import RedistributionResult, redistribute_forces
from .space import ForceNullSpace


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
    n = space.particular.size
    lead_idx = list(lead_indices)
    trail_idx = list(trail_indices)
    grip_idx = list(grip_indices) if grip_indices is not None else []
    ground_idx = list(ground_indices) if ground_indices is not None else []

    all_joint_indices = lead_idx + trail_idx

    units = {
        "torque": "N*m",
        "force": "N",
        "power": "W",
        "cop": "m",
        "angle": "rad",
    }

    def evaluate_alternative(
        name: str,
        strategy: str,
        weights: Array,
        c: ForceConstraints,
        rationale: str,
    ) -> TradeoffAlternative:
        res = redistribute_forces(space, weights, c, tolerance=tolerance)
        x = res.forces
        lead_effort = float(np.linalg.norm(x[lead_idx])) if lead_idx else 0.0
        trail_effort = float(np.linalg.norm(x[trail_idx])) if trail_idx else 0.0
        tot_effort = lead_effort + trail_effort
        trail_share = float(trail_effort / tot_effort) if tot_effort > 1e-12 else 0.0
        grip_load = float(np.linalg.norm(x[grip_idx])) if grip_idx else 0.0
        ground_load = float(np.linalg.norm(x[ground_idx])) if ground_idx else 0.0

        per_joint_torque = (
            _array(x[all_joint_indices], 1, "per_joint_torque")
            if all_joint_indices
            else np.empty(0)
        )
        if joint_velocities is not None:
            jv = np.asarray(joint_velocities, dtype=float)
            if jv.size != per_joint_torque.size:
                raise ValueError(
                    f"joint_velocities length ({jv.size}) must match the number of "
                    f"joint indices ({per_joint_torque.size})"
                )
            per_joint_power = _array(per_joint_torque * jv, 1, "per_joint_power")
        else:
            per_joint_power = _array(
                np.zeros_like(per_joint_torque), 1, "per_joint_power"
            )

        cop_estimate = float("nan")

        grip_wrench = _array(x[grip_idx], 1, "grip_wrench") if grip_idx else np.empty(0)

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
            cop_estimate=cop_estimate,
            grip_wrench=grip_wrench,
            active_constraints=active,
            mechanical_rationale=rationale,
            units=units,
        )

    alternatives: list[TradeoffAlternative] = []

    # 1. baseline_minimum_effort
    w_base = np.ones(n)
    alternatives.append(
        evaluate_alternative(
            "baseline_minimum_effort",
            "unweighted minimum-effort distribution (least squares)",
            w_base,
            constraints,
            "Unweighted minimum-effort distribution allocating reaction load uniformly across degrees of freedom.",
        )
    )

    # 2. conservative_default
    w_cons = np.ones(n)
    for i in lead_idx:
        w_cons[i] = 1.0
    for i in trail_idx:
        w_cons[i] = 1.5
    for i in grip_idx:
        w_cons[i] = 1.2
    for i in ground_idx:
        w_cons[i] = 1.0
    alternatives.append(
        evaluate_alternative(
            "conservative_default",
            "conservative balanced distribution with torque safety margins",
            w_cons,
            constraints,
            "Conservative default distributes dynamic swing reaction torques across lead and trail arm kinematic chains while maintaining reserve torque margins below actuator peak capacities and preserving ground contact COP within support boundaries. Avoids single-arm saturation and excessive grip shear.",
        )
    )

    # 3. trail_arm_reduced_50
    w_50 = np.ones(n)
    for i in lead_idx:
        w_50[i] = 1.0
    for i in trail_idx:
        w_50[i] = 2.5
    alternatives.append(
        evaluate_alternative(
            "trail_arm_reduced_50",
            "reduced trail arm share (50% target reduction)",
            w_50,
            constraints,
            "Penalizes trail arm involvement by approximately 50%, shifting dynamic swing load onto the lead arm kinematic chain.",
        )
    )

    # 4. trail_arm_reduced_80
    w_80 = np.ones(n)
    for i in lead_idx:
        w_80[i] = 1.0
    for i in trail_idx:
        w_80[i] = 8.0
    alternatives.append(
        evaluate_alternative(
            "trail_arm_reduced_80",
            "heavily reduced trail arm share (80% target reduction)",
            w_80,
            constraints,
            "Heavily reduces trail arm contribution to ~20% of baseline, requiring significant lead arm torque authority.",
        )
    )

    # 5. hard_zero_trail
    lo_zero = constraints.lower.copy()
    hi_zero = constraints.upper_bound.copy()
    for i in trail_idx:
        lo_zero[i] = 0.0
        hi_zero[i] = 0.0
    c_zero = ForceConstraints(
        lower=lo_zero,
        upper_bound=hi_zero,
        matrix=constraints.matrix,
        upper=constraints.upper,
        cones=constraints.cones,
    )
    alternatives.append(
        evaluate_alternative(
            "hard_zero_trail",
            "hard zero trail arm effort (lead arm carries full upper body load)",
            np.ones(n),
            c_zero,
            "Strictly zeroes out trail arm torques, forcing the lead arm and body structure to support 100% of upper-extremity reactions. Infeasible if lead bounds are exceeded.",
        )
    )

    # 6. relaxed_minimum_trail
    w_relaxed = np.ones(n)
    for i in lead_idx:
        w_relaxed[i] = 1.0
    for i in trail_idx:
        w_relaxed[i] = 50.0
    alternatives.append(
        evaluate_alternative(
            "relaxed_minimum_trail",
            "relaxed minimum trail arm load (closest admissible solution)",
            w_relaxed,
            constraints,
            "Finds the closest physically admissible solution when hard-zero trail is infeasible by heavily penalizing trail arm involvement while respecting actuator capacity bounds.",
        )
    )

    # 7. grip_squeeze_minimized (if grip_indices given)
    if grip_idx:
        w_grip = np.ones(n)
        for i in grip_idx:
            w_grip[i] = 20.0
        alternatives.append(
            evaluate_alternative(
                "grip_squeeze_minimized",
                "minimized grip squeeze and internal hand shear wrench",
                w_grip,
                constraints,
                "Penalizes internal hand grip forces and contact shear wrench, minimizing interface reaction loads.",
            )
        )

    # 8. ground_load_regularized (if ground_indices given)
    if ground_idx:
        w_ground = np.ones(n)
        for i in ground_idx:
            w_ground[i] = 5.0
        alternatives.append(
            evaluate_alternative(
                "ground_load_regularized",
                "regularized ground reaction distribution and COP centering",
                w_ground,
                constraints,
                "Regularizes ground reaction normal and frictional distribution, centering center-of-pressure within the support polygon.",
            )
        )

    default_rationale = (
        "Conservative default distributes dynamic swing reaction torques across lead and trail arm kinematic chains "
        "while maintaining reserve torque margins below actuator peak capacities and preserving ground contact COP "
        "within support boundaries. Avoids single-arm saturation and excessive grip shear."
    )

    return TorqueDistributionTradeoff(
        alternatives=tuple(alternatives),
        default_rationale=default_rationale,
        selected_default_name="conservative_default",
    )
