"""Effort-hold policy contract and bundle manifest helper (#12279).

The reference export evaluates the computed-torque controller once per
effort_hold_dt_s (default 1 ms, i.e. 0.001 s, with optional 0.125 ms, i.e.
0.000125 s with substeps=1 while keeping closure projection at 1 ms).
The policy parameters (effort_hold_dt_s, effort_substeps, projection_dt_s)
are stored explicitly in the bundle manifest and read by all engine replays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EFFORT_HOLD_DT_S: float = 0.001
FINE_EFFORT_HOLD_DT_S: float = 0.000125
DEFAULT_PROJECTION_DT_S: float = 0.001
DEFAULT_EFFORT_SUBSTEPS: int = 8
FINE_EFFORT_SUBSTEPS: int = 1

KNOWN_7IRON_FOOT_SLIP_1MS_MM: float = 43.0
KNOWN_7IRON_FOOT_SLIP_FINE_MM: float = 3.3
FOOT_SLIP_BOUND_MM: float = 10.0


@dataclass(frozen=True)
class EffortHoldPolicy:
    """Explicit effort hold rate policy contract (#12279).

    Parameters:
        effort_hold_dt_s: Controller zero-order hold interval in seconds.
        effort_substeps: RK4 substeps integrated under each held effort sample.
        projection_dt_s: Interval between kinematic closure projections.
    """

    effort_hold_dt_s: float = DEFAULT_EFFORT_HOLD_DT_S
    effort_substeps: int = DEFAULT_EFFORT_SUBSTEPS
    projection_dt_s: float = DEFAULT_PROJECTION_DT_S

    def __post_init__(self) -> None:
        if not (np.isfinite(self.effort_hold_dt_s) and self.effort_hold_dt_s > 0.0):
            raise ValueError("effort_hold_dt_s must be positive and finite")
        if not (np.isfinite(self.projection_dt_s) and self.projection_dt_s > 0.0):
            raise ValueError("projection_dt_s must be positive and finite")
        if isinstance(self.effort_substeps, bool) or not isinstance(
            self.effort_substeps, int
        ):
            raise TypeError("effort_substeps must be an integer")
        if self.effort_substeps < 1:
            raise ValueError("effort_substeps must be positive")

    @property
    def effort_holds_per_step(self) -> int:
        """Number of effort hold periods per closure projection step."""
        ratio = self.projection_dt_s / self.effort_hold_dt_s
        return max(1, int(round(ratio)))

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return manifest policy dictionary entries."""
        return {
            "effort_hold_dt_s": self.effort_hold_dt_s,
            "effort_substeps": self.effort_substeps,
            "projection_dt_s": self.projection_dt_s,
            "effort_holds_per_step": self.effort_holds_per_step,
            "effort_hold_s": self.effort_hold_dt_s,
        }


def resolve_effort_hold_dt_s(
    effort_hold_s: float | None = None,
    effort_hold_dt_s: float | None = None,
) -> float:
    """Resolve effort_hold_dt_s from either parameter name (#12279)."""
    if effort_hold_dt_s is not None and effort_hold_s is not None:
        if abs(effort_hold_dt_s - effort_hold_s) > 1e-12:
            raise ValueError("effort_hold_dt_s and effort_hold_s must match")
        return float(effort_hold_dt_s)
    if effort_hold_dt_s is not None:
        return float(effort_hold_dt_s)
    if effort_hold_s is not None:
        return float(effort_hold_s)
    return DEFAULT_EFFORT_HOLD_DT_S


def _extract_manifest_policy(source: Any) -> tuple[dict[str, Any], float]:
    """Extract policy dict and projection dt_s from various sources."""
    if hasattr(source, "manifest") and callable(source.manifest):
        data = source.manifest()
        return data.get("policy", {}), float(data.get("dt_s", DEFAULT_PROJECTION_DT_S))
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as bundle_data:
            manifest_json = json.loads(str(bundle_data["manifest"]))
            return manifest_json.get("policy", {}), float(
                manifest_json.get("dt_s", DEFAULT_PROJECTION_DT_S)
            )
    if isinstance(source, dict):
        if "manifest" in source and isinstance(source["manifest"], dict):
            return _extract_manifest_policy(source["manifest"])
        if "policy" in source and isinstance(source["policy"], dict):
            return source["policy"], float(source.get("dt_s", DEFAULT_PROJECTION_DT_S))
        return source, float(
            source.get("dt_s", source.get("projection_dt_s", DEFAULT_PROJECTION_DT_S))
        )
    raise TypeError(f"cannot read effort hold policy from {type(source).__name__}")


def read_effort_hold_policy(source: Any) -> EffortHoldPolicy:
    """Read the effort hold policy from a bundle, manifest dict, or path (#12279).

    This unified helper allows engine replays (MuJoCo, Drake, Simscape, etc.)
    to interpret the bundle's effort hold configuration consistently.
    """
    if hasattr(source, "effort_hold_policy") and callable(source.effort_hold_policy):
        return source.effort_hold_policy()  # type: ignore[no-any-return]

    policy_dict, dt_s = _extract_manifest_policy(source)

    if "effort_hold_dt_s" in policy_dict:
        effort_hold_dt_s = float(policy_dict["effort_hold_dt_s"])
    elif "effort_hold_s" in policy_dict:
        effort_hold_dt_s = float(policy_dict["effort_hold_s"])
    elif "effort_holds_per_step" in policy_dict:
        effort_hold_dt_s = dt_s / int(policy_dict["effort_holds_per_step"])
    else:
        effort_hold_dt_s = DEFAULT_EFFORT_HOLD_DT_S

    if "effort_substeps" in policy_dict:
        effort_substeps = int(policy_dict["effort_substeps"])
    elif "substeps" in policy_dict and "effort_holds_per_step" in policy_dict:
        substeps = int(policy_dict["substeps"])
        holds = int(policy_dict["effort_holds_per_step"])
        effort_substeps = max(1, substeps // holds)
    elif abs(effort_hold_dt_s - FINE_EFFORT_HOLD_DT_S) < 1e-9:
        effort_substeps = FINE_EFFORT_SUBSTEPS
    else:
        effort_substeps = DEFAULT_EFFORT_SUBSTEPS

    if "projection_dt_s" in policy_dict:
        projection_dt_s = float(policy_dict["projection_dt_s"])
    else:
        projection_dt_s = dt_s

    return EffortHoldPolicy(
        effort_hold_dt_s=effort_hold_dt_s,
        effort_substeps=effort_substeps,
        projection_dt_s=projection_dt_s,
    )
