"""Native replay-ready packages with explicit unmet qualification gates (CO-04)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.shared.python.motion_matching.club_only.pendulum_match import (
    PendulumMatchResult,
)

REPLAY_SCHEMA = "club-pendulum-replay/1.0.0"

__all__ = [
    "REPLAY_SCHEMA",
    "ClubPendulumReplayPackage",
    "build_replay_package",
]

_DEFAULT_BLOCKERS = (
    "native_g1_qualification_requires_tb05_or_desk_native_receipt",
    "synthetic_or_software_contract_fit_is_not_native_evidence",
)


@dataclass(frozen=True)
class ClubPendulumReplayPackage:
    """Replay inputs for independent CO-08 review; never invents native G1 pass."""

    schema: str
    model_id: str
    trial_id: str
    hub_variant_id: str
    native_g1_pass: bool
    claims_native_qualification: bool
    qualification_blockers: tuple[str, ...]
    replay_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema != REPLAY_SCHEMA:
            raise ValueError(f"schema must be {REPLAY_SCHEMA!r}")
        if self.native_g1_pass and not self.claims_native_qualification:
            raise ValueError("native_g1_pass requires claims_native_qualification")
        if self.native_g1_pass and self.qualification_blockers:
            raise ValueError("native_g1_pass cannot retain qualification blockers")
        if not self.native_g1_pass and not self.qualification_blockers:
            raise ValueError("unmet native gates must name at least one blocker")
        object.__setattr__(
            self, "qualification_blockers", tuple(self.qualification_blockers)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "trial_id": self.trial_id,
            "hub_variant_id": self.hub_variant_id,
            "native_g1_pass": self.native_g1_pass,
            "claims_native_qualification": self.claims_native_qualification,
            "qualification_blockers": list(self.qualification_blockers),
            "replay_inputs": dict(self.replay_inputs),
        }


def build_replay_package(result: PendulumMatchResult) -> ClubPendulumReplayPackage:
    """Serialize a match into a replay-ready package with explicit blockers."""
    blockers = tuple(result.qualification_blockers) or _DEFAULT_BLOCKERS
    replay_inputs = {
        "theta": np.asarray(result.theta, dtype=np.float64).tolist(),
        "q0": np.asarray(result.q0, dtype=np.float64).tolist(),
        "v0": np.asarray(result.v0, dtype=np.float64).tolist(),
        "times_s": np.asarray(result.times_s, dtype=np.float64).tolist(),
        "l_arm_m": result.l_arm_m,
        "l_club_m": result.l_club_m,
        "l_hub_m": result.l_hub_m,
        "hub_mode": result.hub_mode.value,
        "selected_start": result.selected_start,
        "t0_evaluated_before_step": result.t0_evaluated_before_step,
        "in_plane_rmse_m": result.in_plane_rmse_m,
        "original_3d_rmse_m": result.original_3d_rmse_m,
        "external_work_joules": result.external_work_joules,
        "coverage_fraction": result.coverage_fraction,
    }
    return ClubPendulumReplayPackage(
        schema=REPLAY_SCHEMA,
        model_id=result.model_id,
        trial_id=result.trial_id,
        hub_variant_id=result.hub_variant_id,
        native_g1_pass=False,
        claims_native_qualification=False,
        qualification_blockers=blockers,
        replay_inputs=replay_inputs,
    )
