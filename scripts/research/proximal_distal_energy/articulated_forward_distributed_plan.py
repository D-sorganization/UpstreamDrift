"""Prospective serial distributed-attribution contract for #9153."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    ForwardAttributionStudyPlan,
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DistributedAttributionStudyPlan:
    """Frozen event-bearing distributed smoke; no human inference is eligible."""

    source_revision: str
    source_data_sha256: str

    def __post_init__(self) -> None:
        if not _SHA40.fullmatch(self.source_revision):
            raise ValueError("source_revision must be a lowercase 40-character SHA")
        if not _SHA256.fullmatch(self.source_data_sha256):
            raise ValueError("source_data_sha256 must be a lowercase SHA-256")

    def to_manifest(self) -> dict[str, Any]:
        """Return the deterministic, runner-compatible prospective manifest."""

        manifest = ForwardAttributionStudyPlan(
            source_revision=self.source_revision,
            source_data_sha256=self.source_data_sha256,
            duration_s=0.05,
        ).to_manifest()
        manifest["schema_version"] = "distributed-forward-attribution/1.0.0"
        manifest["preregistration"] = {
            "timing": "before_registered_distributed_campaign",
            "pilot_disclosure": [
                "rigid smoke identified nominal and high-damping coarse-grid transients",
                "a noncampaign 50 ms distributed probe located opening, reattachment, and friction-limit-entry roots",
            ],
            "pilot_exclusion": "pilot outputs are not registered campaign evidence",
            "bound_evaluator_revision": self.source_revision,
        }
        design = manifest["design"]
        design["smoke_states"] = [
            {
                "source_case_index": 0,
                "source_sample_index": 6,
                "source_time_s": 0.12,
                "role": "event_bearing_runtime_and_pipeline_qualification_only",
            }
        ]
        design["variants"] = [row["name"] for row in _variant_rows()]
        design["variant_parameters"] = _variant_rows()
        design["distributed_contact_law"] = {
            "name": "distributed_tension_with_regularized_coulomb_limit",
            "station_count_per_hand": 3,
            "station_width_m": 0.03,
            "tangential_damping_n_s_m": 18.0,
            "friction_coefficient": 0.3,
            "slack_distance_m": 0.0015,
            "static_stick_modeled": False,
        }
        design["event_kinds"] = [
            "opening",
            "reattachment",
            "friction_limit_entry",
            "friction_limit_exit",
        ]
        manifest["execution"]["estimated_registered_case_count"] = 42
        manifest["execution"]["launch_status"] = "not_started"
        manifest["promotion"].update(
            {
                "original_rigid_smoke_failure_erased": False,
                "cross_engine_parity_required": True,
                "static_stick_claims": False,
                "event_topology_retained": True,
            }
        )
        return manifest


def _variant_rows() -> list[dict[str, float | str]]:
    common: dict[str, float] = {
        "stiffness_factor": 1.0,
        "damping_factor": 1.0,
        "displacement_factor": 1.0,
        "velocity_factor": 1.0,
        "friction_coefficient_factor": 1.0,
        "slack_distance_factor": 1.0,
        "station_width_factor": 1.0,
        "full_state_velocity_factor": 1.0,
    }

    def row(name: str, **changes: float) -> dict[str, float | str]:
        return {"name": name, **common, **changes}

    return [
        row("nominal"),
        row("frictionless", friction_coefficient_factor=0.0),
        row("high_friction", friction_coefficient_factor=2.0),
        row("zero_slack", slack_distance_factor=0.0),
        row(
            "velocity_reversed",
            velocity_factor=-1.0,
            full_state_velocity_factor=-1.0,
        ),
        row("zero_preload", displacement_factor=0.0, velocity_factor=0.0),
        row("opening_probe", velocity_factor=-16.0),
    ]


__all__ = ["DistributedAttributionStudyPlan"]
