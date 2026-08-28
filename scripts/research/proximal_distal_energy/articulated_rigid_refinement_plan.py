"""Prospective rigid-refinement extension after the disclosed #9153 pilot."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    ForwardAttributionStudyPlan,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    registered_variants,
)


_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIRMATORY_STEPS_S = (0.0002, 0.0001, 0.00005)
_PILOT_STEPS_S = (0.00025, 0.000125, 0.0000625)
_CASE_INDICES = (0, 4, 8, 9, 13, 17)
_SAMPLE_INDICES = (0, 6, 12)
_SAMPLE_TIMES_S = {0: 0.0, 6: 0.12, 12: 0.24}
_VARIANTS = ("nominal", "damping_high")


@dataclass(frozen=True, slots=True)
class RigidRefinementExtensionPlan:
    """Freeze a screening-state refinement study that is disjoint from its pilot."""

    source_revision: str
    source_data_sha256: str
    worker_count: int = 1

    def __post_init__(self) -> None:
        if not _SHA40.fullmatch(self.source_revision):
            raise ValueError("source_revision must be a lowercase 40-character SHA")
        if not _SHA256.fullmatch(self.source_data_sha256):
            raise ValueError("source_data_sha256 must be a lowercase SHA-256")
        if self.worker_count != 1:
            raise ValueError("worker_count must remain one for this serial extension")

    def to_manifest(self) -> dict[str, Any]:
        """Return the deterministic runner-compatible preregistration."""

        manifest = ForwardAttributionStudyPlan(
            source_revision=self.source_revision,
            source_data_sha256=self.source_data_sha256,
            time_steps_s=_CONFIRMATORY_STEPS_S,
            worker_count=self.worker_count,
        ).to_manifest()
        manifest["schema_version"] = "rigid-refinement-extension/1.0.0"
        manifest["preregistration"] = {
            "timing": "after_disclosed_development_pilot_before_extension_execution",
            "reason": (
                "the original rigid smoke failed the frozen work-refinement gate "
                "for nominal and damping_high"
            ),
            "original_smoke_result": {
                "nominal_work_refinement_ratios": [
                    0.8291266315013308,
                    0.5859514678225363,
                ],
                "damping_high_work_refinement_ratios": [
                    1.2354797329442686,
                    0.6009489640811517,
                ],
                "refinement_ratio_limit": 0.8,
                "status": "failed_and_retained",
            },
            "pilot_steps_s": list(_PILOT_STEPS_S),
            "pilot_disclosure": (
                "a post-result development-only run on the original smoke state "
                "showed contraction at finer steps"
            ),
            "pilot_exclusion": "pilot outputs are not confirmatory campaign evidence",
            "confirmatory_steps_disjoint": not bool(
                set(_CONFIRMATORY_STEPS_S) & set(_PILOT_STEPS_S)
            ),
            "state_selection": (
                "all 18 previously declared screening states; no state was selected "
                "from the pilot outcome"
            ),
            "bound_evaluator_revision": self.source_revision,
        }
        variants = {variant.name: variant for variant in registered_variants()}
        design = manifest["design"]
        design["variants"] = list(_VARIANTS)
        design["variant_parameters"] = [
            {
                "name": variants[name].name,
                "stiffness_factor": variants[name].stiffness_factor,
                "damping_factor": variants[name].damping_factor,
                "displacement_factor": variants[name].displacement_factor,
                "velocity_factor": variants[name].velocity_factor,
            }
            for name in _VARIANTS
        ]
        design["smoke_states"] = [
            {
                "source_case_index": case_index,
                "source_sample_index": sample_index,
                "source_time_s": _SAMPLE_TIMES_S[sample_index],
                "role": "prospective_screening_state",
            }
            for case_index in _CASE_INDICES
            for sample_index in _SAMPLE_INDICES
        ]
        execution = manifest["execution"]
        execution.update(
            {
                "estimated_registered_case_count": 216,
                "estimated_native_attempt_count": 108,
                "estimated_runtime_s": 90,
                "resource_contract": (
                    "DeskComputer serial CPU; one Python process; no ControlTower use"
                ),
                "launch_status": "not_started",
            }
        )
        manifest["promotion"].update(
            {
                "original_rigid_smoke_failure_erased": False,
                "all_screening_states_retained": True,
                "cross_engine_parity_required": True,
                "pilot_outputs_eligible": False,
            }
        )
        return manifest


__all__ = ["RigidRefinementExtensionPlan"]
