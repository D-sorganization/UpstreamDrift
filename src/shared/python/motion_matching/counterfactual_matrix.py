"""Executable Multi-Engine Counterfactual Capability Matrix (#10286, CF-4).

Ratified per-engine qualification matrix for native swing counterfactual
acceleration and spatial reaction-wrench analysis across the five primary
physics backends: Pinocchio, MuJoCo, Drake, OpenSim, and Simscape R2025b.

DbC & Invariants:
- All capability descriptors are immutable and type-checked.
- State restoration must be guaranteed via exception-safe try/finally blocks.
- Distinguishes unconstrained ABA / pendulum approximations from qualified
  constrained dynamics and 6D spatial contact reactions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from src.shared.python.engine_core.capabilities import CapabilityLevel

__all__ = [
    "COUNTERFACTUAL_CAPABILITY_MATRIX",
    "CounterfactualEngineCapability",
    "format_capability_matrix_markdown",
    "get_counterfactual_capability_matrix",
    "get_engine_counterfactual_capability",
]


@dataclass(frozen=True, slots=True)
class CounterfactualEngineCapability:
    """Immutable capability record for a physics engine counterfactual provider."""

    engine_name: str
    backend_key: str
    ztcf_status: CapabilityLevel
    zvcf_status: CapabilityLevel
    reaction_wrench_status: CapabilityLevel
    state_restoration_guarantee: bool
    constraint_dynamics_support: str
    actuation_mapping: str
    qualification_evidence: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert capability record to serialized JSON-ready dictionary."""
        data = asdict(self)
        data["ztcf_status"] = self.ztcf_status.name
        data["zvcf_status"] = self.zvcf_status.name
        data["reaction_wrench_status"] = self.reaction_wrench_status.name
        return data

    @property
    def supports_closed_loop_reaction(self) -> bool:
        """Return whether engine computes qualified 6D spatial closure reaction wrenches."""
        return self.reaction_wrench_status == CapabilityLevel.FULL


_MATRIX_DATA: dict[str, CounterfactualEngineCapability] = {
    "pinocchio": CounterfactualEngineCapability(
        engine_name="Pinocchio",
        backend_key="pinocchio",
        ztcf_status=CapabilityLevel.FULL,
        zvcf_status=CapabilityLevel.FULL,
        reaction_wrench_status=CapabilityLevel.FULL,
        state_restoration_guarantee=True,
        constraint_dynamics_support=(
            "Full native pin.constraintDynamics with 6D rigid weld contact-force "
            "extraction and world/joint frame transport via NativePinocchioModel"
        ),
        actuation_mapping="Joint-conjugate generalized torques and primitive polynomial efforts",
        qualification_evidence=(
            "src/engines/physics_engines/pinocchio/python/native_model.py, "
            "src/shared/python/motion_matching/counterfactual.py, "
            "tests/unit/motion_matching/test_native_counterfactual.py"
        ),
        notes="Zero caller mutation; exact closure a_act = a_ztcf + delta_a_ctrl",
    ),
    "mujoco": CounterfactualEngineCapability(
        engine_name="MuJoCo",
        backend_key="mujoco",
        ztcf_status=CapabilityLevel.FULL,
        zvcf_status=CapabilityLevel.FULL,
        reaction_wrench_status=CapabilityLevel.PARTIAL,
        state_restoration_guarantee=True,
        constraint_dynamics_support=(
            "Kinematic equality constraints (weld/connect), contact manifolds, "
            "and smooth forward dynamics via mj_forward"
        ),
        actuation_mapping="Actuator control vector (ctrl) mapped through transmission gearings",
        qualification_evidence=(
            "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py, "
            "tests/unit/test_mujoco_physics_engine.py"
        ),
        notes="Full try/finally state and control restoration with post-restoration mj_forward",
    ),
    "drake": CounterfactualEngineCapability(
        engine_name="Drake",
        backend_key="drake",
        ztcf_status=CapabilityLevel.FULL,
        zvcf_status=CapabilityLevel.FULL,
        reaction_wrench_status=CapabilityLevel.PARTIAL,
        state_restoration_guarantee=True,
        constraint_dynamics_support="Hydroelastic/point contact and MultibodyPlant kinematic constraints",
        actuation_mapping=(
            "Actuation matrix B (MakeActuationMatrix) projection from actuator commands "
            "u (nu) to generalized forces tau (nv)"
        ),
        qualification_evidence=(
            "src/engines/physics_engines/drake/python/drake_physics_engine.py, "
            "tests/analytical/test_engine_zvcf_jacobian_parity.py"
        ),
        notes="Actuator selector handles nu != nv underactuation and preserves context state",
    ),
    "opensim": CounterfactualEngineCapability(
        engine_name="OpenSim",
        backend_key="opensim",
        ztcf_status=CapabilityLevel.FULL,
        zvcf_status=CapabilityLevel.FULL,
        reaction_wrench_status=CapabilityLevel.PARTIAL,
        state_restoration_guarantee=True,
        constraint_dynamics_support="Simbody coordinate couplers and kinematic loop constraints via realizeDynamics",
        actuation_mapping="Coordinate actuators, generalized torque actuators, and Hill-type muscle fibers",
        qualification_evidence=(
            "src/engines/physics_engines/opensim/python/opensim_physics_engine.py, "
            "tests/unit/engines/test_mujoco_opensim_capabilities_7050.py"
        ),
        notes="Exception-safe try/finally restoration with re-realization to dynamics stage",
    ),
    "simscape": CounterfactualEngineCapability(
        engine_name="Simscape Multibody R2025b",
        backend_key="simscape",
        ztcf_status=CapabilityLevel.FULL,
        zvcf_status=CapabilityLevel.FULL,
        reaction_wrench_status=CapabilityLevel.FULL,
        state_restoration_guarantee=True,
        constraint_dynamics_support="3D multibody closed loops, weld joints, and internal variable-step ODE solvers",
        actuation_mapping="Joint torque primitives and motor actuators driven by polynomial inputs",
        qualification_evidence=(
            "src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab_optimized/core/simulation/run_ztcf_simulation.m, "
            "src/shared/python/simulation_store/replay_bundle.py, "
            "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
        ),
        notes="Evaluated via R2025b killswitch runs or verified offline replay bundles with contact sensing",
    ),
}

COUNTERFACTUAL_CAPABILITY_MATRIX: Mapping[str, CounterfactualEngineCapability] = (
    MappingProxyType(_MATRIX_DATA)
)


def get_counterfactual_capability_matrix() -> Mapping[
    str, CounterfactualEngineCapability
]:
    """Return the authoritative 5-engine counterfactual capability matrix."""
    return COUNTERFACTUAL_CAPABILITY_MATRIX


def get_engine_counterfactual_capability(
    engine_name: str,
) -> CounterfactualEngineCapability:
    """Look up the counterfactual capability record for an engine by name or backend key."""
    key = engine_name.strip().lower()
    for entry_key, entry in COUNTERFACTUAL_CAPABILITY_MATRIX.items():
        if key in (entry_key, entry.engine_name.lower(), entry.backend_key.lower()):
            return entry
    valid = ", ".join(COUNTERFACTUAL_CAPABILITY_MATRIX.keys())
    raise KeyError(f"Unknown engine '{engine_name}'; expected one of {valid}")


def format_capability_matrix_markdown() -> str:
    """Format the 5-engine capability matrix as a GitHub-flavored Markdown table."""
    lines: list[str] = [
        "| Engine | Backend Key | ZTCF | ZVCF | Spatial Wrench | State Restoration | Constraint Support | Actuator Mapping |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cap in COUNTERFACTUAL_CAPABILITY_MATRIX.values():
        lines.append(
            f"| {cap.engine_name} | `{cap.backend_key}` | {cap.ztcf_status.name} | "
            f"{cap.zvcf_status.name} | {cap.reaction_wrench_status.name} | "
            f"{'Guaranteed' if cap.state_restoration_guarantee else 'None'} | "
            f"{cap.constraint_dynamics_support[:45]}... | {cap.actuation_mapping[:40]}... |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # Format and output to stdout
    sys.stdout.write(
        format_capability_matrix_markdown() + "\n\nJSON Capability Summary:\n"
    )
    summary = {k: v.to_dict() for k, v in COUNTERFACTUAL_CAPABILITY_MATRIX.items()}
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
