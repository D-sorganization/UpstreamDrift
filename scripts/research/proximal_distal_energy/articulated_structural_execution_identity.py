"""Resolve fail-closed identities for structural headline atlas execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_structural_propagation_plan import (
    DEFAULT_OUTPUT,
    validate_structural_propagation_plan,
)

Pathway = Literal["shaft", "ground"]
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_SCHEMA_VERSION = "articulated-structural-checkpoint/v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _configuration_record(
    configuration: ArticulatedShaftAtlasConfig | ArticulatedGroundAtlasConfig,
) -> dict[str, Any]:
    record = asdict(configuration)
    record.pop("worker_count")
    return json.loads(json.dumps(record))


def scientific_configuration_sha256(
    configuration: ArticulatedShaftAtlasConfig | ArticulatedGroundAtlasConfig,
) -> str:
    """Digest scientific configuration while excluding operational parallelism."""

    if not isinstance(
        configuration, (ArticulatedShaftAtlasConfig, ArticulatedGroundAtlasConfig)
    ):
        raise TypeError("configuration must be a registered structural atlas config")
    return _canonical_sha256(_configuration_record(configuration))


@dataclass(frozen=True, slots=True)
class StructuralExecutionIdentity:
    """Immutable scientific prefix shared by every checkpoint in one execution."""

    corner_id: str
    pathway: Pathway
    authority_sha256: str
    scales: tuple[tuple[str, float], ...]
    model_sha256: tuple[tuple[str, str], ...]
    atlas_source_sha256: str
    scientific_configuration_sha256: str
    plan_design_sha256: str
    plan_contract_sha256: str
    registered_states: tuple[tuple[int, int], ...]
    registered_branches: tuple[tuple[str, int], ...]

    def checkpoint_prefix(self) -> dict[str, Any]:
        """Return a detached JSON-compatible checkpoint identity prefix."""

        return {
            "corner_id": self.corner_id,
            "pathway": self.pathway,
            "authority_sha256": self.authority_sha256,
            "scales": dict(self.scales),
            "model_sha256": dict(self.model_sha256),
            "atlas_source_sha256": self.atlas_source_sha256,
            "scientific_configuration_sha256": (self.scientific_configuration_sha256),
            "plan_design_sha256": self.plan_design_sha256,
            "plan_contract_sha256": self.plan_contract_sha256,
        }


def structural_checkpoint_metadata(
    identity: StructuralExecutionIdentity,
    *,
    state_slot: int,
    state: tuple[int, int],
    branch_kind: str,
    branch_slot: int,
) -> dict[str, Any]:
    """Build the exact JSON-compatible identity for one persisted checkpoint."""

    if not isinstance(identity, StructuralExecutionIdentity):
        raise TypeError("identity must be a StructuralExecutionIdentity")
    if type(state_slot) is not int or not 0 <= state_slot < len(
        identity.registered_states
    ):
        raise ValueError("state_slot must select a registered state")
    if (
        not isinstance(state, tuple)
        or len(state) != 2
        or not all(type(value) is int for value in state)
        or state != identity.registered_states[state_slot]
    ):
        raise ValueError("state must reproduce the registered state at state_slot")
    if (
        not isinstance(branch_kind, str)
        or type(branch_slot) is not int
        or (branch_kind, branch_slot) not in identity.registered_branches
    ):
        raise ValueError("branch kind and slot must select a registered branch")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **identity.checkpoint_prefix(),
        "state_slot": state_slot,
        "state": list(state),
        "branch_kind": branch_kind,
        "branch_slot": branch_slot,
    }


def validate_structural_checkpoint_metadata(
    metadata: dict[str, Any],
    identity: StructuralExecutionIdentity,
    *,
    state_slot: int,
    state: tuple[int, int],
    branch_kind: str,
    branch_slot: int,
) -> dict[str, Any]:
    """Reject any missing, extra, or altered persisted checkpoint identity."""

    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    expected = structural_checkpoint_metadata(
        identity,
        state_slot=state_slot,
        state=state,
        branch_kind=branch_kind,
        branch_slot=branch_slot,
    )
    if metadata != expected:
        differing = sorted(
            key
            for key in set(metadata) | set(expected)
            if metadata.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "structural checkpoint identity does not reproduce: " + ", ".join(differing)
        )
    return expected


def _require_configuration(
    pathway: Pathway,
    configuration: object,
) -> ArticulatedShaftAtlasConfig | ArticulatedGroundAtlasConfig:
    expected = (
        ArticulatedShaftAtlasConfig
        if pathway == "shaft"
        else ArticulatedGroundAtlasConfig
    )
    if not isinstance(configuration, expected):
        raise TypeError(f"{pathway} pathway requires {expected.__name__}")
    return configuration


def resolve_structural_execution_identity(
    authority: ArticulatedAtlasAuthority,
    *,
    corner_id: str,
    pathway: str,
    configuration: ArticulatedShaftAtlasConfig | ArticulatedGroundAtlasConfig,
    plan_path: Path = DEFAULT_OUTPUT,
) -> StructuralExecutionIdentity:
    """Reproduce and validate every immutable prefix field before execution."""

    if not isinstance(authority, ArticulatedAtlasAuthority):
        raise TypeError("authority must be an ArticulatedAtlasAuthority")
    if pathway not in {"shaft", "ground"}:
        raise ValueError("pathway must be shaft or ground")
    typed_pathway: Pathway = pathway
    typed_configuration = _require_configuration(typed_pathway, configuration)
    plan = validate_structural_propagation_plan(plan_path)
    corner = next(
        (row for row in plan["corners"] if row["corner_id"] == corner_id),
        None,
    )
    if corner is None:
        raise ValueError(f"corner_id is not registered: {corner_id}")
    provenance = authority.provenance_record()
    if provenance != corner["authority"]:
        raise RuntimeError("corner authority does not reproduce the governed plan")

    configuration_record = _configuration_record(typed_configuration)
    expected_configuration = plan["design"][f"{typed_pathway}_configuration"]
    if configuration_record != expected_configuration:
        raise RuntimeError("scientific configuration does not reproduce the plan")
    expected_identity = plan["design"]["execution_identity"][typed_pathway]
    source_hashes = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in expected_identity["atlas_source_paths"]
    }
    source_digest = _canonical_sha256(source_hashes)
    configuration_digest = scientific_configuration_sha256(typed_configuration)
    if source_digest != expected_identity["atlas_source_sha256"]:
        raise RuntimeError("atlas source set does not reproduce the plan")
    if configuration_digest != expected_identity["scientific_configuration_sha256"]:
        raise RuntimeError(
            "scientific configuration digest does not reproduce the plan"
        )

    registered_states = tuple(
        (case, sample)
        for case in typed_configuration.case_indices
        for sample in typed_configuration.sample_indices
    )
    if typed_pathway == "shaft":
        registered_branches = tuple(
            ("activation", slot) for slot in range(len(typed_configuration.activations))
        )
    else:
        registered_branches = tuple(
            [
                ("primary", slot)
                for slot in range(len(typed_configuration.ground_activations))
            ]
            + [
                ("control", slot)
                for slot in range(len(typed_configuration.control_names))
            ]
        )

    return StructuralExecutionIdentity(
        corner_id=corner_id,
        pathway=typed_pathway,
        authority_sha256=provenance["authority_sha256"],
        scales=tuple(provenance["scales"].items()),
        model_sha256=tuple(provenance["model_sha256"].items()),
        atlas_source_sha256=source_digest,
        scientific_configuration_sha256=configuration_digest,
        plan_design_sha256=plan["design_sha256"],
        plan_contract_sha256=plan["contract_sha256"],
        registered_states=registered_states,
        registered_branches=registered_branches,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "StructuralExecutionIdentity",
    "resolve_structural_execution_identity",
    "scientific_configuration_sha256",
    "structural_checkpoint_metadata",
    "validate_structural_checkpoint_metadata",
]
