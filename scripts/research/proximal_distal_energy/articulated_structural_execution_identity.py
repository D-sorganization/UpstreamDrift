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
    configuration_digest = _canonical_sha256(configuration_record)
    if source_digest != expected_identity["atlas_source_sha256"]:
        raise RuntimeError("atlas source set does not reproduce the plan")
    if configuration_digest != expected_identity["scientific_configuration_sha256"]:
        raise RuntimeError(
            "scientific configuration digest does not reproduce the plan"
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
    )


__all__ = [
    "StructuralExecutionIdentity",
    "resolve_structural_execution_identity",
]
