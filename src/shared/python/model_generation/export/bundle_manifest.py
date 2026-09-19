"""Model bundle manifest definition and validation contracts.

Provides versioned schema binding the canonical specification, URDF XML,
dynamics sidecar, and asset hashes into an immutable, relocatable package.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


class IncompletePhysicsError(RuntimeError):
    """Raised when dynamics qualification is attempted on an incomplete model bundle."""


@dataclass
class ModelBundleManifest:
    """Versioned metadata manifest for a shared multi-engine model bundle.

    Attributes:
        schema_version: Manifest specification schema version (default: 1).
        model_sha256: Hex-encoded SHA-256 hash of the canonical JSON specification.
        urdf_sha256: Hex-encoded SHA-256 hash of the generated URDF XML document.
        sidecar_sha256: Hex-encoded SHA-256 hash of the dynamics sidecar JSON.
        nq: Number of generalized configuration coordinates.
        nv: Number of generalized velocity coordinates.
        coordinate_order: Ordered list of generalized coordinate names.
        root_type: Root mobility type (e.g., 'floating', 'fixed').
        mesh_hashes: Mapping from relative mesh path to SHA-256 hash.
        body_links: Mapping from specification body names to URDF link names.
        solid_links: Mapping from specification solid names to URDF link names.
        frame_links: Mapping from anatomical frame names to URDF frame links.
        requires_sidecar: Whether dynamics evaluation requires the sidecar.
        limit_semantics: Policy for unbounded joint limits in dynamics.
        incomplete_physics_status: Descriptive status if physics is incomplete.
        capability_losses: Documented feature losses in bare URDF export.
    """

    schema_version: int = 1
    model_sha256: str = ""
    urdf_sha256: str = ""
    sidecar_sha256: str = ""
    nq: int = 0
    nv: int = 0
    coordinate_order: list[str] = field(default_factory=list)
    root_type: str = "floating"
    mesh_hashes: dict[str, str] = field(default_factory=dict)
    body_links: dict[str, str] = field(default_factory=dict)
    solid_links: dict[str, str] = field(default_factory=dict)
    frame_links: dict[str, str] = field(default_factory=dict)
    requires_sidecar: bool = True
    limit_semantics: str = "restore-unbounded-before-dynamics"
    incomplete_physics_status: str | None = None
    capability_losses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelBundleManifest:
        """Instantiate manifest from a dictionary with schema validation.

        Preconditions:
            data must contain schema_version and coordinate_order.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for manifest, got {type(data).__name__}")
        version = data.get("schema_version")
        if version != 1:
            raise ValueError(f"Unsupported manifest schema_version: {version}")
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            model_sha256=str(data.get("model_sha256", "")),
            urdf_sha256=str(data.get("urdf_sha256", "")),
            sidecar_sha256=str(data.get("sidecar_sha256", "")),
            nq=int(data.get("nq", 0)),
            nv=int(data.get("nv", 0)),
            coordinate_order=list(data.get("coordinate_order", [])),
            root_type=str(data.get("root_type", "floating")),
            mesh_hashes=dict(data.get("mesh_hashes", {})),
            body_links=dict(data.get("body_links", {})),
            solid_links=dict(data.get("solid_links", {})),
            frame_links=dict(data.get("frame_links", {})),
            requires_sidecar=bool(data.get("requires_sidecar", True)),
            limit_semantics=str(
                data.get("limit_semantics", "restore-unbounded-before-dynamics")
            ),
            incomplete_physics_status=data.get("incomplete_physics_status"),
            capability_losses=list(data.get("capability_losses", [])),
        )

    def validate(self) -> None:
        """Validate invariant contracts on the manifest.

        Raises:
            ValueError: If manifest invariants or coordinate dimensions are inconsistent.
        """
        if self.schema_version != 1:
            raise ValueError(f"Invalid schema_version {self.schema_version}")
        if self.nq < 0 or self.nv < 0:
            raise ValueError(f"nq ({self.nq}) and nv ({self.nv}) must be non-negative")
        if len(self.coordinate_order) != self.nq:
            raise ValueError(
                f"coordinate_order length ({len(self.coordinate_order)}) does not match nq ({self.nq})"
            )
