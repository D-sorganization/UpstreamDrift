"""Unified artifact and project context handoff between workspaces (#10517).

Provides versioned handoff objects, typed artifact references with cryptographic
hash verification, schema validation, and registered adapter conversions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Final

_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ArtifactKind(str, Enum):
    """Canonical artifact contract kinds across workspaces."""

    STATIC_POSE = "static_pose"
    DYNAMIC_STATE = "dynamic_state"
    OBSERVATION = "observation"
    TRAJECTORY = "trajectory"
    RECEIPT = "receipt"


SUPPORTED_KINDS: Final[frozenset[str]] = frozenset(
    {kind.value for kind in ArtifactKind}
)

SUPPORTED_FRAMES: Final[frozenset[str]] = frozenset(
    {
        "world",
        "ground",
        "pelvis",
        "base",
        "canonical",
        "model",
        "flight_xfwd_yleft_zup",
        "app_xtarget_yup_zright",
    }
)

SUPPORTED_SCHEMAS: Final[frozenset[str]] = frozenset(
    {
        "pose_interchange/canonical/1",
        "swing_sim.ball_flight_trajectory/1",
        "simulation_backend.trace/2.1.0",
        "motion_capture.c3d/1",
        "pipeline.ground_support_receipt/1",
        "workspace.handoff/1.0.0",
        "dataset/c3d",
        "dataset/h5",
    }
)

SUPPORTED_HANDOFF_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"1.0.0"})

VALID_STATUSES: Final[frozenset[str]] = frozenset(
    {"draft", "in_progress", "completed", "qualified", "failed", "canceled"}
)


def _utc_now() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    iso_text = timestamp.isoformat()
    return iso_text.replace("+00:00", "Z")


def compute_file_sha256(path: Path | str) -> str:
    """Compute cryptographic SHA-256 hash formatted as ``sha256:<hex>``."""
    path_obj = Path(path)
    hasher = hashlib.sha256()
    with path_obj.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class ArtifactReference:
    """A typed reference to an on-disk artifact payload with schema and hash."""

    artifact_id: str
    path: str
    hash: str
    schema: str
    kind: ArtifactKind | str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_id, str)
            or _ID_RE.fullmatch(self.artifact_id) is None
        ):
            raise ValueError(
                f"artifact_id must match {_ID_RE.pattern}, got {self.artifact_id!r}"
            )
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("path must be a non-empty string or Path")
        if not isinstance(self.hash, str) or not self.hash.strip():
            raise ValueError("hash must be a non-empty string")
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("schema must be a non-empty string")

        raw_kind = (
            self.kind.value if isinstance(self.kind, ArtifactKind) else str(self.kind)
        )
        if raw_kind not in SUPPORTED_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(SUPPORTED_KINDS)}, got {raw_kind!r}"
            )
        if not isinstance(self.kind, ArtifactKind):
            object.__setattr__(self, "kind", ArtifactKind(raw_kind))

    def resolve_path(self, root: Path | str | None = None) -> Path:
        """Resolve path relative to root if relative."""
        p = Path(self.path)
        if p.is_absolute() or root is None:
            return p
        return Path(root) / p

    def verify_on_disk(self, root: Path | str | None = None) -> None:
        """Verify the referenced file exists and matches the recorded hash."""
        resolved = self.resolve_path(root)
        if not resolved.exists() or not resolved.is_file():
            kind_str = (
                self.kind.value
                if isinstance(self.kind, ArtifactKind)
                else str(self.kind)
            )
            raise FileNotFoundError(
                f"Artifact '{self.artifact_id}' ({kind_str}) file is missing at '{resolved}'. "
                f"No substitute was invented."
            )
        actual_hash = compute_file_sha256(resolved)
        expected_hash = self.hash
        # Support comparison whether prefix sha256: is present or not
        exp_clean = expected_hash.removeprefix("sha256:").lower()
        act_clean = actual_hash.removeprefix("sha256:").lower()
        if exp_clean != act_clean:
            raise ValueError(
                f"Artifact '{self.artifact_id}' hash mismatch at '{resolved}': "
                f"expected {expected_hash}, got {actual_hash}"
            )


@dataclass(frozen=True)
class WorkspaceHandoff:
    """Versioned project context handoff carrying parameters and artifact references."""

    handoff_id: str
    project_id: str
    session_id: str
    subject_id: str
    engine: str
    model_id: str
    club: dict[str, Any]
    units: dict[str, str]
    frame: str
    timebase: dict[str, Any]
    parameters: dict[str, Any]
    inputs: tuple[ArtifactReference, ...] = ()
    outputs: tuple[ArtifactReference, ...] = ()
    status: str = "draft"
    qualification: dict[str, Any] | None = None
    engine_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.handoff_id, str)
            or _ID_RE.fullmatch(self.handoff_id) is None
        ):
            raise ValueError(
                f"handoff_id must match {_ID_RE.pattern}, got {self.handoff_id!r}"
            )
        if (
            not isinstance(self.project_id, str)
            or _ID_RE.fullmatch(self.project_id) is None
        ):
            raise ValueError(
                f"project_id must match {_ID_RE.pattern}, got {self.project_id!r}"
            )
        if (
            not isinstance(self.session_id, str)
            or _ID_RE.fullmatch(self.session_id) is None
        ):
            raise ValueError(
                f"session_id must match {_ID_RE.pattern}, got {self.session_id!r}"
            )
        if (
            not isinstance(self.subject_id, str)
            or _ID_RE.fullmatch(self.subject_id) is None
        ):
            raise ValueError(
                f"subject_id must match {_ID_RE.pattern}, got {self.subject_id!r}"
            )

        if self.schema_version not in SUPPORTED_HANDOFF_SCHEMA_VERSIONS:
            raise ValueError(
                f"unknown or unsupported schema version: {self.schema_version}; "
                f"supported versions are {sorted(SUPPORTED_HANDOFF_SCHEMA_VERSIONS)}"
            )

        if self.frame not in SUPPORTED_FRAMES:
            raise ValueError(
                f"unknown or unsupported frame: {self.frame!r}; "
                f"supported frames are {sorted(SUPPORTED_FRAMES)}"
            )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(VALID_STATUSES)}, got {self.status!r}"
            )

        # Invariant: canceled or failed jobs never become completed results
        if self.status in {"failed", "canceled"} and self.qualification is not None:
            if self.qualification.get("passed", False):
                raise ValueError(
                    f"{self.status} run cannot be marked with completed qualification"
                )

        # Normalize inputs and outputs to tuples
        if not isinstance(self.inputs, tuple):
            object.__setattr__(self, "inputs", tuple(self.inputs))
        if not isinstance(self.outputs, tuple):
            object.__setattr__(self, "outputs", tuple(self.outputs))

        # Check artifact schemas
        for art in (*self.inputs, *self.outputs):
            if art.schema not in SUPPORTED_SCHEMAS:
                raise ValueError(
                    f"unknown or unsupported schema: {art.schema!r}; "
                    f"supported schemas are {sorted(SUPPORTED_SCHEMAS)}"
                )


# ============================================================================
# Named Artifact Adapters
# ============================================================================

_ADAPTER_REGISTRY: dict[
    str, tuple[str, str, Callable[[Path, Path, dict[str, Any]], None]]
] = {}


def register_artifact_adapter(
    name: str,
    source_kind: ArtifactKind | str,
    target_kind: ArtifactKind | str,
    adapter_fn: Callable[[Path, Path, dict[str, Any]], None],
) -> None:
    """Register a named adapter for explicit artifact kind conversion."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("adapter name must be a non-empty string")
    src = (
        source_kind.value if isinstance(source_kind, ArtifactKind) else str(source_kind)
    )
    dst = (
        target_kind.value if isinstance(target_kind, ArtifactKind) else str(target_kind)
    )
    if src not in SUPPORTED_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(SUPPORTED_KINDS)}")
    if dst not in SUPPORTED_KINDS:
        raise ValueError(f"target_kind must be one of {sorted(SUPPORTED_KINDS)}")
    if not callable(adapter_fn):
        raise TypeError("adapter_fn must be callable")
    _ADAPTER_REGISTRY[name] = (src, dst, adapter_fn)


def convert_artifact(
    source: ArtifactReference,
    target_kind: ArtifactKind | str,
    *,
    adapter_name: str,
    output_path: Path | str,
    provenance: dict[str, Any],
    artifact_id: str,
    schema: str,
    root: Path | str | None = None,
) -> ArtifactReference:
    """Convert an artifact across distinct contracts using a registered adapter."""
    if adapter_name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"unknown or unregistered adapter: {adapter_name!r}; "
            f"registered adapters are {sorted(_ADAPTER_REGISTRY.keys())}"
        )
    src_kind, dst_kind, adapter_fn = _ADAPTER_REGISTRY[adapter_name]
    target_k = (
        target_kind.value if isinstance(target_kind, ArtifactKind) else str(target_kind)
    )
    source_k = (
        source.kind.value if isinstance(source.kind, ArtifactKind) else str(source.kind)
    )

    if source_k != src_kind:
        raise ValueError(
            f"adapter {adapter_name!r} expects source kind {src_kind!r}, got {source_k!r}"
        )
    if target_k != dst_kind:
        raise ValueError(
            f"adapter {adapter_name!r} produces target kind {dst_kind!r}, got {target_k!r}"
        )
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError(
            "provenance must be non-empty dictionary with recorded provenance"
        )

    src_resolved = source.resolve_path(root)
    source.verify_on_disk(root)

    out_p = Path(output_path)
    dst_resolved = out_p if out_p.is_absolute() or root is None else Path(root) / out_p
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)

    adapter_fn(src_resolved, dst_resolved, provenance)

    if not dst_resolved.exists():
        raise FileNotFoundError(
            f"adapter {adapter_name} failed to produce output at {dst_resolved}"
        )

    out_hash = compute_file_sha256(dst_resolved)
    metadata = {
        "conversion_provenance": {
            "adapter": adapter_name,
            "source_artifact_id": source.artifact_id,
            "source_kind": source_k,
            "source_hash": source.hash,
            **provenance,
        }
    }

    return ArtifactReference(
        artifact_id=artifact_id,
        path=str(output_path),
        hash=out_hash,
        schema=schema,
        kind=target_k,
        metadata=metadata,
    )
