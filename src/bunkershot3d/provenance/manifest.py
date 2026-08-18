"""Run manifest for BunkerShot3D result artifacts (issue #8617, finding B18).

A result file that cannot say what produced it is not an audit artifact. The
manifest answers: which configuration (hashed two ways), which RNG streams,
which library versions, which commit and was the tree dirty, which solver and
fidelity tier, was the result inside the solver's validity envelope, how long it
took, and on which host.

It is persisted twice, on purpose:

* as HDF5 root attributes, so the manifest cannot be separated from the data;
* as a sibling ``<artifact>.provenance.json``, so it is greppable without h5py
  and matches the existing repository convention (see
  :mod:`src.shared.python.data_io.provenance`).
"""

from __future__ import annotations

import hashlib
import json
import platform as platform_mod
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .rng import SeedRecord

__all__ = [
    "MANIFEST_ATTR_PREFIX",
    "PROVENANCE_SUFFIX",
    "RunManifest",
    "Validity",
    "library_versions",
]

#: Suffix appended to the artifact filename for the sibling JSON manifest.
PROVENANCE_SUFFIX = ".provenance.json"

#: Prefix applied to every manifest entry stored as an HDF5 root attribute.
MANIFEST_ATTR_PREFIX = "manifest_"

#: Manifest members stored as JSON strings rather than HDF5 scalars.
_JSON_FIELDS = ("seeds", "library_versions")


class Validity(str, Enum):
    """Verdict on whether the result may be used as an answer.

    A solver run outside its calibrated envelope must say so rather than return
    a plausible number (ADR-0032).
    """

    VALID = "valid"
    OUT_OF_ENVELOPE = "out_of_envelope"
    INVALID = "invalid"
    UNKNOWN = "unknown"


def library_versions() -> dict[str, str]:
    """Return versions of the libraries whose behaviour can change results.

    Returns:
        Mapping of distribution name to version string; absent optional
        packages are simply omitted.
    """
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "h5py", "pydantic", "mujoco"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _decode(value: Any) -> Any:
    """Return HDF5 attribute ``value`` as a native Python scalar."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return item()
    return value


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable provenance record attached to one result artifact."""

    config_hash: str
    physics_hash: str
    seeds: tuple[SeedRecord, ...]
    solver: str
    fidelity_tier: str
    validity: Validity
    validity_reason: str = ""
    library_versions: Mapping[str, str] = field(default_factory=dict)
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    started_at_utc: str = ""
    wall_clock_s: float = 0.0

    def __post_init__(self) -> None:
        """Validate the record.

        Raises:
            ValueError: If no seed is recorded (an unrecorded stream makes the
                run unreproducible) or ``validity`` is not a :class:`Validity`.
        """
        if not self.seeds:
            raise ValueError(
                "a run manifest must record at least one RNG seed; "
                "unrecorded seeds make the run unreproducible"
            )
        object.__setattr__(self, "seeds", tuple(self.seeds))
        try:
            verdict = Validity(self.validity)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in Validity)
            raise ValueError(
                f"validity must be one of {allowed}, got {self.validity!r}"
            ) from exc
        object.__setattr__(self, "validity", verdict)
        object.__setattr__(self, "library_versions", dict(self.library_versions))

    # -- construction -----------------------------------------------------

    @classmethod
    def capture(
        cls,
        *,
        config_hash: str,
        physics_hash: str,
        seeds: tuple[SeedRecord, ...],
        solver: str,
        fidelity_tier: str,
        validity: Validity | str,
        validity_reason: str = "",
        wall_clock_s: float = 0.0,
    ) -> RunManifest:
        """Build a manifest, filling the environment fields from this process.

        Git, interpreter and library details are captured through the shared
        :class:`~src.shared.python.data_io.provenance.ProvenanceInfo` helper so
        there is one implementation of "which commit is this".

        Args:
            config_hash: Digest over the whole configuration.
            physics_hash: Digest over the physics-relevant configuration.
            seeds: Every RNG stream used by the run.
            solver: Solver identifier (``"drft"``, ``"mujoco"``, ...).
            fidelity_tier: Fidelity tier per ADR-0032 (``"F0"`` ... ``"F3"``).
            validity: Verdict on the result's usability.
            validity_reason: Human-readable justification for the verdict.
            wall_clock_s: Measured run duration in seconds.

        Returns:
            A fully populated manifest.
        """
        from src.shared.python.data_io.provenance import ProvenanceInfo

        info = ProvenanceInfo.capture()
        return cls(
            config_hash=config_hash,
            physics_hash=physics_hash,
            seeds=tuple(seeds),
            solver=solver,
            fidelity_tier=fidelity_tier,
            validity=Validity(validity),
            validity_reason=validity_reason,
            library_versions=library_versions(),
            git_commit=info.git_commit_sha or "",
            git_branch=info.git_branch or "",
            git_dirty=bool(info.git_is_dirty),
            python_version=info.python_version or platform_mod.python_version(),
            platform=platform_mod.platform(),
            hostname=socket.gethostname(),
            started_at_utc=info.timestamp_utc,
            wall_clock_s=float(wall_clock_s),
        )

    def with_wall_clock(self, wall_clock_s: float) -> RunManifest:
        """Return a copy of this manifest with ``wall_clock_s`` replaced.

        Args:
            wall_clock_s: Measured run duration in seconds.

        Returns:
            A new manifest; the original is unchanged.
        """
        return replace(self, wall_clock_s=float(wall_clock_s))

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the manifest."""
        return {
            "config_hash": self.config_hash,
            "physics_hash": self.physics_hash,
            "seeds": [record.to_dict() for record in self.seeds],
            "solver": self.solver,
            "fidelity_tier": self.fidelity_tier,
            "validity": self.validity.value,
            "validity_reason": self.validity_reason,
            "library_versions": dict(self.library_versions),
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": bool(self.git_dirty),
            "python_version": self.python_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "started_at_utc": self.started_at_utc,
            "wall_clock_s": float(self.wall_clock_s),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunManifest:
        """Rebuild a manifest from :meth:`to_dict` output.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed manifest.

        Raises:
            KeyError: If a required member is missing.
            ValueError: If the payload fails manifest validation.
        """
        return cls(
            config_hash=str(payload["config_hash"]),
            physics_hash=str(payload["physics_hash"]),
            seeds=tuple(SeedRecord.from_dict(dict(item)) for item in payload["seeds"]),
            solver=str(payload["solver"]),
            fidelity_tier=str(payload["fidelity_tier"]),
            validity=Validity(payload["validity"]),
            validity_reason=str(payload.get("validity_reason", "")),
            library_versions=dict(payload.get("library_versions", {})),
            git_commit=str(payload.get("git_commit", "")),
            git_branch=str(payload.get("git_branch", "")),
            git_dirty=bool(payload.get("git_dirty", False)),
            python_version=str(payload.get("python_version", "")),
            platform=str(payload.get("platform", "")),
            hostname=str(payload.get("hostname", "")),
            started_at_utc=str(payload.get("started_at_utc", "")),
            wall_clock_s=float(payload.get("wall_clock_s", 0.0)),
        )

    # -- persistence ------------------------------------------------------

    def write_attrs(self, group: Any) -> None:
        """Write the manifest onto an HDF5 group as ``manifest_*`` attributes.

        Args:
            group: An open :class:`h5py.Group` (typically the file root).

        Postconditions:
            ``RunManifest.read_attrs(group)`` returns an equal manifest.
        """
        payload = self.to_dict()
        for key, value in payload.items():
            attr = f"{MANIFEST_ATTR_PREFIX}{key}"
            if key in _JSON_FIELDS:
                group.attrs[attr] = json.dumps(value, sort_keys=True)
            else:
                group.attrs[attr] = value

    @classmethod
    def read_attrs(cls, group: Any) -> RunManifest | None:
        """Read a manifest previously written by :meth:`write_attrs`.

        Args:
            group: An open :class:`h5py.Group`.

        Returns:
            The manifest, or ``None`` when the group carries no manifest.
        """
        key = f"{MANIFEST_ATTR_PREFIX}config_hash"
        if key not in group.attrs:
            return None
        payload: dict[str, Any] = {}
        for attr in group.attrs:
            if not attr.startswith(MANIFEST_ATTR_PREFIX):
                continue
            name = attr[len(MANIFEST_ATTR_PREFIX) :]
            value = _decode(group.attrs[attr])
            payload[name] = json.loads(value) if name in _JSON_FIELDS else value
        return cls.from_dict(payload)

    def write_sidecar(
        self,
        artifact_path: Path | str,
        *,
        artifact_format: str = "hdf5",
        artifact_extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write ``<artifact>.provenance.json`` next to the artifact.

        Args:
            artifact_path: Path of the result file the manifest describes. The
                file must already be closed so its checksum is stable.
            artifact_format: Format label recorded for the artifact.
            artifact_extra: Extra artifact-level entries (e.g. schema version).

        Returns:
            The path of the sidecar file written.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"cannot checksum a missing artifact: {path}")
        artifact: dict[str, Any] = {
            "path": path.name,
            "format": artifact_format,
            "checksum_algorithm": "sha256",
            "checksum_sha256": _file_sha256(path),
        }
        if artifact_extra:
            artifact.update(dict(artifact_extra))
        document = {"artifact": artifact, "run_manifest": self.to_dict()}
        sidecar = path.parent / f"{path.name}{PROVENANCE_SUFFIX}"
        sidecar.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return sidecar


def _file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
