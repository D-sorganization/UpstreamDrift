"""Capability State Contract (ORG-02, Issue #10511).

Disentangles capability identity, maturity, per-surface availability,
and evidence-backed qualification across the UpstreamDrift catalog.

Design by Contract:
    Preconditions:
        - Unavailable surfaces must provide a non-empty reason and actionable remediation
        - Probe keys must include a valid pin or runtime identity (uncacheable targets fail)
    Postconditions:
        - Installed engines without qualified receipts never serialize release-ready
        - Display names resolve identically across native and shared shells
    Invariants:
        - Capability state objects are immutable (frozen dataclasses)
        - Legacy status serialization is preserved for backward compatibility
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import sys
from typing import Any, Final, Literal

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Canonical maturity levels
CapabilityMaturity = Literal[
    "experimental",
    "prototype",
    "beta",
    "stable",
    "deprecated",
]

VALID_MATURITY_LEVELS: Final[frozenset[str]] = frozenset(
    {"experimental", "prototype", "beta", "stable", "deprecated"}
)

# Canonical surfaces supported across shells and tools
VALID_SURFACES: Final[frozenset[str]] = frozenset({"desktop", "web", "api", "cli"})

# Authoritative known physics engine identifiers
KNOWN_PHYSICS_ENGINES: Final[frozenset[str]] = frozenset(
    {
        "mujoco",
        "pinocchio",
        "drake",
        "opensim",
        "simscape",
        "double_pendulum",
        "myosuite",
        "myosim",
        "matlab",
    }
)

# Canonical single authority for tile display names across native and shared shells
CANONICAL_TILE_DISPLAY_NAMES: Final[dict[str, str]] = {
    "matlab_suite": "Matlab Models",
    "golf_simulation_suite": "Golf Simulation Suite",
}


def resolve_canonical_display_name(tile_id: str, default: str | None = None) -> str:
    """Resolve the authoritative display name for a capability tile ID."""
    if tile_id in CANONICAL_TILE_DISPLAY_NAMES:
        return CANONICAL_TILE_DISPLAY_NAMES[tile_id]
    if default is not None and default.strip():
        return default
    return tile_id.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Per-Surface Availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurfaceAvailability:
    """Availability status and remediation instructions for a specific surface."""

    available: bool
    reason: str | None = None
    remediation: str | None = None

    def __post_init__(self) -> None:
        """DbC Invariant: unavailable surfaces must provide actionable reason and remediation."""
        if not self.available:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Unavailable surface must provide a non-empty reason")
            if not isinstance(self.remediation, str) or not self.remediation.strip():
                raise ValueError(
                    "Unavailable surface must provide an actionable remediation command or action"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for catalog and API consumers."""
        data: dict[str, Any] = {"available": self.available}
        if self.reason is not None:
            data["reason"] = self.reason
        if self.remediation is not None:
            data["remediation"] = self.remediation
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfaceAvailability:
        """Deserialize from dictionary."""
        available = bool(data.get("available", False))
        reason = data.get("reason")
        remediation = data.get("remediation")
        if not available:
            if not isinstance(reason, str) or not reason.strip():
                reason = "Surface is currently unavailable"
            if not isinstance(remediation, str) or not remediation.strip():
                remediation = (
                    "Consult documentation or launch via an alternative surface"
                )
        return cls(
            available=available,
            reason=reason,
            remediation=remediation,
        )


@dataclass(frozen=True, slots=True)
class CapabilityAvailability:
    """Per-surface availability across desktop, web, api, and cli surfaces."""

    surfaces: Mapping[str, SurfaceAvailability] = field(default_factory=dict)

    def for_surface(self, surface: str) -> SurfaceAvailability:
        """Retrieve availability for a specific surface."""
        if surface not in self.surfaces:
            raise KeyError(f"Unknown surface: {surface!r}")
        return self.surfaces[surface]

    @property
    def desktop(self) -> SurfaceAvailability:
        """PyQt6 native desktop availability."""
        return self.surfaces.get(
            "desktop",
            SurfaceAvailability(
                available=False,
                reason="Surface not declared",
                remediation="Check catalog configuration",
            ),
        )

    @property
    def web(self) -> SurfaceAvailability:
        """Browser and Tauri dashboard availability."""
        return self.surfaces.get(
            "web",
            SurfaceAvailability(
                available=False,
                reason="Surface not declared",
                remediation="Check catalog configuration",
            ),
        )

    @property
    def api(self) -> SurfaceAvailability:
        """REST/WebSocket API availability."""
        return self.surfaces.get(
            "api",
            SurfaceAvailability(
                available=False,
                reason="Surface not declared",
                remediation="Check catalog configuration",
            ),
        )

    @property
    def cli(self) -> SurfaceAvailability:
        """CLI tool availability."""
        return self.surfaces.get(
            "cli",
            SurfaceAvailability(
                available=False,
                reason="Surface not declared",
                remediation="Check catalog configuration",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {surface: state.to_dict() for surface, state in self.surfaces.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityAvailability:
        """Deserialize from dictionary."""
        surfaces = {
            surface: SurfaceAvailability.from_dict(state_data)
            for surface, state_data in data.items()
            if isinstance(state_data, dict)
        }
        return cls(surfaces=surfaces)


# ---------------------------------------------------------------------------
# Evidence-Backed Qualification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityQualification:
    """Evidence-backed qualification for physical engines and scientific tools."""

    status: str
    is_qualified: bool
    receipt_path: str | None = None
    evidence_hash: str | None = None
    failure_reasons: tuple[str, ...] = ()
    engine_result: Any | None = None

    def __post_init__(self) -> None:
        """Validate qualification state invariants."""
        if self.is_qualified and self.status in (
            "qualification_failed",
            "unsupported",
            "unqualified",
            "exempt",
        ):
            raise ValueError(
                f"Contradictory qualification state: is_qualified=True with status={self.status!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize qualification metadata."""
        data: dict[str, Any] = {
            "status": self.status,
            "is_qualified": self.is_qualified,
        }
        if self.receipt_path is not None:
            data["receipt_path"] = self.receipt_path
        if self.evidence_hash is not None:
            data["evidence_hash"] = self.evidence_hash
        if self.failure_reasons:
            data["failure_reasons"] = list(self.failure_reasons)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityQualification:
        """Deserialize from dictionary."""
        return cls(
            status=str(data.get("status", "unqualified")),
            is_qualified=bool(data.get("is_qualified", False)),
            receipt_path=data.get("receipt_path"),
            evidence_hash=data.get("evidence_hash"),
            failure_reasons=tuple(data.get("failure_reasons", [])),
        )


def adapt_engine_matrix_qualification(
    *,
    engine_name: str | None,
    is_engine: bool,
    receipt: Any | None = None,
    receipt_path: str | None = None,
    is_advertised: bool = True,
) -> CapabilityQualification:
    """Narrow adapter consuming the engine matrix qualification contract owned by #10351."""
    if not is_engine or not engine_name:
        # Non-engine tools need availability without pretending to have scientific qualification
        return CapabilityQualification(
            status="exempt",
            is_qualified=False,
            failure_reasons=(),
        )

    # Lazily import #10351 audit function to keep catalog load clean
    from src.shared.python.shadow_tracker.engine_matrix import audit_engine_conformance

    res = audit_engine_conformance(
        engine_name=engine_name,
        receipt=receipt,
        is_advertised=is_advertised,
    )
    return CapabilityQualification(
        status=res.status,
        is_qualified=res.is_qualified,
        receipt_path=receipt_path,
        failure_reasons=res.failure_reasons,
        engine_result=res,
    )


# ---------------------------------------------------------------------------
# Lazy Runtime Probing and Caching
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeProbeKey:
    """Cache key tied to provider pin or runtime identity."""

    target_id: str
    pin_or_identity: str
    env_fingerprint: str = sys.version

    def __post_init__(self) -> None:
        """DbC Precondition: uncacheable target without pin/identity fails validation."""
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("RuntimeProbeKey target_id must be non-empty")
        if (
            not isinstance(self.pin_or_identity, str)
            or not self.pin_or_identity.strip()
        ):
            raise ValueError(
                "RuntimeProbeKey pin_or_identity must be non-empty (uncacheable target fails probe contract)"
            )


class RuntimeProbeCache:
    """In-memory thread-safe memoization cache for lazy runtime availability probes."""

    def __init__(self) -> None:
        self._cache: dict[RuntimeProbeKey, Any] = {}

    def get(self, key: RuntimeProbeKey) -> Any | None:
        """Retrieve cached result or None."""
        return self._cache.get(key)

    def set(self, key: RuntimeProbeKey, value: Any) -> None:
        """Cache probe result for the given runtime pin identity."""
        self._cache[key] = value

    def has_key(self, key: RuntimeProbeKey) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def clear(self) -> None:
        """Invalidate all cached probe results."""
        self._cache.clear()


# Process-wide runtime probe cache instance
GLOBAL_PROBE_CACHE = RuntimeProbeCache()
