"""Canonical-core 3D viewport provider evaluation.

This module is intentionally dependency-light. It records the Rerun, MeshCat,
and VTK evaluation for CC-33 without importing optional renderers at module
import time. Callers can select a provider, inspect degradation reasons, and
build a backend-neutral payload from the existing Trace v2 schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib.util import find_spec

import numpy as np

from src.shared.python.simulation_backends.protocol import Trace

ImportChecker = Callable[[str], bool]


class ViewportProvider(str, Enum):
    """Supported 3D viewport provider choices."""

    MESHCAT = "meshcat"
    RERUN = "rerun"
    VTK = "vtk"


class ProviderAvailability(str, Enum):
    """Dependency availability state for a viewport provider."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ViewportProviderMetadata:
    """Static metadata used to evaluate a 3D viewport provider."""

    provider: ViewportProvider
    display_name: str
    modules: tuple[str, ...]
    optional_extra: str | None
    install_hint: str
    strengths: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    supports_embedding: bool
    supports_timeline: bool
    supports_markers: bool
    supports_wrenches: bool
    selected_default: bool = False


@dataclass(frozen=True)
class ViewportProviderStatus:
    """Runtime availability status for a viewport provider."""

    metadata: ViewportProviderMetadata
    availability: ProviderAvailability
    missing_modules: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether all provider modules are import-discoverable."""

        return self.availability == ProviderAvailability.AVAILABLE

    @property
    def degradation_reason(self) -> str | None:
        """Human-readable reason the provider cannot be used now."""

        if self.available:
            return None
        missing = ", ".join(self.missing_modules)
        return (
            f"{self.metadata.display_name} is unavailable because optional "
            f"module(s) are missing: {missing}. {self.metadata.install_hint}"
        )


@dataclass(frozen=True)
class ViewportSelection:
    """Result of selecting a viewport provider."""

    selected: ViewportProviderStatus | None
    statuses: tuple[ViewportProviderStatus, ...]
    requested: ViewportProvider | None = None

    @property
    def degraded(self) -> bool:
        """Whether no available provider matched the request/default."""

        return self.selected is None

    @property
    def reason(self) -> str | None:
        """Explain why selection degraded, when it did."""

        if not self.degraded:
            return None
        if self.requested is not None:
            for status in self.statuses:
                if status.metadata.provider == self.requested:
                    return status.degradation_reason
        return "No supported 3D viewport provider is available."


@dataclass(frozen=True)
class ViewportOverlayPayload:
    """Backend-neutral render payload for canonical trajectory overlays."""

    time_s: np.ndarray
    trajectory_xyz: np.ndarray
    markers_xyz: np.ndarray | None = None
    marker_names: tuple[str, ...] = ()
    contact_points_xyz: np.ndarray | None = None
    wrench: np.ndarray | None = None
    convention: str = "canonical-v2"
    frame: str = "world_Zup"
    units: str = "SI"
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "time_s", np.asarray(self.time_s, dtype=float).reshape(-1)
        )
        object.__setattr__(
            self,
            "trajectory_xyz",
            np.asarray(self.trajectory_xyz, dtype=float),
        )
        n = self.time_s.shape[0]
        if self.trajectory_xyz.shape != (n, 3):
            raise ValueError(
                "trajectory_xyz must have shape "
                f"({n}, 3), got {self.trajectory_xyz.shape}"
            )
        if not np.all(np.isfinite(self.trajectory_xyz)):
            raise ValueError("trajectory_xyz must contain only finite values")
        if self.markers_xyz is not None:
            markers = np.asarray(self.markers_xyz, dtype=float)
            if markers.ndim != 3 or markers.shape[0] != n or markers.shape[2] != 3:
                raise ValueError(
                    "markers_xyz must have shape "
                    f"({n}, n_markers, 3), got {markers.shape}"
                )
            if self.marker_names and len(self.marker_names) != markers.shape[1]:
                raise ValueError(
                    "marker_names length must match markers axis 1; "
                    f"got {len(self.marker_names)} names for {markers.shape[1]}"
                )
            if not np.all(np.isfinite(markers)):
                raise ValueError("markers_xyz must contain only finite values")
            object.__setattr__(self, "markers_xyz", markers)
        if self.contact_points_xyz is not None:
            contacts = np.asarray(self.contact_points_xyz, dtype=float)
            if contacts.ndim != 3 or contacts.shape[0] != n or contacts.shape[2] != 3:
                raise ValueError(
                    "contact_points_xyz must have shape "
                    f"({n}, n_contacts, 3), got {contacts.shape}"
                )
            if not np.all(np.isfinite(contacts)):
                raise ValueError("contact_points_xyz must contain only finite values")
            object.__setattr__(self, "contact_points_xyz", contacts)
        if self.wrench is not None:
            wrench = np.asarray(self.wrench, dtype=float)
            if wrench.shape != (n, 6):
                raise ValueError(f"wrench must have shape ({n}, 6), got {wrench.shape}")
            if not np.all(np.isfinite(wrench)):
                raise ValueError("wrench must contain only finite values")
            object.__setattr__(self, "wrench", wrench)
        if not self.convention:
            raise ValueError("convention must be non-empty")
        if self.frame != "world_Zup":
            raise ValueError("ViewportOverlayPayload requires frame='world_Zup'")
        if self.units != "SI":
            raise ValueError("ViewportOverlayPayload requires units='SI'")

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        marker_names: Sequence[str] = (),
    ) -> ViewportOverlayPayload:
        """Build a viewport payload from the shared Trace v2 schema."""

        if trace is None:
            raise ValueError("trace must be provided")
        meta = dict(trace.meta)
        trajectory = _trajectory_from_trace(trace, meta)
        return cls(
            time_s=trace.t,
            trajectory_xyz=trajectory,
            markers_xyz=trace.markers,
            marker_names=tuple(marker_names),
            contact_points_xyz=trace.contacts,
            wrench=trace.wrench,
            convention=str(meta.get("convention", "canonical-v2")),
            frame=str(meta.get("frame", "world_Zup")),
            units=str(meta.get("units", "SI")),
            meta=meta,
        )

    @property
    def has_marker_overlay(self) -> bool:
        """Whether marker keypoints are present."""

        return self.markers_xyz is not None and self.markers_xyz.shape[1] > 0

    @property
    def has_wrench_overlay(self) -> bool:
        """Whether GRF/wrench arrows are present."""

        return self.wrench is not None


PROVIDER_METADATA: tuple[ViewportProviderMetadata, ...] = (
    ViewportProviderMetadata(
        provider=ViewportProvider.MESHCAT,
        display_name="MeshCat",
        modules=("meshcat",),
        optional_extra="pinocchio",
        install_hint="Install with `pip install -e .[pinocchio]`.",
        strengths=(
            "Already optional through the Pinocchio extra.",
            "Native fit for Drake and Pinocchio workflows.",
            "Embeddable in PyQt/Tauri through a web view.",
        ),
        tradeoffs=(
            "Less durable as an analysis log than Rerun.",
            "Requires a local browser/WebGL surface.",
        ),
        supports_embedding=True,
        supports_timeline=True,
        supports_markers=True,
        supports_wrenches=True,
        selected_default=True,
    ),
    ViewportProviderMetadata(
        provider=ViewportProvider.RERUN,
        display_name="Rerun",
        modules=("rerun",),
        optional_extra=None,
        install_hint="Install `rerun-sdk` in an opt-in visualization env.",
        strengths=(
            "Strong timeline and multimodal logging model.",
            "Good fit for recorded diagnostics and review artifacts.",
        ),
        tradeoffs=(
            "Not already used by the engine stack.",
            "Adds another viewer runtime before app-shell integration is ready.",
        ),
        supports_embedding=False,
        supports_timeline=True,
        supports_markers=True,
        supports_wrenches=True,
    ),
    ViewportProviderMetadata(
        provider=ViewportProvider.VTK,
        display_name="VTK/PyVista",
        modules=("vtk",),
        optional_extra=None,
        install_hint="Install `vtk` or use a PyVista-based tool environment.",
        strengths=(
            "Mature scientific visualization stack.",
            "Strong mesh and offscreen-rendering ecosystem.",
        ),
        tradeoffs=(
            "Heavier native dependency footprint.",
            "Less aligned with existing Drake/Pinocchio MeshCat flows.",
        ),
        supports_embedding=True,
        supports_timeline=False,
        supports_markers=True,
        supports_wrenches=True,
    ),
)


def evaluate_viewport_providers(
    import_checker: ImportChecker | None = None,
) -> tuple[ViewportProviderStatus, ...]:
    """Return availability and evaluation metadata for every provider."""

    checker = import_checker or _module_available
    statuses = []
    for metadata in PROVIDER_METADATA:
        missing = tuple(module for module in metadata.modules if not checker(module))
        availability = (
            ProviderAvailability.AVAILABLE
            if not missing
            else ProviderAvailability.UNAVAILABLE
        )
        statuses.append(
            ViewportProviderStatus(
                metadata=metadata,
                availability=availability,
                missing_modules=missing,
            )
        )
    return tuple(statuses)


def select_viewport_provider(
    preferred: ViewportProvider | str | None = None,
    *,
    import_checker: ImportChecker | None = None,
) -> ViewportSelection:
    """Select the preferred/default viewport provider if available."""

    statuses = evaluate_viewport_providers(import_checker=import_checker)
    requested = _coerce_provider(preferred)
    target = requested or ViewportProvider.MESHCAT
    for status in statuses:
        if status.metadata.provider == target and status.available:
            return ViewportSelection(
                selected=status,
                statuses=statuses,
                requested=requested,
            )
    return ViewportSelection(selected=None, statuses=statuses, requested=requested)


def selected_viewport_decision() -> ViewportProviderMetadata:
    """Return the recorded CC-33 provider decision metadata."""

    for metadata in PROVIDER_METADATA:
        if metadata.selected_default:
            return metadata
    raise ValueError("No selected default viewport provider is recorded")


def _coerce_provider(
    provider: ViewportProvider | str | None,
) -> ViewportProvider | None:
    if provider is None:
        return None
    if isinstance(provider, ViewportProvider):
        return provider
    try:
        return ViewportProvider(provider)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ViewportProvider)
        raise ValueError(
            f"unknown viewport provider {provider!r}; valid: {valid}"
        ) from exc


def _module_available(module: str) -> bool:
    return find_spec(module) is not None


def _trajectory_from_trace(trace: Trace, meta: Mapping[str, object]) -> np.ndarray:
    if trace.q.shape[1] >= 3:
        return np.asarray(trace.q[:, :3], dtype=float)
    origin = meta.get("viewport_origin_xyz")
    if origin is None:
        return np.zeros((trace.num_steps, 3), dtype=float)
    origin_xyz = np.asarray(origin, dtype=float)
    if origin_xyz.shape != (3,):
        raise ValueError("viewport_origin_xyz metadata must be a length-3 vector")
    return np.broadcast_to(origin_xyz, (trace.num_steps, 3)).copy()
