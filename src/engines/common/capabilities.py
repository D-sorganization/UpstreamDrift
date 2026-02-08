"""Engine Capabilities — standardized capability reporting for physics engines.

Every physics engine can report which optional capabilities it supports,
enabling the UI and API layers to dynamically adapt their feature set.

Design by Contract:
    Invariants:
        - Capabilities are immutable after engine initialization
        - The base REQUIRED capabilities are always True
        - Optional capabilities default to False

Usage:
    caps = engine.get_capabilities()
    if caps.has_video_export:
        exporter = engine.create_video_exporter(...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityLevel(Enum):
    """Support level for an engine capability.

    Attributes:
        FULL: Complete, production-ready implementation
        PARTIAL: Working but incomplete (e.g., placeholder contact forces)
        NONE: Not implemented
    """

    FULL = auto()
    PARTIAL = auto()
    NONE = auto()


@dataclass(frozen=True)
class EngineCapabilities:
    """Immutable capability report for a physics engine.

    Every PhysicsEngine implementation should return an instance of this
    class from its ``get_capabilities()`` method. The ``frozen=True``
    ensures capabilities cannot be mutated after engine initialization.

    Attributes:
        engine_name: Human-readable engine name (e.g., "MuJoCo")
        mass_matrix: Level of mass matrix support
        jacobian: Level of Jacobian computation support
        contact_forces: Level of contact force reporting
        inverse_dynamics: Level of inverse dynamics support
        drift_acceleration: Level of drift (ZTCF) support
        video_export: Level of video export support
        dataset_export: Level of CSV/JSON/HDF5 export support
        force_visualization: Level of force vector overlay support
        model_positioning: Level of model translate/rotate support
        measurements: Level of distance/angle measurement support
    """

    engine_name: str = ""

    # Dynamics (required by PhysicsEngine protocol, but may be partial)
    mass_matrix: CapabilityLevel = CapabilityLevel.NONE
    jacobian: CapabilityLevel = CapabilityLevel.NONE
    contact_forces: CapabilityLevel = CapabilityLevel.NONE
    inverse_dynamics: CapabilityLevel = CapabilityLevel.NONE
    drift_acceleration: CapabilityLevel = CapabilityLevel.NONE

    # Export (#1176)
    video_export: CapabilityLevel = CapabilityLevel.NONE
    dataset_export: CapabilityLevel = CapabilityLevel.NONE

    # Visualization (#1179)
    force_visualization: CapabilityLevel = CapabilityLevel.NONE
    model_positioning: CapabilityLevel = CapabilityLevel.NONE
    measurements: CapabilityLevel = CapabilityLevel.NONE

    # Extra metadata
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_video_export(self) -> bool:
        """Check if video export is available (FULL or PARTIAL)."""
        return self.video_export != CapabilityLevel.NONE

    @property
    def has_dataset_export(self) -> bool:
        """Check if dataset export is available."""
        return self.dataset_export != CapabilityLevel.NONE

    @property
    def has_force_visualization(self) -> bool:
        """Check if force vector visualization is available."""
        return self.force_visualization != CapabilityLevel.NONE

    @property
    def has_contact_forces(self) -> bool:
        """Check if contact force reporting is available."""
        return self.contact_forces != CapabilityLevel.NONE

    @property
    def has_measurements(self) -> bool:
        """Check if measurement tools are available."""
        return self.measurements != CapabilityLevel.NONE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to API-friendly dictionary.

        Returns:
            Dictionary with capability names and their levels as strings.
        """
        return {
            "engine_name": self.engine_name,
            "mass_matrix": self.mass_matrix.name.lower(),
            "jacobian": self.jacobian.name.lower(),
            "contact_forces": self.contact_forces.name.lower(),
            "inverse_dynamics": self.inverse_dynamics.name.lower(),
            "drift_acceleration": self.drift_acceleration.name.lower(),
            "video_export": self.video_export.name.lower(),
            "dataset_export": self.dataset_export.name.lower(),
            "force_visualization": self.force_visualization.name.lower(),
            "model_positioning": self.model_positioning.name.lower(),
            "measurements": self.measurements.name.lower(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineCapabilities:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with capability names and level strings.

        Returns:
            EngineCapabilities instance.
        """
        level_map = {
            "full": CapabilityLevel.FULL,
            "partial": CapabilityLevel.PARTIAL,
            "none": CapabilityLevel.NONE,
        }

        def _get_level(key: str) -> CapabilityLevel:
            raw = data.get(key, "none")
            return level_map.get(str(raw).lower(), CapabilityLevel.NONE)

        return cls(
            engine_name=data.get("engine_name", ""),
            mass_matrix=_get_level("mass_matrix"),
            jacobian=_get_level("jacobian"),
            contact_forces=_get_level("contact_forces"),
            inverse_dynamics=_get_level("inverse_dynamics"),
            drift_acceleration=_get_level("drift_acceleration"),
            video_export=_get_level("video_export"),
            dataset_export=_get_level("dataset_export"),
            force_visualization=_get_level("force_visualization"),
            model_positioning=_get_level("model_positioning"),
            measurements=_get_level("measurements"),
        )
