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
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, TypeAlias

logger = logging.getLogger(__name__)

SPATIAL_JACOBIAN_ORDER = ("angular", "linear")
"""Canonical suite row order for ``jacobian["spatial"]``.

Engine wrappers expose separate ``linear`` and ``angular`` matrices and a
combined ``spatial`` matrix. The combined matrix is stacked as
``[angular; linear]`` for compatibility with Drake spatial velocity ordering.
Wrappers whose native API returns ``[linear; angular]`` must restack before
returning ``spatial``.
"""


class Capability(str, Enum):
    """Canonical engine capability identifiers.

    The enum is the stable query key shared across engine-core capability
    reports and narrower adapter-specific descriptors.
    """

    MASS_MATRIX = "mass_matrix"
    JACOBIAN = "jacobian"
    CONTACT_FORCES = "contact_forces"
    INVERSE_DYNAMICS = "inverse_dynamics"
    DRIFT_ACCELERATION = "drift_acceleration"
    PARAMETER_GRADIENTS = "parameter_gradients"
    STATE_CONTROL_GRADIENTS = "state_control_gradients"
    FORWARD_SIM = "forward_sim"
    CONTACT_STEP = "contact_step"
    TRAJECTORY_OPT = "trajectory_opt"
    VIDEO_EXPORT = "video_export"
    DATASET_EXPORT = "dataset_export"
    FORCE_VISUALIZATION = "force_visualization"
    MODEL_POSITIONING = "model_positioning"
    MEASUREMENTS = "measurements"

    # Adapter-boundary capabilities. These are queryable through the same
    # contract without forcing every engine report to grow backend-only fields.
    BATCHED_ROLLOUT = "batched_rollout"
    DIFFERENTIABLE_ROLLOUT = "differentiable_rollout"
    DYNAMICS_PRIMITIVES = "dynamics_primitives"


CapabilityRef: TypeAlias = Capability | str


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


class CapabilityQuery(Protocol):
    """Shared query contract for capability descriptors."""

    def level_for(self, capability: CapabilityRef) -> CapabilityLevel:
        """Return the advertised support level for ``capability``."""
        ...

    def supports(
        self,
        capability: CapabilityRef,
        *,
        minimum: CapabilityLevel = CapabilityLevel.PARTIAL,
    ) -> bool:
        """Return whether ``capability`` meets ``minimum`` support."""
        ...


ENGINE_CAPABILITY_FIELDS: Mapping[Capability, str] = {
    Capability.MASS_MATRIX: "mass_matrix",
    Capability.JACOBIAN: "jacobian",
    Capability.CONTACT_FORCES: "contact_forces",
    Capability.INVERSE_DYNAMICS: "inverse_dynamics",
    Capability.DRIFT_ACCELERATION: "drift_acceleration",
    Capability.PARAMETER_GRADIENTS: "parameter_gradients",
    Capability.STATE_CONTROL_GRADIENTS: "state_control_gradients",
    Capability.FORWARD_SIM: "forward_sim",
    Capability.CONTACT_STEP: "contact_step",
    Capability.TRAJECTORY_OPT: "trajectory_opt",
    Capability.VIDEO_EXPORT: "video_export",
    Capability.DATASET_EXPORT: "dataset_export",
    Capability.FORCE_VISUALIZATION: "force_visualization",
    Capability.MODEL_POSITIONING: "model_positioning",
    Capability.MEASUREMENTS: "measurements",
}
"""Capabilities stored directly on :class:`EngineCapabilities`."""

ADAPTER_BOUNDARY_CAPABILITIES = (
    Capability.BATCHED_ROLLOUT,
    Capability.DIFFERENTIABLE_ROLLOUT,
    Capability.DYNAMICS_PRIMITIVES,
)
"""Canonical capabilities answered by narrower backend adapter descriptors."""

_CAPABILITY_ALIASES: Mapping[str, Capability] = {
    "batched": Capability.BATCHED_ROLLOUT,
    "batched_rollouts": Capability.BATCHED_ROLLOUT,
    "supports_batched": Capability.BATCHED_ROLLOUT,
    "differentiable": Capability.DIFFERENTIABLE_ROLLOUT,
    "is_differentiable": Capability.DIFFERENTIABLE_ROLLOUT,
    "provides_dynamics": Capability.DYNAMICS_PRIMITIVES,
}

_CAPABILITY_LEVEL_RANK: Mapping[CapabilityLevel, int] = {
    CapabilityLevel.NONE: 0,
    CapabilityLevel.PARTIAL: 1,
    CapabilityLevel.FULL: 2,
}


def normalize_capability(capability: CapabilityRef) -> Capability:
    """Return the canonical :class:`Capability` for ``capability``.

    Existing backend flag names such as ``"supports_batched"`` are accepted as
    aliases so callers can migrate to the enum without a hard flag day.
    """
    if isinstance(capability, Capability):
        return capability

    key = str(capability).strip().lower()
    if key in _CAPABILITY_ALIASES:
        return _CAPABILITY_ALIASES[key]
    try:
        return Capability(key)
    except ValueError as exc:
        valid = ", ".join(c.value for c in Capability)
        raise ValueError(
            f"unknown capability {capability!r}; expected one of {valid}"
        ) from exc


def capability_level_supported(
    level: CapabilityLevel,
    *,
    minimum: CapabilityLevel = CapabilityLevel.PARTIAL,
) -> bool:
    """Return whether ``level`` satisfies the requested support threshold."""
    return _CAPABILITY_LEVEL_RANK[level] >= _CAPABILITY_LEVEL_RANK[minimum]


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
        parameter_gradients: Level of model-parameter gradient support
        state_control_gradients: Level of state/control gradient support
        forward_sim: Level of forward simulation support
        contact_step: Level of contact-aware step support
        trajectory_opt: Level of trajectory optimization support
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
    parameter_gradients: CapabilityLevel = CapabilityLevel.NONE
    state_control_gradients: CapabilityLevel = CapabilityLevel.NONE
    forward_sim: CapabilityLevel = CapabilityLevel.NONE
    contact_step: CapabilityLevel = CapabilityLevel.NONE
    trajectory_opt: CapabilityLevel = CapabilityLevel.NONE

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
    def has_parameter_gradients(self) -> bool:
        """Check if model-parameter gradients are available."""
        return self.parameter_gradients != CapabilityLevel.NONE

    @property
    def has_state_control_gradients(self) -> bool:
        """Check if state/control gradients are available."""
        return self.state_control_gradients != CapabilityLevel.NONE

    @property
    def has_forward_sim(self) -> bool:
        """Check if forward simulation is available."""
        return self.forward_sim != CapabilityLevel.NONE

    @property
    def has_contact_step(self) -> bool:
        """Check if contact-aware stepping is available."""
        return self.contact_step != CapabilityLevel.NONE

    @property
    def has_trajectory_opt(self) -> bool:
        """Check if trajectory optimization support is available."""
        return self.trajectory_opt != CapabilityLevel.NONE

    @property
    def has_measurements(self) -> bool:
        """Check if measurement tools are available."""
        return self.measurements != CapabilityLevel.NONE

    def level_for(self, capability: CapabilityRef) -> CapabilityLevel:
        """Return the support level for a canonical capability.

        Adapter-boundary capabilities are known query keys but are not stored on
        this engine-core report, so they answer ``NONE`` here.
        """
        normalized = normalize_capability(capability)
        field_name = ENGINE_CAPABILITY_FIELDS.get(normalized)
        if field_name is None:
            return CapabilityLevel.NONE
        return getattr(self, field_name)

    def supports(
        self,
        capability: CapabilityRef,
        *,
        minimum: CapabilityLevel = CapabilityLevel.PARTIAL,
    ) -> bool:
        """Return whether ``capability`` is supported at ``minimum`` level."""
        return capability_level_supported(
            self.level_for(capability),
            minimum=minimum,
        )

    def to_capability_map(self) -> dict[Capability, CapabilityLevel]:
        """Return a complete canonical capability-to-level mapping."""
        return {capability: self.level_for(capability) for capability in Capability}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to API-friendly dictionary.

        Returns:
            Dictionary with capability names and their levels as strings.
        """
        data = {
            "engine_name": self.engine_name,
            "spatial_jacobian_order": "_".join(SPATIAL_JACOBIAN_ORDER),
        }
        data.update(
            {
                capability.value: self.level_for(capability).name.lower()
                for capability in ENGINE_CAPABILITY_FIELDS
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineCapabilities:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with capability names and level strings.

        Returns:
            EngineCapabilities instance.
        """
        if data is None:
            raise ValueError("data must be provided")
        level_map = {
            "full": CapabilityLevel.FULL,
            "partial": CapabilityLevel.PARTIAL,
            "none": CapabilityLevel.NONE,
        }

        def _get_level(key: str) -> CapabilityLevel:
            raw = data.get(key, "none")
            return level_map.get(str(raw).lower(), CapabilityLevel.NONE)

        levels: dict[str, Any] = {
            field_name: _get_level(capability.value)
            for capability, field_name in ENGINE_CAPABILITY_FIELDS.items()
        }

        return cls(
            engine_name=data.get("engine_name", ""),
            **levels,
        )


__all__ = [
    "ADAPTER_BOUNDARY_CAPABILITIES",
    "ENGINE_CAPABILITY_FIELDS",
    "SPATIAL_JACOBIAN_ORDER",
    "Capability",
    "CapabilityLevel",
    "CapabilityQuery",
    "CapabilityRef",
    "EngineCapabilities",
    "capability_level_supported",
    "normalize_capability",
]
