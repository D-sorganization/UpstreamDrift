"""Common Engine Components Package.

This package provides reusable components shared across all physics engine
implementations, reducing code duplication by 15-20%.

Modules:
    physics: Common physics equations (drag, lift, magnus effect)
    state: State management utilities and mixins
    capabilities: Engine capability reporting
    export: Video and dataset export interfaces
    simulation_control: Simulation lifecycle and tools

Usage:
    from src.engines.common import (
        BallPhysics,
        StateManager,
        EngineStateMixin,
        EngineCapabilities,
        CapabilityLevel,
        DatasetExporter,
    )
"""

from src.shared.python.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.engines.common.export import (
    DatasetExporter,
    DatasetRecord,
    VideoConfig,
    VideoExportProtocol,
)
from src.engines.common.physics import AerodynamicsCalculator, BallPhysics
from src.engines.common.simulation_control import (
    ForceOverlay,
    MeasurementResult,
    SimulationController,
    SimulationMode,
)
from src.engines.common.state import (
    EngineStateMixin,
    ForceAccumulator,
    StateManager,
)

__all__ = [
    "AerodynamicsCalculator",
    "BallPhysics",
    "CapabilityLevel",
    "DatasetExporter",
    "DatasetRecord",
    "EngineCapabilities",
    "EngineStateMixin",
    "ForceAccumulator",
    "ForceOverlay",
    "MeasurementResult",
    "SimulationController",
    "SimulationMode",
    "StateManager",
    "VideoConfig",
    "VideoExportProtocol",
]
