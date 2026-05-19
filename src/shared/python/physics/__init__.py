"""Physics engines, aerodynamics, terrain, and impact models."""

from src.shared.python.physics.dimple_geometry import DimpleGeometry, dimple_adjusted_cd
from src.shared.python.physics.mud_ball import (
    MudBallAdjustment,
    mud_ball_aero_adjustments,
)
from src.shared.python.physics.swing_ball_flight_pipeline import (
    FlightSimulatorProtocol,
    PipelineResult,
    SwingBallFlightPipeline,
    SwingState,
)
from src.shared.python.physics.water_hazard import (
    WaterEntryResult,
    water_entry_kinematics,
)

__all__: list[str] = [
    "mud_ball_aero_adjustments",
    "MudBallAdjustment",
    "water_entry_kinematics",
    "WaterEntryResult",
    "DimpleGeometry",
    "dimple_adjusted_cd",
    # Swing-to-flight pipeline (Issue #5337)
    "FlightSimulatorProtocol",
    "PipelineResult",
    "SwingBallFlightPipeline",
    "SwingState",
]
