"""Configuration Manager for Golf Modeling Suite.

Handles validation, loading, and saving of simulation configurations.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.python.common_utils import GolfModelingError


@dataclass
class SimulationConfig:
    """Robust configuration dataclass with defaults."""

    # Dimensions
    height_m: float = 1.8
    weight_percent: float = 100.0

    # Equipment
    club_length: float = 1.0
    club_mass: float = 0.5

    # Simulation
    control_mode: str = "pd"  # pd, lqr, poly
    live_view: bool = False
    simulation_duration: float = 3.0

    # Advanced Features
    two_handed: bool = False
    enhance_face: bool = False
    articulated_fingers: bool = False

    # State Persistence
    save_state_path: str = ""
    load_state_path: str = ""

    # Polynomial Control
    polynomial_coefficients: dict[str, list[float]] = field(default_factory=dict)

    # Visualization
    show_contact_forces: bool = True
    show_joint_torques: bool = True
    show_tracers: bool = True
    tracer_bodies: list[str] = field(
        default_factory=lambda: ["pelvis", "torso", "head", "r_hand", "l_hand"]
    )

    # Appearance (Colors)
    colors: dict[str, list[float]] = field(
        default_factory=lambda: {
            "shirt": [0.6, 0.6, 0.6, 1.0],
            "pants": [0.4, 0.2, 0.0, 1.0],
            "shoes": [0.1, 0.1, 0.1, 1.0],
            "skin": [0.8, 0.6, 0.4, 1.0],
            "eyes": [1.0, 1.0, 1.0, 1.0],
            "club": [0.8, 0.8, 0.8, 1.0],
        }
    )

    def validate(self) -> None:
        """Validate configuration values."""
        if self.height_m <= 0:
            raise GolfModelingError("height_m must be positive")
        if self.weight_percent <= 0:
            raise GolfModelingError("weight_percent must be positive")
        if self.club_length <= 0:
            raise GolfModelingError("club_length must be positive")
        if self.control_mode not in ["pd", "lqr", "poly"]:
            raise GolfModelingError(f"Invalid control_mode: {self.control_mode}")


class ConfigurationManager:
    """Manages loading and saving of simulation configurations."""

    def __init__(self, config_path: Path) -> None:
        """Initialize with path to config file."""
        self.config_path = config_path

    def load(self) -> SimulationConfig:
        """Load configuration from file."""
        if not self.config_path.exists():
            return SimulationConfig()

        try:
            with open(self.config_path) as f:
                data = json.load(f)

            # Use dacite or simple dict unpacking with filtering
            # For simplicity and robustness, we explicitly map fields or allow extra fields to be ignored
            # if we use **data, but we need to match keys.
            # Using a filtered dict to avoid TypeError on unknown keys
            valid_keys = SimulationConfig.__annotations__.keys()
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}

            config = SimulationConfig(**filtered_data)
            config.validate()
            return config
        except Exception as e:
            # Fallback to default if load fails, or raise?
            # Raising is safer to alert user of corruption
            raise GolfModelingError(f"Failed to load config: {e}") from e

    def save(self, config: SimulationConfig) -> None:
        """Save configuration to file."""
        try:
            data = asdict(config)
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            raise GolfModelingError(f"Failed to save config: {e}") from e
