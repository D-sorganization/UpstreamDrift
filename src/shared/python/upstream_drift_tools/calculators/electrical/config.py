"""Electrode Configuration Module

Contains the configuration dataclass for electrode system parameters.
Extracted from electrode_advisor.py for better organization.

Author: Chemical Equilibrium Calculator Team
Date: July 8, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Note: Removed PyQt6 dependency to make this a pure logic/data module.
# Colors can be handled by the UI layer or stored as hex strings/tuples here.


@dataclass
class ElectrodeConfig:
    """Configuration dataclass for electrode system parameters"""

    k_scaling_factor: float = 0.035
    glass_depth: float = 15.0  # inches
    metal_layer_height: float = 7.0  # inches above metal layer
    electrode_spacing_degrees: float = 120.0
    bath_temperature_base: float = 1200.0  # °C
    cylinder_segments: int = 10  # For cylindrical path modeling

    # Heat transfer parameters
    heat_transfer_coefficient: float = 100.0  # W/(m²·K)
    glass_thermal_conductivity: float = 1.5  # W/(m·K)
    metal_conductivity: float = 1000.0  # W/(m·K) - Much higher for metal

    # Furnace dimensions
    furnace_width: float = 120.0  # inches
    furnace_length: float = 180.0  # inches
    furnace_height: float = 48.0  # inches
    glass_level: float = 36.0  # inches from bottom
    bath_diameter: float = 120.0  # inches
    tip_diameter: float = 24.0  # inches
    metal_depth: float = 2.0  # inches
    conductive_height: float = 2.0  # inches
    bath_temperature: float = 1350.0  # °C

    # Electrical parameters
    k_factors: dict[str, float] = field(
        default_factory=lambda: {"K_tt": 1.0, "K_vert": 1.0}
    )
    electrode_depths: np.ndarray = field(default_factory=lambda: np.zeros(3))
    phase_voltages: np.ndarray = field(default_factory=lambda: np.ones(3) * 100.0)
    phase_currents: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Spreading factors for conductive paths
    vertical_spreading_factor: float = 1.5  # Vertical spread for horizontal paths
    horizontal_spreading_factor: float = 1.2  # Horizontal spread for vertical paths

    # Display colors (Hex strings) - Decoupled from Qt
    colors: dict[str, str] | None = None

    # Color schemes for current/power visualization
    color_schemes: dict[str, dict] | None = None

    def __post_init__(self) -> None:
        if self.colors is None:
            self.colors = {
                "window_bg": "#E6F0FF",  # QColor(230, 240, 255)
                "input_panel_bg": "#F0FFF0",  # QColor(240, 255, 240)
                "status_ok": "#C8FFC8",  # QColor(200, 255, 200)
                "status_warn": "#FFFFB4",  # QColor(255, 255, 180)
                "status_err": "#FF9696",  # QColor(255, 150, 150)
                "electrode": "#808080",  # QColor(128, 128, 128)
                "glass_cold": "#FFA500",  # Orange for cold glass
                "glass_hot": "#FF4500",  # Red-orange for hot molten glass
                "glass_molten": "#FF8C00",  # Dark orange for molten glass
                "metal_layer": "#505050",  # QColor(80, 80, 80)
                "current_path": "#0072BD",  # QColor(0, 114, 189)
                "metal_path": "#C0C0C0",  # Silver for metal paths
                "metal_shell": "#646464",  # Dark grey for metal shell
            }

        if self.color_schemes is None:
            self.color_schemes = {
                "default": {
                    "direct_glass": "#4169E1",  # Royal blue
                    "via_metal_down": "#DC143C",  # Crimson
                    "via_metal_horizontal": "#C0C0C0",  # Silver
                    "via_metal_up": "#DC143C",  # Crimson
                },
                "current_intensity": {
                    # Uses gradient from blue (low) to red (high)
                    "gradient": "coolwarm",
                    "min_color": "#0000FF",
                    "max_color": "#FF0000",
                },
                "power_dissipation": {
                    # Uses gradient from green (low) to yellow to red (high)
                    "gradient": "RdYlGn_r",
                    "min_color": "#00FF00",
                    "max_color": "#FF0000",
                },
            }
