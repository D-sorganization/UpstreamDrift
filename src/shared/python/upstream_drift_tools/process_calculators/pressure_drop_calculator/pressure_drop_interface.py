#!/usr/bin/env python3
"""Thin public facade for advanced pressure drop calculations.

The implementation now lives in focused helper modules:
- ``pressure_drop_reference`` for discovery/help utilities
- ``pressure_drop_validation`` for input validation
- ``pressure_drop_api`` for public calculation entrypoints and orchestration
- ``pressure_drop_results`` for formatting and result presentation
"""

from __future__ import annotations

from .pressure_drop_api import (
    calculate_pressure_drop,
    calculate_pressure_drop_custom_gas,
    calculate_pressure_drop_syngas,
)
from .pressure_drop_reference import (
    compare_friction_methods,
    list_fittings,
    list_flow_units,
    list_gas_components,
    list_materials,
    list_pipe_sizes,
    show_help,
)
from .pressure_drop_results import print_results
from .pressure_drop_validation import validate_inputs
from .utils.pipe_database import get_roughness


def main() -> None:
    """Command line entrypoint with a representative example."""
    result = calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=100,
        flow_rate=1000,
        flow_unit="SCFM",
        pressure=5,
        pressure_unit="bar",
        temperature=400,
        temperature_unit="K",
        fittings=[
            {"type": "90_elbow_std", "quantity": 4},
            {"type": "gate_valve_open", "quantity": 2},
        ],
    )
    print_results(result, "Example: Air Flow")


__all__ = [
    "show_help",
    "list_gas_components",
    "list_fittings",
    "list_pipe_sizes",
    "list_flow_units",
    "list_materials",
    "get_roughness",
    "compare_friction_methods",
    "validate_inputs",
    "calculate_pressure_drop",
    "calculate_pressure_drop_custom_gas",
    "calculate_pressure_drop_syngas",
    "print_results",
    "main",
]
