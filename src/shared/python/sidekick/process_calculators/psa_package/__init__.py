"""
PSA (Pressure Swing Adsorption) Analysis Package.

This package provides tools for analyzing two-stage PSA systems including:
- Core calculation model (psa_model.py)
- Interactive GUI (psa_gui.py)
- Jupyter notebook analysis (psa_analysis.ipynb)
- Test suite (test_psa_model.py)
"""

from .psa_model import (
    DEFAULT_COMPONENTS,
    ComponentData,
    PSAModel,
    PSAResults,
    StreamCompositions,
    StreamFlows,
    calculate_o2_safety_analysis,
    calculate_sensitivity,
    get_flammability_status,
)

__all__ = [
    "ComponentData",
    "DEFAULT_COMPONENTS",
    "PSAModel",
    "PSAResults",
    "StreamCompositions",
    "StreamFlows",
    "calculate_o2_safety_analysis",
    "calculate_sensitivity",
    "get_flammability_status",
]
