from __future__ import annotations

import numpy as np

from src.shared.python.contracts import require

from ._shaft_data import ShaftFlexModel, ShaftProperties
from ._shaft_fem import FiniteElementShaftModel
from ._shaft_model import ModalShaftModel, RigidShaftModel, ShaftModel
from ._shaft_properties import compute_EI_profile, create_standard_shaft


def compute_static_deflection(
    properties: ShaftProperties,
    load_position: float,
    load_force: float,
) -> np.ndarray:
    """Compute static deflection for cantilever beam with point load.

    Assumes cantilevered at butt end, load applied at position.
    Uses Euler-Bernoulli beam theory.

    For point load P at distance a from fixed end on beam of length L:
    w(x) = Px²(3a-x)/(6EI) for x ≤ a
    w(x) = Pa²(3x-a)/(6EI) for x > a

    Args:
        properties: Shaft properties
        load_position: Position of load from butt end [m]
        load_force: Load magnitude [N]

    Returns:
        Deflection at each station [m]
    """
    require(properties is not None, "properties must be provided", properties)
    require(
        0.0 <= load_position <= properties.length,
        "load_position must be within shaft length [0, length]",
        load_position,
    )
    EI = compute_EI_profile(properties)
    EI_avg = float(np.mean(EI))  # Use average for simplicity

    stations = properties.station_positions
    a = load_position  # Load position from butt (fixed end)

    deflection = np.zeros(len(stations))

    for i, x in enumerate(stations):
        if x <= a:
            deflection[i] = load_force * x**2 * (3 * a - x) / (6 * EI_avg)
        else:
            deflection[i] = load_force * a**2 * (3 * x - a) / (6 * EI_avg)

    return deflection


def create_shaft_model(
    model_type: ShaftFlexModel,
    properties: ShaftProperties | None = None,
    n_elements: int = 10,
) -> ShaftModel:
    """Factory function to create shaft model.

    Args:
        model_type: Type of shaft model
        properties: Shaft properties (uses default if None)
        n_elements: Number of elements for FE model (default 10)

    Returns:
        Initialized shaft model
    """
    if model_type is None:
        raise ValueError("model_type must be provided")
    if properties is None:
        properties = create_standard_shaft()

    model: ShaftModel
    if model_type == ShaftFlexModel.RIGID:
        model = RigidShaftModel()
    elif model_type == ShaftFlexModel.MODAL:
        model = ModalShaftModel()
    elif model_type == ShaftFlexModel.FINITE_ELEMENT:
        model = FiniteElementShaftModel(n_elements=n_elements)
    else:
        raise ValueError(f"Unknown shaft model type: {model_type}")

    model.initialize(properties)
    return model
