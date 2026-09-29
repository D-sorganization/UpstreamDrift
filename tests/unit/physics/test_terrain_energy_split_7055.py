"""Value tests for terrain energy-absorption weights and grass constant (#7055).

The energy-absorption blend weights (0.5 / 0.3 / 0.2) and the grass-resistance
coefficient (0.1) in ``_terrain_physics.py`` were previously unprovenanced
magic numbers. They are now named, documented module constants; these tests
pin both their documented invariants and the resulting physical behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.physics._terrain_physics import (
    ENERGY_ABSORPTION_COMPRESSIBILITY_WEIGHT,
    ENERGY_ABSORPTION_DAMPING_WEIGHT,
    ENERGY_ABSORPTION_RESTITUTION_WEIGHT,
    GRASS_RESISTANCE_COEFFICIENT,
    CompressibleTurfModel,
)


class TestEnergyAbsorptionWeights:
    """Documented invariants of the absorption blend weights."""

    @pytest.mark.unit
    def test_weights_form_convex_combination(self) -> None:
        """Weights must sum to exactly 1.0 (convex combination)."""
        total = (
            ENERGY_ABSORPTION_COMPRESSIBILITY_WEIGHT
            + ENERGY_ABSORPTION_DAMPING_WEIGHT
            + ENERGY_ABSORPTION_RESTITUTION_WEIGHT
        )
        assert total == pytest.approx(1.0)

    @pytest.mark.unit
    def test_weights_are_documented_values(self) -> None:
        assert pytest.approx(0.5) == ENERGY_ABSORPTION_COMPRESSIBILITY_WEIGHT
        assert pytest.approx(0.3) == ENERGY_ABSORPTION_DAMPING_WEIGHT
        assert pytest.approx(0.2) == ENERGY_ABSORPTION_RESTITUTION_WEIGHT

    @pytest.mark.unit
    def test_grass_coefficient_documented_value(self) -> None:
        assert pytest.approx(0.1) == GRASS_RESISTANCE_COEFFICIENT


@dataclass
class _StubMaterial:
    compressibility: float
    compression_damping: float
    restitution: float


class _StubTerrain:
    """Minimal terrain stub exposing the surface used by the contact model."""

    def __init__(self, material: _StubMaterial) -> None:
        self._material = material

    def get_material(self, x: float, y: float) -> _StubMaterial:
        return self._material

    def get_normal(self, x: float, y: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])


class TestEnergyAbsorptionBehaviour:
    """The energy-split weights produce the documented absorption factor."""

    @pytest.mark.unit
    def test_absorption_factor_matches_weighted_sum(self) -> None:
        """Absorbed normal energy == weighted blend of material mechanisms.

        With a purely normal impact (velocity along +Z) the absorbed energy is
        ``0.5*m*v^2 * absorption_factor`` where ``absorption_factor`` is the
        documented convex combination. Pin it against the hand-computed value.
        """
        mat = _StubMaterial(
            compressibility=0.6,
            compression_damping=0.4,
            restitution=0.3,
        )
        model = CompressibleTurfModel(terrain=_StubTerrain(mat))  # type: ignore[arg-type]

        m = 0.04593
        v = 20.0
        result = model.compute_energy_absorption(
            x=0.0, y=0.0, impact_velocity=np.array([0.0, 0.0, -v]), mass=m
        )

        expected_factor = (
            0.6 * ENERGY_ABSORPTION_COMPRESSIBILITY_WEIGHT
            + 0.4 * ENERGY_ABSORPTION_DAMPING_WEIGHT
            + (1.0 - 0.3) * ENERGY_ABSORPTION_RESTITUTION_WEIGHT
        )
        normal_energy = 0.5 * m * v**2
        expected_absorbed = normal_energy * expected_factor

        assert result["absorbed_energy"] == pytest.approx(expected_absorbed, rel=1e-9)

    @pytest.mark.unit
    def test_absorption_bounded_in_unit_interval(self) -> None:
        """With every mechanism maxed, absorption factor stays <= 1.0."""
        mat = _StubMaterial(
            compressibility=1.0, compression_damping=1.0, restitution=0.0
        )
        model = CompressibleTurfModel(terrain=_StubTerrain(mat))  # type: ignore[arg-type]
        result = model.compute_energy_absorption(
            x=0.0, y=0.0, impact_velocity=np.array([0.0, 0.0, -20.0])
        )
        assert 0.0 <= result["energy_absorption_ratio"] <= 1.0
