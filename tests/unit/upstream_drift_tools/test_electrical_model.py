"""Tests for sidekick.calculators.electrical.electrical_model (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from sidekick.calculators.electrical.config import (
    ElectrodeConfig,
)
from sidekick.calculators.electrical.electrical_model import (
    ThreePhaseElectricalModelEnhanced,
)
from sidekick.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)


def _make_model() -> ThreePhaseElectricalModelEnhanced:
    return ThreePhaseElectricalModelEnhanced(
        ElectrodeConfig(), GlassPropertiesInterface()
    )


class TestThreePhaseElectricalModelInit:
    def test_electrical_model_instantiates(self) -> None:
        m = _make_model()
        assert m is not None

    def test_stores_config(self) -> None:
        cfg = ElectrodeConfig()
        m = ThreePhaseElectricalModelEnhanced(cfg, GlassPropertiesInterface())
        assert m.config is cfg

    def test_electrode_positions_shape(self) -> None:
        m = _make_model()
        assert m.electrode_positions.shape == (3,)

    def test_power_history_initially_empty(self) -> None:
        m = _make_model()
        assert len(m.power_history) == 0


class TestCalculateSystemState:
    _DEPTHS = np.array([10.0, 10.0, 10.0])
    _K = {"K_tt": 1.0, "K_vert": 1.0}

    def test_electrical_model_returns_dict(self) -> None:
        m = _make_model()
        state = m.calculate_system_state(self._DEPTHS, 120.0, 24.0, 2.0, self._K)
        assert isinstance(state, dict)

    def test_has_resistances_key(self) -> None:
        m = _make_model()
        state = m.calculate_system_state(self._DEPTHS, 120.0, 24.0, 2.0, self._K)
        assert "resistances" in state

    def test_has_current_paths_key(self) -> None:
        m = _make_model()
        state = m.calculate_system_state(self._DEPTHS, 120.0, 24.0, 2.0, self._K)
        assert "current_paths" in state

    def test_has_electrode_positions_key(self) -> None:
        m = _make_model()
        state = m.calculate_system_state(self._DEPTHS, 120.0, 24.0, 2.0, self._K)
        assert "electrode_positions" in state

    def test_resistances_are_positive(self) -> None:
        m = _make_model()
        state = m.calculate_system_state(self._DEPTHS, 120.0, 24.0, 2.0, self._K)
        for key, val in state["resistances"].items():
            assert val > 0.0, f"Resistance {key!r} should be positive, got {val}"

    def test_symmetric_depths_equal_resistances(self) -> None:
        m = _make_model()
        state = m.calculate_system_state(
            np.array([10.0, 10.0, 10.0]), 120.0, 24.0, 2.0, self._K
        )
        resistances = list(state["resistances"].values())
        # With identical depths all 3 phase resistances should be approximately equal
        assert abs(resistances[0] - resistances[1]) / resistances[0] < 0.01

    def test_deeper_electrodes_lower_resistance(self) -> None:
        m = _make_model()
        state_shallow = m.calculate_system_state(
            np.array([5.0, 5.0, 5.0]), 120.0, 24.0, 2.0, self._K
        )
        state_deep = m.calculate_system_state(
            np.array([20.0, 20.0, 20.0]), 120.0, 24.0, 2.0, self._K
        )
        # Deeper electrodes should have lower total resistance
        avg_shallow = sum(state_shallow["resistances"].values()) / len(
            state_shallow["resistances"]
        )
        avg_deep = sum(state_deep["resistances"].values()) / len(
            state_deep["resistances"]
        )
        assert avg_deep < avg_shallow


class TestParallelResistance:
    def test_two_equal_resistances(self) -> None:
        m = _make_model()
        r = m._parallel_resistance(10.0, 10.0)
        assert abs(r - 5.0) < 1e-10

    def test_parallel_less_than_either(self) -> None:
        m = _make_model()
        r = m._parallel_resistance(100.0, 200.0)
        assert r < 100.0
        assert r < 200.0
