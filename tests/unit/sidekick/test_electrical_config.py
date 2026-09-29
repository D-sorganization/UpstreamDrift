"""Tests for sidekick.calculators.electrical.config (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from sidekick.calculators.electrical.config import (
    ElectrodeConfig,
)


class TestElectrodeConfigDefaults:
    def test_instantiates_with_defaults(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg is not None

    def test_default_bath_temperature_base(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.bath_temperature_base == 1200.0

    def test_default_bath_temperature(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.bath_temperature == 1350.0

    def test_default_electrode_spacing(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.electrode_spacing_degrees == 120.0

    def test_default_electrode_depths_shape(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.electrode_depths.shape == (3,)

    def test_default_electrode_depths_zeros(self) -> None:
        cfg = ElectrodeConfig()
        np.testing.assert_array_equal(cfg.electrode_depths, np.zeros(3))

    def test_default_phase_voltages_shape(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.phase_voltages.shape == (3,)

    def test_default_phase_voltages_nonzero(self) -> None:
        cfg = ElectrodeConfig()
        assert np.all(cfg.phase_voltages > 0)

    def test_default_k_factors_keys(self) -> None:
        cfg = ElectrodeConfig()
        assert "K_tt" in cfg.k_factors
        assert "K_vert" in cfg.k_factors


class TestElectrodeConfigPostInit:
    def test_colors_populated_after_init(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.colors is not None
        assert isinstance(cfg.colors, dict)

    def test_colors_has_electrode_key(self) -> None:
        cfg = ElectrodeConfig()
        assert "electrode" in cfg.colors  # type: ignore[index]

    def test_color_schemes_populated_after_init(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.color_schemes is not None
        assert isinstance(cfg.color_schemes, dict)

    def test_color_schemes_has_default(self) -> None:
        cfg = ElectrodeConfig()
        assert "default" in cfg.color_schemes  # type: ignore[index]

    def test_custom_colors_not_overwritten(self) -> None:
        custom_colors = {"electrode": "#FF0000"}
        cfg = ElectrodeConfig(colors=custom_colors)
        assert cfg.colors == custom_colors

    def test_custom_color_schemes_not_overwritten(self) -> None:
        custom_schemes = {"my_scheme": {"key": "#ABC"}}
        cfg = ElectrodeConfig(color_schemes=custom_schemes)
        assert cfg.color_schemes == custom_schemes


class TestElectrodeConfigCustomValues:
    def test_custom_bath_temperature(self) -> None:
        cfg = ElectrodeConfig(bath_temperature=1400.0)
        assert cfg.bath_temperature == 1400.0

    def test_custom_furnace_dimensions(self) -> None:
        cfg = ElectrodeConfig(furnace_width=200.0, furnace_length=300.0)
        assert cfg.furnace_width == 200.0
        assert cfg.furnace_length == 300.0

    def test_custom_heat_transfer_coefficient(self) -> None:
        cfg = ElectrodeConfig(heat_transfer_coefficient=200.0)
        assert cfg.heat_transfer_coefficient == 200.0

    def test_custom_k_factors(self) -> None:
        k = {"K_tt": 0.5, "K_vert": 0.8}
        cfg = ElectrodeConfig(k_factors=k)
        assert cfg.k_factors["K_tt"] == 0.5
