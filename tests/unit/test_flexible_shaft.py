"""Tests for Flexible Beam Shaft.

Guideline B5 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from shared.python.flexible_shaft import (
    GRAPHITE_DENSITY,
    GRAPHITE_E,
    SHAFT_LENGTH_DRIVER,
    STEEL_DENSITY,
    STEEL_E,
    ModalShaftModel,
    RigidShaftModel,
    ShaftFlexModel,
    ShaftMaterial,
    ShaftProperties,
    compute_EI_profile,
    compute_mass_profile,
    compute_section_area,
    compute_section_inertia,
    compute_static_deflection,
    create_shaft_model,
    create_standard_shaft,
)


class TestSectionProperties:
    """Tests for section property calculations."""

    def test_section_inertia_solid_rod(self) -> None:
        """Solid rod (zero wall thickness) should have correct inertia."""
        D = 0.01  # 10mm diameter
        # I = π/64 * D⁴
        expected = np.pi / 64 * D**4

        result = compute_section_inertia(D, D / 2)  # Wall = radius = solid

        assert result == pytest.approx(expected, rel=1e-6)

    def test_section_inertia_hollow_tube(self) -> None:
        """Hollow tube should have correct inertia."""
        D_outer = 0.015  # 15mm outer
        t = 0.001  # 1mm wall
        D_inner = D_outer - 2 * t  # 13mm inner

        # I = π/64 * (D_o⁴ - D_i⁴)
        expected = np.pi / 64 * (D_outer**4 - D_inner**4)

        result = compute_section_inertia(D_outer, t)

        assert result == pytest.approx(expected, rel=1e-6)

    def test_section_area_hollow_tube(self) -> None:
        """Hollow tube should have correct area."""
        D_outer = 0.015
        t = 0.001
        D_inner = D_outer - 2 * t

        # A = π/4 * (D_o² - D_i²)
        expected = np.pi / 4 * (D_outer**2 - D_inner**2)

        result = compute_section_area(D_outer, t)

        assert result == pytest.approx(expected, rel=1e-6)


class TestShaftProperties:
    """Tests for shaft property computation."""

    @pytest.fixture
    def standard_shaft(self) -> ShaftProperties:
        """Create standard test shaft."""
        return create_standard_shaft()

    def test_create_standard_shaft(self, standard_shaft: ShaftProperties) -> None:
        """Standard shaft should have correct length and stations."""
        assert standard_shaft.length == pytest.approx(SHAFT_LENGTH_DRIVER)
        assert len(standard_shaft.station_positions) == 11
        assert standard_shaft.station_positions[0] == 0.0
        assert standard_shaft.station_positions[-1] == pytest.approx(
            standard_shaft.length
        )

    def test_taper_direction(self, standard_shaft: ShaftProperties) -> None:
        """Shaft should taper from tip to butt."""
        # Tip is narrower than butt
        assert standard_shaft.outer_diameter[0] < standard_shaft.outer_diameter[-1]

    def test_material_properties_graphite(self) -> None:
        """Graphite shaft should have correct material properties."""
        shaft = create_standard_shaft(material=ShaftMaterial.GRAPHITE)

        assert shaft.youngs_modulus == pytest.approx(GRAPHITE_E)
        assert shaft.density == pytest.approx(GRAPHITE_DENSITY)

    def test_material_properties_steel(self) -> None:
        """Steel shaft should have correct material properties."""
        shaft = create_standard_shaft(material=ShaftMaterial.STEEL)

        assert shaft.youngs_modulus == pytest.approx(STEEL_E)
        assert shaft.density == pytest.approx(STEEL_DENSITY)


class TestEIProfile:
    """Tests for bending stiffness profile."""

    def test_ei_increases_with_diameter(self) -> None:
        """EI should increase from tip (narrow) to butt (wide)."""
        shaft = create_standard_shaft()
        EI = compute_EI_profile(shaft)

        # EI at butt (last) should be larger than at tip (first)
        assert EI[-1] > EI[0]

    def test_ei_positive(self) -> None:
        """All EI values should be positive."""
        shaft = create_standard_shaft()
        EI = compute_EI_profile(shaft)

        assert np.all(EI > 0)


class TestMassProfile:
    """Tests for mass distribution."""

    def test_mass_per_length_reasonable(self) -> None:
        """Mass per length should be in reasonable range."""
        shaft = create_standard_shaft()
        mass = compute_mass_profile(shaft)

        # For graphite shaft, typically 0.05-0.15 kg/m
        assert np.all(mass > 0.01)
        assert np.all(mass < 0.5)

    def test_mass_increases_with_diameter(self) -> None:
        """Mass per length should increase with diameter."""
        shaft = create_standard_shaft()
        mass = compute_mass_profile(shaft)

        assert mass[-1] > mass[0]


class TestRigidShaftModel:
    """Tests for rigid shaft model."""

    def test_zero_deflection(self) -> None:
        """Rigid shaft should have zero deflection."""
        model = RigidShaftModel()
        model.initialize(create_standard_shaft())

        state = model.get_state()

        np.testing.assert_allclose(state.deflections, 0.0)
        np.testing.assert_allclose(state.velocities, 0.0)

    def test_load_has_no_effect(self) -> None:
        """Applied load should not affect rigid shaft."""
        model = RigidShaftModel()
        model.initialize(create_standard_shaft())

        model.apply_load(0.5, np.array([10.0, 0.0, 0.0]))
        state = model.get_state()

        np.testing.assert_allclose(state.deflections, 0.0)


class TestModalShaftModel:
    """Tests for modal shaft model."""

    @pytest.fixture
    def modal_model(self) -> ModalShaftModel:
        """Create initialized modal model."""
        model = ModalShaftModel(n_modes=3)
        model.initialize(create_standard_shaft())
        return model

    def test_creates_requested_modes(self, modal_model: ModalShaftModel) -> None:
        """Should create requested number of modes."""
        assert len(modal_model.modes) == 3

    def test_mode_frequencies_positive(self, modal_model: ModalShaftModel) -> None:
        """All mode frequencies should be positive."""
        for mode in modal_model.modes:
            assert mode.frequency > 0

    def test_mode_frequencies_ascending(self, modal_model: ModalShaftModel) -> None:
        """Mode frequencies should be in ascending order."""
        freqs = [m.frequency for m in modal_model.modes]
        assert freqs == sorted(freqs)

    def test_first_mode_frequency_reasonable(
        self, modal_model: ModalShaftModel
    ) -> None:
        """First mode frequency should be in typical range (5-20 Hz)."""
        first_freq = modal_model.modes[0].frequency

        # Golf shaft first bending mode typically 3-15 Hz
        assert 1 < first_freq < 50

    def test_initial_state_zero(self, modal_model: ModalShaftModel) -> None:
        """Initial state should be zero deflection."""
        state = modal_model.get_state()

        np.testing.assert_allclose(state.deflections, 0.0)
        np.testing.assert_allclose(state.velocities, 0.0)

    def test_step_advances_time(self, modal_model: ModalShaftModel) -> None:
        """Step should advance simulation time."""
        initial_time = modal_model.time

        modal_model.step(0.01)

        assert modal_model.time == pytest.approx(initial_time + 0.01)


class TestStaticDeflection:
    """Tests for static deflection calculation."""

    def test_deflection_at_load_point(self) -> None:
        """Maximum deflection should be at or beyond load point."""
        shaft = create_standard_shaft()
        load_pos = shaft.length * 0.8  # 80% along shaft

        deflection = compute_static_deflection(shaft, load_pos, 10.0)

        # Maximum should be at tip (end of shaft)
        max_idx = np.argmax(deflection)
        assert max_idx == len(deflection) - 1

    def test_deflection_proportional_to_load(self) -> None:
        """Deflection should be proportional to load (linear elasticity)."""
        shaft = create_standard_shaft()
        load_pos = shaft.length

        defl_1 = compute_static_deflection(shaft, load_pos, 1.0)
        defl_10 = compute_static_deflection(shaft, load_pos, 10.0)

        np.testing.assert_allclose(defl_10, 10.0 * defl_1, rtol=1e-10)

    def test_fixed_end_zero_deflection(self) -> None:
        """Fixed end (butt) should have zero deflection."""
        shaft = create_standard_shaft()
        load_pos = shaft.length

        deflection = compute_static_deflection(shaft, load_pos, 10.0)

        assert deflection[0] == pytest.approx(0.0)


class TestShaftModelFactory:
    """Tests for shaft model factory."""

    def test_creates_rigid_model(self) -> None:
        """Factory should create rigid model."""
        model = create_shaft_model(ShaftFlexModel.RIGID)
        assert isinstance(model, RigidShaftModel)

    def test_creates_modal_model(self) -> None:
        """Factory should create modal model."""
        model = create_shaft_model(ShaftFlexModel.MODAL)
        assert isinstance(model, ModalShaftModel)

    def test_fe_falls_back_to_modal(self) -> None:
        """FE model should fall back to modal (not fully implemented)."""
        model = create_shaft_model(ShaftFlexModel.FINITE_ELEMENT)
        # Currently falls back to modal
        assert isinstance(model, ModalShaftModel)
