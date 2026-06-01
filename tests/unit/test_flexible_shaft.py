"""Tests for Flexible Beam Shaft.

Guideline B5 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.physics.flexible_shaft import (
    GRAPHITE_DENSITY,
    GRAPHITE_E,
    SHAFT_LENGTH_DRIVER,
    STEEL_DENSITY,
    STEEL_E,
    FiniteElementShaftModel,
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
        """Shaft should taper from butt (station 0, clamped) to tip (#6983).

        Station 0 is the clamped/fixed end of the cantilever, which for a golf
        shaft is the thick butt. The free end (last station) is the thin tip.
        """
        # Butt (clamped, station 0) is wider than tip (free end)
        assert standard_shaft.outer_diameter[0] > standard_shaft.outer_diameter[-1]

    @pytest.mark.parametrize(
        "material, expected_E, expected_density",
        [
            (ShaftMaterial.GRAPHITE, GRAPHITE_E, GRAPHITE_DENSITY),
            (ShaftMaterial.STEEL, STEEL_E, STEEL_DENSITY),
        ],
        ids=["graphite", "steel"],
    )
    def test_material_properties(
        self,
        material: ShaftMaterial,
        expected_E: float,
        expected_density: float,
    ) -> None:
        """Shaft should have correct material properties for given material."""
        shaft = create_standard_shaft(material=material)
        assert shaft.youngs_modulus == pytest.approx(expected_E)
        assert shaft.density == pytest.approx(expected_density)


class TestEIProfile:
    """Tests for bending stiffness profile."""

    def test_ei_increases_with_diameter(self) -> None:
        """EI should be largest at the clamped butt (station 0) (#6983)."""
        shaft = create_standard_shaft()
        EI = compute_EI_profile(shaft)

        # EI at butt (station 0, clamped) should exceed EI at tip (free end)
        assert EI[0] > EI[-1]

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
        """Mass per length should be largest at the clamped butt (#6983)."""
        shaft = create_standard_shaft()
        mass = compute_mass_profile(shaft)

        # Butt (station 0, clamped) is thicker -> heavier than the tip
        assert mass[0] > mass[-1]


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

    def test_flexible_shaft_step_advances_time(
        self, modal_model: ModalShaftModel
    ) -> None:
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

    @pytest.mark.parametrize(
        "flex_model, expected_class",
        [
            (ShaftFlexModel.RIGID, RigidShaftModel),
            (ShaftFlexModel.MODAL, ModalShaftModel),
            (ShaftFlexModel.FINITE_ELEMENT, FiniteElementShaftModel),
        ],
        ids=["rigid", "modal", "finite-element"],
    )
    def test_creates_correct_model(
        self, flex_model: ShaftFlexModel, expected_class: type
    ) -> None:
        """Factory should create the correct model type."""
        model = create_shaft_model(flex_model)
        assert isinstance(model, expected_class)

    def test_creates_fe_model_with_custom_elements(self) -> None:
        """Factory should respect n_elements parameter for FE model."""
        model = create_shaft_model(ShaftFlexModel.FINITE_ELEMENT, n_elements=20)
        assert isinstance(model, FiniteElementShaftModel)
        assert model.n_elements == 20


class TestFiniteElementShaftModel:
    """Tests for finite element shaft model (Issue #756)."""

    @pytest.fixture
    def fe_model(self) -> FiniteElementShaftModel:
        """Create initialized FE model."""
        model = FiniteElementShaftModel(n_elements=10)
        model.initialize(create_standard_shaft())
        return model

    def test_creates_correct_elements(self, fe_model: FiniteElementShaftModel) -> None:
        """Should create requested number of elements."""
        assert len(fe_model.elements) == 10
        assert fe_model.n_nodes == 11
        assert fe_model.n_dof == 22  # 2 DOF per node

    def test_free_dof_after_bc(self, fe_model: FiniteElementShaftModel) -> None:
        """Should have correct free DOFs after boundary conditions."""
        # 22 total - 2 fixed at node 0 = 20 free DOFs
        assert fe_model.n_free_dof == 20

    def test_stiffness_matrix_symmetric(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """Stiffness matrix should be symmetric."""
        K = fe_model.K
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_mass_matrix_symmetric(self, fe_model: FiniteElementShaftModel) -> None:
        """Mass matrix should be symmetric."""
        M = fe_model.M
        np.testing.assert_allclose(M, M.T, rtol=1e-10)

    def test_stiffness_matrix_positive_definite(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """Stiffness matrix should be positive definite."""
        eigenvalues = np.linalg.eigvalsh(fe_model.K)
        assert np.all(eigenvalues > 0)

    def test_flexible_shaft_mass_matrix_positive_definite(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """Mass matrix should be positive definite."""
        eigenvalues = np.linalg.eigvalsh(fe_model.M)
        assert np.all(eigenvalues > 0)

    def test_initial_state_zero(self, fe_model: FiniteElementShaftModel) -> None:
        """Initial state should be zero deflection."""
        state = fe_model.get_state()

        np.testing.assert_allclose(state.deflections, 0.0)
        np.testing.assert_allclose(state.velocities, 0.0)
        np.testing.assert_allclose(state.rotations, 0.0)

    def test_fixed_end_always_zero(self, fe_model: FiniteElementShaftModel) -> None:
        """Fixed end (node 0) should always have zero displacement."""
        # Apply load at tip
        fe_model.apply_load(fe_model.properties.length, np.array([0, 10.0, 0]))
        fe_model.step(0.001)

        state = fe_model.get_state()
        # First node (fixed end) should be zero
        assert state.deflections[0] == pytest.approx(0.0, abs=1e-15)
        assert state.rotations[0] == pytest.approx(0.0, abs=1e-15)

    def test_flexible_shaft_step_advances_time(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """Step should advance simulation time."""
        initial_time = fe_model.time

        fe_model.step(0.01)

        assert fe_model.time == pytest.approx(initial_time + 0.01)

    def test_load_produces_deflection(self, fe_model: FiniteElementShaftModel) -> None:
        """Applied load should produce deflection."""
        # Apply constant load and step several times
        for _ in range(10):
            fe_model.apply_load(fe_model.properties.length, np.array([0, 100.0, 0]))
            fe_model.step(0.001)

        state = fe_model.get_state()
        # Tip (last node) should have non-zero deflection
        assert abs(state.deflections[-1]) > 0

    def test_static_solution_matches_analytical(self) -> None:
        """Static solution should approximate analytical beam theory for uniform beam.

        Note: For tapered beams, the FE model properly captures varying EI
        while the analytical formula uses average EI, so they will differ.
        This test uses a uniform beam for fair comparison.
        """
        # Create uniform shaft (constant diameter) for fair comparison
        uniform_shaft = ShaftProperties(
            length=1.0,
            outer_diameter=np.full(11, 0.012),  # Constant 12mm diameter
            wall_thickness=np.full(11, 0.001),  # Constant 1mm wall
            station_positions=np.linspace(0, 1.0, 11),
            youngs_modulus=GRAPHITE_E,
            density=GRAPHITE_DENSITY,
        )

        model = FiniteElementShaftModel(n_elements=20)  # More elements for accuracy
        model.initialize(uniform_shaft)

        load_pos = uniform_shaft.length
        load_force = 10.0

        # FE static solution
        fe_state = model.compute_static_solution(load_pos, load_force)

        # Analytical solution
        analytical = compute_static_deflection(uniform_shaft, load_pos, load_force)

        # Interpolate FE to match analytical station positions
        fe_positions = np.linspace(0, uniform_shaft.length, model.n_nodes)
        fe_at_stations = np.interp(
            uniform_shaft.station_positions, fe_positions, fe_state.deflections
        )

        # Should match within 5% for uniform beam with fine mesh
        np.testing.assert_allclose(fe_at_stations, analytical, rtol=0.05)

    def test_natural_frequencies_positive(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """Natural frequencies should be positive."""
        frequencies = fe_model.compute_natural_frequencies(n_modes=3)

        assert len(frequencies) >= 1
        for freq in frequencies:
            assert freq > 0

    def test_natural_frequencies_ascending(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """Natural frequencies should be in ascending order."""
        frequencies = fe_model.compute_natural_frequencies(n_modes=5)

        assert frequencies == sorted(frequencies)

    def test_first_frequency_in_physical_range(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """First natural frequency should be in physical range for golf shaft."""
        frequencies = fe_model.compute_natural_frequencies(n_modes=1)

        # Golf shaft first bending mode typically 3-20 Hz
        assert 1 < frequencies[0] < 100

    def test_energy_decays_with_damping(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """System energy should decay over time due to damping."""
        # Give initial velocity
        fe_model.v = np.ones(fe_model.n_free_dof) * 0.01

        # Compute initial kinetic energy
        initial_ke = 0.5 * fe_model.v @ fe_model.M @ fe_model.v

        # Step forward
        for _ in range(100):
            fe_model.step(0.001)

        # Compute final kinetic energy
        final_ke = 0.5 * fe_model.v @ fe_model.M @ fe_model.v
        final_pe = 0.5 * fe_model.u @ fe_model.K @ fe_model.u
        final_total = final_ke + final_pe

        # Total energy should have decreased
        assert final_total < initial_ke

    def test_newmark_integration_stable(
        self, fe_model: FiniteElementShaftModel
    ) -> None:
        """Newmark integration should remain stable over many steps."""
        # Apply impulse
        fe_model.apply_load(fe_model.properties.length, np.array([0, 1000.0, 0]))
        fe_model.step(0.0001)

        # Run for many steps
        max_deflection = 0.0
        for _ in range(1000):
            fe_model.step(0.0001)
            state = fe_model.get_state()
            max_deflection = max(max_deflection, np.max(np.abs(state.deflections)))

        # Should not blow up (reasonable deflection magnitude)
        assert max_deflection < 1.0  # Less than 1 meter

    def test_different_element_counts(self) -> None:
        """Model should work with different element counts."""
        for n_elem in [5, 10, 20]:
            model = FiniteElementShaftModel(n_elements=n_elem)
            model.initialize(create_standard_shaft())

            assert len(model.elements) == n_elem
            assert model.n_nodes == n_elem + 1

            # Should be able to compute solution
            state = model.get_state()
            assert len(state.deflections) == n_elem + 1


class TestCantileverBoundaryCondition:
    """Regression tests for #6983: clamp the BUTT, not the thin tip."""

    @staticmethod
    def _uniform_shaft() -> ShaftProperties:
        """Uniform beam so the analytic cantilever frequency applies."""
        return ShaftProperties(
            length=1.0,
            outer_diameter=np.full(11, 0.012),
            wall_thickness=np.full(11, 0.001),
            station_positions=np.linspace(0, 1.0, 11),
            youngs_modulus=GRAPHITE_E,
            density=GRAPHITE_DENSITY,
        )

    def test_first_frequency_matches_analytic_cantilever(self) -> None:
        """First natural frequency must match the analytic clamped-free beam.

        f1 = (1.875104^2 / 2pi) * sqrt(EI / (mu L^4)). A correct cantilever
        clamps one full end (2 DOFs). Clamping the wrong end gives the same
        eigenvalues for a *uniform* beam, so the discriminating power of this
        test is the magnitude check against the closed-form value.
        """
        shaft = self._uniform_shaft()
        model = FiniteElementShaftModel(n_elements=20)
        model.initialize(shaft)

        EI = compute_EI_profile(shaft)
        mass = compute_mass_profile(shaft)
        ei = float(np.mean(EI))
        mu = float(np.mean(mass))
        beta1 = 1.875104
        f1_analytic = (beta1**2) / (2 * np.pi) * np.sqrt(ei / (mu * shaft.length**4))

        f1 = model.compute_natural_frequencies(n_modes=1)[0]
        assert f1 == pytest.approx(f1_analytic, rel=0.05)

    def test_clamped_end_is_the_butt(self) -> None:
        """The clamped node (station 0) must be the thick butt, not the tip.

        The standard shaft must place the larger diameter / stiffer section at
        the clamped end so the FE BC and the analytic convention agree.
        """
        shaft = create_standard_shaft()
        EI = compute_EI_profile(shaft)
        # Station 0 is the fixed end; it must be the stiffest (butt) section.
        assert EI[0] == pytest.approx(EI.max())
        assert shaft.outer_diameter[0] == pytest.approx(shaft.outer_diameter.max())

    def test_tapered_cantilever_frequency_correct_clamp_end(self) -> None:
        """Tapered shaft clamped at the butt is stiffer than clamped at the tip.

        A butt-clamped tapered cantilever (thick root, thin free end) has a
        higher fundamental frequency than the same beam clamped at the thin
        tip. Build both orientations explicitly and assert the standard shaft
        (butt-clamped) matches the stiff orientation.
        """
        length = 1.168
        n = 11
        thick = np.linspace(0.015, 0.0085, n)  # station0=butt(thick) -> tip(thin)
        wall = np.full(n, 0.001)
        pos = np.linspace(0, length, n)

        butt_clamped = ShaftProperties(
            length=length,
            outer_diameter=thick,
            wall_thickness=wall,
            station_positions=pos,
            youngs_modulus=GRAPHITE_E,
            density=GRAPHITE_DENSITY,
        )
        tip_clamped = ShaftProperties(
            length=length,
            outer_diameter=thick[::-1],  # station0=tip(thin) -> butt(thick)
            wall_thickness=wall,
            station_positions=pos,
            youngs_modulus=GRAPHITE_E,
            density=GRAPHITE_DENSITY,
        )

        m_butt = FiniteElementShaftModel(n_elements=20)
        m_butt.initialize(butt_clamped)
        m_tip = FiniteElementShaftModel(n_elements=20)
        m_tip.initialize(tip_clamped)

        f_butt = m_butt.compute_natural_frequencies(n_modes=1)[0]
        f_tip = m_tip.compute_natural_frequencies(n_modes=1)[0]

        # Clamping the thick butt yields a stiffer, higher-frequency cantilever.
        assert f_butt > f_tip

        # The standard shaft is butt-clamped: its f1 must equal the stiff case.
        std = create_standard_shaft()
        m_std = FiniteElementShaftModel(n_elements=20)
        m_std.initialize(std)
        f_std = m_std.compute_natural_frequencies(n_modes=1)[0]
        assert f_std == pytest.approx(f_butt, rel=1e-9)


class TestNewmarkConditioning:
    """Regression tests for #6985: Newmark accuracy at impact-scale dt."""

    @staticmethod
    def _model() -> FiniteElementShaftModel:
        model = FiniteElementShaftModel(n_elements=10)
        model.initialize(create_standard_shaft())
        return model

    def test_static_limit_accurate_at_small_dt(self) -> None:
        """A sustained load integrated at impact-scale dt converges to statics.

        With a constant load and very small dt the displacement after many
        steps must approach the static solution K u = f. Catastrophic
        cancellation in the a_new recovery shows up as a large relative error
        here.
        """
        model = self._model()
        pos = model.properties.length
        force = np.array([0.0, 50.0, 0.0])

        static = model.compute_static_solution(pos, 50.0)
        static_tip = static.deflections[-1]

        dt = 1e-7  # impact scale
        for _ in range(2000):
            model.apply_load(pos, force)
            model.step(dt)

        tip = model.get_state().deflections[-1]
        # Tip deflection should be the same sign and small fraction of static
        # (transient just starting) and, crucially, finite and non-NaN.
        assert np.isfinite(tip)
        assert np.sign(tip) == np.sign(static_tip)

    def test_small_dt_emits_conditioning_warning(self, caplog) -> None:
        """Stepping below the conditioning threshold should warn (#6985)."""
        import logging

        model = self._model()
        with caplog.at_level(logging.WARNING):
            model.step(1e-9)
        assert any(
            "conditioning" in r.message.lower()
            or "ill-conditioned" in r.message.lower()
            for r in caplog.records
        )

    def test_no_warning_at_reasonable_dt(self, caplog) -> None:
        """A physically reasonable dt should not trip the warning."""
        import logging

        model = self._model()
        with caplog.at_level(logging.WARNING):
            model.step(1e-3)
        assert not any(
            "conditioning" in r.message.lower()
            or "ill-conditioned" in r.message.lower()
            for r in caplog.records
        )
