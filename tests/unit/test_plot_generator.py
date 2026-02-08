"""Tests for the Plot Generator module.

Tests plot creation, configuration, and output for all standard plot types.

Follows TDD and Design by Contract principles.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.plot_generator import (
    ALL_PLOT_TYPES,
    PlotConfig,
    PlotGenerator,
    PlotType,
    SimulationData,
)

# ---- Fixtures ----


@pytest.fixture
def sample_data() -> SimulationData:
    """Create sample simulation data for plotting."""
    n_steps = 100
    n_joints = 4
    times = np.linspace(0, 2.0, n_steps)

    return SimulationData(
        times=times,
        positions=np.random.randn(n_steps, n_joints) * 0.5,
        velocities=np.random.randn(n_steps, n_joints) * 1.0,
        accelerations=np.random.randn(n_steps, n_joints) * 2.0,
        torques=np.random.randn(n_steps, n_joints) * 10.0,
        energies={
            "kinetic": np.abs(np.random.randn(n_steps)) * 5.0,
            "potential": np.abs(np.random.randn(n_steps)) * 3.0,
        },
        contact_forces=np.random.randn(n_steps, 3) * 100.0,
        drift_accelerations=np.random.randn(n_steps, n_joints) * 1.0,
        control_accelerations=np.random.randn(n_steps, n_joints) * 1.0,
        mass_matrices=np.tile(np.eye(n_joints), (n_steps, 1, 1)),
        joint_names=["shoulder", "elbow", "wrist", "hip"],
        model_name="test_model",
    )


@pytest.fixture
def generator() -> PlotGenerator:
    """Create a default plot generator."""
    return PlotGenerator()


# ---- Configuration Tests ----


class TestPlotConfig:
    """Tests for PlotConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PlotConfig()
        assert config.output_format == "png"
        assert config.dpi == 150
        assert config.show_grid is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PlotConfig(
            output_format="svg",
            dpi=300,
            show_grid=False,
            joint_indices=[0, 1],
        )
        assert config.output_format == "svg"
        assert config.dpi == 300
        assert config.show_grid is False
        assert config.joint_indices == [0, 1]


# ---- Plot Type Tests ----


class TestPlotTypes:
    """Tests for plot type definitions."""

    def test_all_plot_types_defined(self) -> None:
        """Test that all standard plot types are defined."""
        assert len(ALL_PLOT_TYPES) >= 8

    def test_plot_type_values(self) -> None:
        """Test plot type string values."""
        assert PlotType.JOINT_POSITIONS == "joint_positions"
        assert PlotType.ENERGY == "energy"
        assert PlotType.PHASE_PORTRAIT == "phase_portrait"

    def test_available_types_descriptions(self) -> None:
        """Test available plot types have descriptions."""
        gen = PlotGenerator()
        types = gen.get_available_plot_types()
        assert len(types) >= 8
        for t in types:
            assert "type" in t
            assert "description" in t
            assert len(t["description"]) > 0


# ---- Plot Generation Tests ----


@pytest.mark.skipif(
    not True,  # matplotlib is always importable in this env
    reason="matplotlib not available",
)
class TestPlotGeneration:
    """Tests for actual plot generation."""

    def test_generate_single_plot(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test generating a single plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.JOINT_POSITIONS)
        assert fig is not None

    def test_generate_single_plot_with_save(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test generating and saving a single plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"
            generator.generate_single_plot(
                sample_data, PlotType.JOINT_POSITIONS, output_path
            )
            assert output_path.exists()

    def test_generate_all_standard_plots(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test generating all standard plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generator.generate_standard_plots(sample_data, tmpdir)
            assert len(generated) > 0
            for path in generated:
                assert path.exists()

    def test_generate_joint_positions(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test joint positions plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.JOINT_POSITIONS)
        assert fig is not None

    def test_generate_joint_velocities(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test joint velocities plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.JOINT_VELOCITIES)
        assert fig is not None

    def test_generate_energy_plot(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test energy plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.ENERGY)
        assert fig is not None

    def test_generate_phase_portrait(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test phase portrait plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.PHASE_PORTRAIT)
        assert fig is not None

    def test_generate_contact_forces(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test contact forces plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.CONTACT_FORCES)
        assert fig is not None

    def test_generate_drift_vs_control(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test drift vs control decomposition plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.DRIFT_VS_CONTROL)
        assert fig is not None

    def test_generate_power_plot(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test power plot."""
        fig = generator.generate_single_plot(sample_data, PlotType.POWER)
        assert fig is not None

    def test_generate_mass_matrix_condition(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test mass matrix condition number plot."""
        fig = generator.generate_single_plot(
            sample_data, PlotType.MASS_MATRIX_CONDITION
        )
        assert fig is not None

    def test_unknown_plot_type(
        self, generator: PlotGenerator, sample_data: SimulationData
    ) -> None:
        """Test unknown plot type returns None."""
        fig = generator.generate_single_plot(sample_data, "unknown_type")
        assert fig is None


# ---- Missing Data Tests ----


class TestMissingData:
    """Tests for graceful handling of missing data."""

    def test_no_accelerations(self, generator: PlotGenerator) -> None:
        """Test plot with missing accelerations."""
        data = SimulationData(
            times=np.linspace(0, 1, 50),
            positions=np.random.randn(50, 3),
            velocities=np.random.randn(50, 3),
            accelerations=None,
        )
        fig = generator.generate_single_plot(data, PlotType.JOINT_ACCELERATIONS)
        assert fig is None  # Should handle gracefully

    def test_no_torques(self, generator: PlotGenerator) -> None:
        """Test plot with missing torques."""
        data = SimulationData(
            times=np.linspace(0, 1, 50),
            positions=np.random.randn(50, 3),
            velocities=np.random.randn(50, 3),
            torques=None,
        )
        fig = generator.generate_single_plot(data, PlotType.JOINT_TORQUES)
        assert fig is None

    def test_no_contact_forces(self, generator: PlotGenerator) -> None:
        """Test plot with missing contact forces."""
        data = SimulationData(
            times=np.linspace(0, 1, 50),
            positions=np.random.randn(50, 3),
            velocities=np.random.randn(50, 3),
            contact_forces=None,
        )
        fig = generator.generate_single_plot(data, PlotType.CONTACT_FORCES)
        assert fig is None

    def test_no_energies(self, generator: PlotGenerator) -> None:
        """Test plot with missing energy data."""
        data = SimulationData(
            times=np.linspace(0, 1, 50),
            positions=np.random.randn(50, 3),
            velocities=np.random.randn(50, 3),
        )
        fig = generator.generate_single_plot(data, PlotType.ENERGY)
        assert fig is None


# ---- Custom Config Tests ----


class TestCustomConfig:
    """Tests for custom plot configurations."""

    def test_subset_joint_indices(self, sample_data: SimulationData) -> None:
        """Test plotting only specific joints."""
        config = PlotConfig(joint_indices=[0, 2])
        gen = PlotGenerator(config)
        fig = gen.generate_single_plot(sample_data, PlotType.JOINT_POSITIONS)
        assert fig is not None

    def test_custom_figsize(self, sample_data: SimulationData) -> None:
        """Test custom figure size."""
        config = PlotConfig(figsize=(16, 10))
        gen = PlotGenerator(config)
        fig = gen.generate_single_plot(sample_data, PlotType.JOINT_POSITIONS)
        assert fig is not None

    def test_specific_plot_types(self, sample_data: SimulationData) -> None:
        """Test generating only specific plot types."""
        config = PlotConfig(plot_types=[PlotType.JOINT_POSITIONS, PlotType.ENERGY])
        gen = PlotGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = gen.generate_standard_plots(sample_data, tmpdir)
            assert len(generated) == 2
