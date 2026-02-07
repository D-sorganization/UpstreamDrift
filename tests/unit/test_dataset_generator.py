"""Tests for the Dataset Generator module.

Tests dataset generation, parameter variation, export to multiple formats,
and integration with the PhysicsEngine protocol.

Follows TDD and Design by Contract principles.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.dataset_generator import (
    ControlProfile,
    DatasetGenerator,
    GeneratorConfig,
    ParameterRange,
    TrainingDataset,
)
from src.shared.python.mock_engine import MockPhysicsEngine

# ---- Fixtures ----


@pytest.fixture
def mock_engine() -> MockPhysicsEngine:
    """Create and initialize a mock physics engine."""
    engine = MockPhysicsEngine(num_joints=4)
    engine.load_from_string("<mock/>")
    return engine


@pytest.fixture
def generator(mock_engine: MockPhysicsEngine) -> DatasetGenerator:
    """Create a dataset generator with mock engine."""
    return DatasetGenerator(mock_engine)


@pytest.fixture
def basic_config() -> GeneratorConfig:
    """Create a basic generator configuration."""
    return GeneratorConfig(
        num_samples=3,
        duration=0.1,
        timestep=0.01,
        seed=42,
    )


# ---- ParameterRange Tests ----


class TestParameterRange:
    """Tests for ParameterRange dataclass."""

    def test_valid_range(self) -> None:
        """Test creating a valid parameter range."""
        pr = ParameterRange(name="test", min_val=0.0, max_val=1.0)
        assert pr.name == "test"
        assert pr.min_val == 0.0
        assert pr.max_val == 1.0

    def test_invalid_range_raises(self) -> None:
        """Precondition: min_val must be <= max_val."""
        with pytest.raises(ValueError, match="min_val.*max_val"):
            ParameterRange(name="bad", min_val=2.0, max_val=1.0)

    def test_invalid_distribution_raises(self) -> None:
        """Precondition: distribution must be a known type."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            ParameterRange(name="bad", min_val=0.0, max_val=1.0, distribution="invalid")

    def test_uniform_sampling(self) -> None:
        """Postcondition: sampled values are within range."""
        rng = np.random.default_rng(42)
        pr = ParameterRange(
            name="test", min_val=-1.0, max_val=1.0, distribution="uniform"
        )
        for _ in range(100):
            val = pr.sample(rng)
            assert -1.0 <= val <= 1.0

    def test_normal_sampling_clipped(self) -> None:
        """Postcondition: normal samples are clipped to range."""
        rng = np.random.default_rng(42)
        pr = ParameterRange(
            name="test", min_val=0.0, max_val=1.0, distribution="normal"
        )
        for _ in range(100):
            val = pr.sample(rng)
            assert 0.0 <= val <= 1.0

    def test_linspace_sampling(self) -> None:
        """Postcondition: linspace samples are from discrete set."""
        rng = np.random.default_rng(42)
        pr = ParameterRange(
            name="test",
            min_val=0.0,
            max_val=1.0,
            distribution="linspace",
            num_points=5,
        )
        expected = np.linspace(0.0, 1.0, 5)
        for _ in range(20):
            val = pr.sample(rng)
            assert any(abs(val - e) < 1e-10 for e in expected)

    def test_linspace_generation(self) -> None:
        """Test linspace array generation."""
        pr = ParameterRange(name="test", min_val=0.0, max_val=1.0, num_points=5)
        points = pr.linspace()
        assert len(points) == 5
        np.testing.assert_allclose(points, [0.0, 0.25, 0.5, 0.75, 1.0])


# ---- ControlProfile Tests ----


class TestControlProfile:
    """Tests for ControlProfile."""

    def test_zero_profile(self) -> None:
        """Test zero control profile generates zeros."""
        profile = ControlProfile(name="zero", profile_type="zero")
        rng = np.random.default_rng(42)
        result = profile.generate(4, 10, 0.01, rng)
        assert result.shape == (10, 4)
        np.testing.assert_array_equal(result, 0.0)

    def test_constant_profile(self) -> None:
        """Test constant control profile."""
        profile = ControlProfile(
            name="const",
            profile_type="constant",
            parameters={"magnitude": 5.0},
        )
        rng = np.random.default_rng(42)
        result = profile.generate(3, 10, 0.01, rng)
        assert result.shape == (10, 3)
        np.testing.assert_array_equal(result, 5.0)

    def test_sinusoidal_profile(self) -> None:
        """Test sinusoidal control profile."""
        profile = ControlProfile(
            name="sin",
            profile_type="sinusoidal",
            parameters={"frequency": 1.0, "amplitude": 2.0},
        )
        rng = np.random.default_rng(42)
        result = profile.generate(2, 100, 0.01, rng)
        assert result.shape == (100, 2)
        assert np.max(np.abs(result)) <= 2.0 + 1e-10

    def test_random_profile(self) -> None:
        """Test random control profile."""
        profile = ControlProfile(
            name="rand",
            profile_type="random",
            parameters={"scale": 1.0},
        )
        rng = np.random.default_rng(42)
        result = profile.generate(4, 50, 0.01, rng)
        assert result.shape == (50, 4)
        assert np.std(result) > 0  # Not all zeros

    def test_step_profile(self) -> None:
        """Test step control profile."""
        profile = ControlProfile(
            name="step",
            profile_type="step",
            parameters={"magnitude": 10.0, "step_time": 0.05},
        )
        rng = np.random.default_rng(42)
        result = profile.generate(3, 20, 0.01, rng)
        assert result.shape == (20, 3)
        # Before step time (step_idx=5): zeros
        np.testing.assert_array_equal(result[:5], 0.0)
        # After step time: magnitude
        np.testing.assert_array_equal(result[5:], 10.0)


# ---- GeneratorConfig Tests ----


class TestGeneratorConfig:
    """Tests for GeneratorConfig validation."""

    def test_valid_config(self) -> None:
        """Test creating a valid configuration."""
        config = GeneratorConfig(num_samples=10, duration=1.0, timestep=0.001)
        assert config.num_samples == 10

    def test_invalid_num_samples(self) -> None:
        """Precondition: num_samples must be positive."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            GeneratorConfig(num_samples=0)

    def test_invalid_duration(self) -> None:
        """Precondition: duration must be positive."""
        with pytest.raises(ValueError, match="duration must be positive"):
            GeneratorConfig(duration=-1.0)

    def test_invalid_timestep(self) -> None:
        """Precondition: timestep must be positive."""
        with pytest.raises(ValueError, match="timestep must be positive"):
            GeneratorConfig(timestep=0.0)


# ---- DatasetGenerator Tests ----


class TestDatasetGenerator:
    """Tests for DatasetGenerator core functionality."""

    def test_generate_basic(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Test basic dataset generation produces correct structure."""
        dataset = generator.generate(basic_config)

        assert isinstance(dataset, TrainingDataset)
        assert dataset.num_samples == 3
        assert dataset.total_frames > 0
        assert len(dataset.joint_names) == 4

    def test_generate_samples_have_correct_shape(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Postcondition: sample arrays have correct dimensions."""
        dataset = generator.generate(basic_config)

        for sample in dataset.samples:
            n_steps = len(sample.times)
            assert n_steps > 0
            assert sample.positions.shape == (n_steps, 4)
            assert sample.velocities.shape == (n_steps, 4)
            assert sample.accelerations.shape == (n_steps, 4)
            assert sample.torques.shape == (n_steps, 4)

    def test_generate_with_mass_matrix(self, generator: DatasetGenerator) -> None:
        """Test that mass matrix recording works."""
        config = GeneratorConfig(
            num_samples=2, duration=0.05, timestep=0.01, record_mass_matrix=True
        )
        dataset = generator.generate(config)

        for sample in dataset.samples:
            assert sample.mass_matrices is not None
            n_steps = len(sample.times)
            assert sample.mass_matrices.shape == (n_steps, 4, 4)

    def test_generate_with_drift_control(self, generator: DatasetGenerator) -> None:
        """Test drift/control decomposition recording."""
        config = GeneratorConfig(
            num_samples=2, duration=0.05, timestep=0.01, record_drift_control=True
        )
        dataset = generator.generate(config)

        for sample in dataset.samples:
            assert sample.drift_accelerations is not None
            assert sample.control_accelerations is not None

    def test_generate_reproducibility(self, mock_engine: MockPhysicsEngine) -> None:
        """Invariant: same seed produces same dataset."""
        config = GeneratorConfig(num_samples=3, duration=0.05, timestep=0.01, seed=123)

        gen1 = DatasetGenerator(mock_engine)
        dataset1 = gen1.generate(config)

        mock_engine.reset()
        gen2 = DatasetGenerator(mock_engine)
        dataset2 = gen2.generate(config)

        assert dataset1.num_samples == dataset2.num_samples
        for s1, s2 in zip(dataset1.samples, dataset2.samples):
            np.testing.assert_array_almost_equal(s1.positions, s2.positions)

    def test_generate_with_position_variation(
        self, generator: DatasetGenerator
    ) -> None:
        """Test that initial positions are varied between samples."""
        config = GeneratorConfig(
            num_samples=5,
            duration=0.02,
            timestep=0.01,
            vary_initial_positions=True,
        )
        dataset = generator.generate(config)

        # Check that initial positions differ between samples
        initial_positions = [s.positions[0] for s in dataset.samples]
        # At least some should differ
        all_same = all(
            np.allclose(initial_positions[0], ip) for ip in initial_positions[1:]
        )
        assert not all_same, "Expected varied initial positions"

    def test_generate_with_custom_ranges(self, generator: DatasetGenerator) -> None:
        """Test parameter ranges are respected."""
        config = GeneratorConfig(
            num_samples=10,
            duration=0.02,
            timestep=0.01,
            vary_initial_positions=True,
            position_ranges=[
                ParameterRange(name="all", min_val=-0.1, max_val=0.1),
            ],
        )
        dataset = generator.generate(config)

        for sample in dataset.samples:
            q0 = sample.positions[0]
            assert np.all(np.abs(q0) <= 0.1 + 1e-10)

    def test_generate_with_multiple_control_profiles(
        self, generator: DatasetGenerator
    ) -> None:
        """Test generation with multiple control profiles."""
        config = GeneratorConfig(
            num_samples=5,
            duration=0.05,
            timestep=0.01,
            control_profiles=[
                ControlProfile(name="zero"),
                ControlProfile(
                    name="rand", profile_type="random", parameters={"scale": 0.5}
                ),
            ],
        )
        dataset = generator.generate(config)
        assert dataset.num_samples == 5

    def test_generate_progress_callback(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Test that progress callback is called."""
        progress_calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        generator.generate(basic_config, progress_callback=on_progress)
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    def test_dataset_properties(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Test TrainingDataset property methods."""
        dataset = generator.generate(basic_config)
        assert dataset.num_samples == 3
        assert dataset.total_frames == sum(len(s.times) for s in dataset.samples)
        assert dataset.creation_time > 0


# ---- Export Tests ----


class TestDatasetExport:
    """Tests for dataset export functionality."""

    def test_export_to_csv(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Test CSV export creates correct files."""
        dataset = generator.generate(basic_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = generator.export_to_csv(dataset, tmpdir)
            assert output_dir.exists()

            # Check sample files
            csv_files = list(Path(tmpdir).glob("sample_*.csv"))
            assert len(csv_files) == 3

            # Check metadata file
            meta_path = Path(tmpdir) / "metadata.json"
            assert meta_path.exists()

            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["num_samples"] == 3

    def test_export_to_sqlite(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Test SQLite export creates valid database."""
        dataset = generator.generate(basic_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = generator.export_to_sqlite(dataset, Path(tmpdir) / "test.db")
            assert db_path.exists()

            # Verify database structure
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM samples")
            assert cursor.fetchone()[0] == 3

            cursor.execute("SELECT COUNT(*) FROM frames")
            assert cursor.fetchone()[0] > 0

            cursor.execute("SELECT value FROM dataset_metadata WHERE key='model_name'")
            assert cursor.fetchone() is not None

            conn.close()

    def test_export_generic_interface(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Test the generic export() method."""
        dataset = generator.generate(basic_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # CSV export
            csv_path = generator.export(dataset, Path(tmpdir) / "csv_out", format="csv")
            assert csv_path.exists()

            # SQLite export
            db_path = generator.export(dataset, Path(tmpdir) / "test", format="sqlite")
            assert db_path.exists()

    def test_export_invalid_format_raises(
        self, generator: DatasetGenerator, basic_config: GeneratorConfig
    ) -> None:
        """Test that invalid export format raises ValueError."""
        dataset = generator.generate(basic_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported export format"):
                generator.export(dataset, Path(tmpdir) / "test", format="invalid")
