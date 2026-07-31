"""Contract tests for #2456: dataset_generator.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
DATA_IO_DIR = REPO / "src/shared/python/data_io"
LOC_BUDGET = 600


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestDatasetGeneratorSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_dataset_models_module_exists(self) -> None:
        assert (DATA_IO_DIR / "_dataset_models.py").exists()

    @pytest.mark.unit
    def test_dataset_export_mixin_module_exists(self) -> None:
        assert (DATA_IO_DIR / "_dataset_export_mixin.py").exists()


class TestDatasetGeneratorFileSizes:
    """Each file must be under 600 LOC after split."""

    @pytest.mark.unit
    def test_dataset_generator_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(DATA_IO_DIR / "dataset_generator/core.py")
        assert (
            loc <= LOC_BUDGET
        ), f"dataset_generator.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_dataset_generator_split_2456_models_loc(self) -> None:
        loc = _count_lines(DATA_IO_DIR / "_dataset_models.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_dataset_models.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_export_mixin_loc(self) -> None:
        loc = _count_lines(DATA_IO_DIR / "_dataset_export_mixin.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_dataset_export_mixin.py has {loc} LOC; budget {LOC_BUDGET}"


class TestDatasetGeneratorPublicAPI:
    """Public API must remain importable from dataset_generator (backward compat)."""

    @pytest.mark.unit
    def test_import_parameter_range(self) -> None:
        from src.shared.python.data_io.dataset_generator import ParameterRange

        assert ParameterRange is not None

    @pytest.mark.unit
    def test_import_control_profile(self) -> None:
        from src.shared.python.data_io.dataset_generator import ControlProfile

        assert ControlProfile is not None

    @pytest.mark.unit
    def test_import_generator_config(self) -> None:
        from src.shared.python.data_io.dataset_generator import GeneratorConfig

        assert GeneratorConfig is not None

    @pytest.mark.unit
    def test_import_simulation_sample(self) -> None:
        from src.shared.python.data_io.dataset_generator import SimulationSample

        assert SimulationSample is not None

    @pytest.mark.unit
    def test_import_training_dataset(self) -> None:
        from src.shared.python.data_io.dataset_generator import TrainingDataset

        assert TrainingDataset is not None

    @pytest.mark.unit
    def test_import_dataset_generator(self) -> None:
        from src.shared.python.data_io.dataset_generator import DatasetGenerator

        assert DatasetGenerator is not None
