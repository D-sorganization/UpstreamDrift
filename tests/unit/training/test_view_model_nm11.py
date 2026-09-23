"""Unit tests for NM-11 View Model items and Controller extensions.

Governing Issue: #10626
Parent Epic: #10603
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.shared.python.training import (
    CompatibilityChecker,
    Dataset,
    DatasetRegistry,
    JobRegistry,
    Scheduler,
    TrainingConfig,
    TrainingFramework,
)
from src.shared.python.training.runtime import InProcessDriver, RunnerRegistry
from src.tools.training_controller import (
    DatasetSchemaItem,
    ModelTopologyItem,
    TrainingDashboardController,
    default_neural_motion_topologies,
)

pytestmark = pytest.mark.unit


def test_model_topology_item_invariants() -> None:
    """ModelTopologyItem validates non-empty strings and types."""
    item = ModelTopologyItem(
        model_id="driven_double_pendulum",
        display_name="Driven Double Pendulum",
        framework="pytorch",
        default_entry_point="neural_motion:train_masked_proposals",
        description="Planar 2-DOF swing.",
    )
    assert item.model_id == "driven_double_pendulum"
    assert item.framework == "pytorch"

    with pytest.raises(ValueError, match="model_id"):
        ModelTopologyItem(
            model_id="",
            display_name="Bad",
            framework="pytorch",
            default_entry_point="ep",
        )

    with pytest.raises(ValueError, match="display_name"):
        ModelTopologyItem(
            model_id="id",
            display_name=" ",
            framework="pytorch",
            default_entry_point="ep",
        )


def test_dataset_schema_item_invariants() -> None:
    """DatasetSchemaItem validates non-empty strings and non-negative sample count."""
    item = DatasetSchemaItem(
        dataset_id="tour_swing_v1",
        display_name="Tour Swing V1",
        sample_count=1200,
        format="parquet",
    )
    assert item.sample_count == 1200
    assert item.format == "parquet"

    with pytest.raises(ValueError, match="sample_count"):
        DatasetSchemaItem(
            dataset_id="id",
            display_name="Name",
            sample_count=-5,
            format="parquet",
        )


def test_default_neural_motion_topologies() -> None:
    """default_neural_motion_topologies returns required topologies."""
    topologies = default_neural_motion_topologies()
    assert len(topologies) >= 3
    ids = {t.model_id for t in topologies}
    assert "driven_double_pendulum" in ids
    assert "mujoco_humanoid_3d" in ids
    assert "pinocchio_golf_arm" in ids


def test_controller_available_topologies_and_schemas(tmp_path: Path) -> None:
    """TrainingDashboardController exposes model topologies and registered datasets."""
    dataset_registry = DatasetRegistry()
    dataset_registry.register(
        Dataset(
            dataset_id="mock_dataset",
            name="Mock Dataset",
            format="json",
            path=tmp_path / "mock.json",
            size_bytes=42,
        )
    )

    runners = RunnerRegistry()
    driver = InProcessDriver(runners, max_workers=1)
    scheduler = Scheduler(registry=JobRegistry(), runners=runners, driver=driver)
    checker = CompatibilityChecker()

    controller = TrainingDashboardController(
        scheduler=scheduler,
        dataset_registry=dataset_registry,
        compatibility_checker=checker,
    )

    topologies = controller.available_model_topologies()
    assert len(topologies) >= 3
    assert any(t.model_id == "driven_double_pendulum" for t in topologies)

    schemas = controller.available_dataset_schemas()
    assert len(schemas) == 1
    assert schemas[0].dataset_id == "mock_dataset"
    assert schemas[0].sample_count == 42
    assert schemas[0].format == "json"
