"""NM-01 (#10616): real reuse of fit_result / provider / TrainingConfig."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.provider import classical_baseline_fit_options
from src.shared.python.neural_motion.experiment import (
    classical_baseline_latency_s,
    pilot_training_config,
)
from src.shared.python.neural_motion.tasks import (
    build_default_learning_tasks,
    labels_from_canonical_fit,
    provider_engine_name,
    refinement_fit_options,
    training_config_for_task,
)
from src.shared.python.training.config import TrainingFramework

pytestmark = pytest.mark.unit


def _fit(wall_clock_s: float = 0.5) -> CanonicalFitResult:
    return CanonicalFitResult(
        theta_optimal=np.asarray([0.1, -0.2], dtype=np.float64),
        final_cost=1.0,
        final_rmse_m=0.01,
        solver_status="success",
        iterations=1,
        n_evaluations=1,
        wall_clock_s=wall_clock_s,
        message="",
        history=(),
        method="test",
        git_commit="deadbeef",
        engine_version="0",
        target_hash="abc",
        timestamp_utc="2026-09-21T00:00:00Z",
    )


def test_labels_and_refinement_reuse_anchors() -> None:
    assert labels_from_canonical_fit(_fit()) == (0.1, -0.2)
    assert refinement_fit_options(seed=2, maxiter=50) == classical_baseline_fit_options(
        maxiter=50, rng_seed=2
    )
    assert provider_engine_name(SimpleNamespace(engine_name="mujoco")) == "mujoco"


def test_training_and_latency_reuse_anchors(tmp_path: Path) -> None:
    task = build_default_learning_tasks("driven_double_pendulum")[0]
    config = training_config_for_task(task, output_dir=tmp_path, seed=1)
    assert config.framework is TrainingFramework.PYTORCH
    assert config.tags["slice"] == "nm01"
    assert classical_baseline_latency_s(_fit(1.25)) == pytest.approx(1.25)
    pilot = pilot_training_config(
        output_dir=tmp_path, model_id="driven_double_pendulum", seed=0
    )
    assert pilot.tags["governing_issue"] == "#10616"
