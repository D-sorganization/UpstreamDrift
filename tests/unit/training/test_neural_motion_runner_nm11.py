"""Unit tests for NM-11 Neural Motion Runner and Registry Integration.

Governing Issue: #10626
Parent Epic: #10603
Prerequisites: #10618 (NM-03), #10623 (NM-08)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
import pytest

from src.shared.python.training import (
    CancelToken,
    Dataset,
    DatasetRegistry,
    ProgressSink,
    RunResult,
    ThreadingCancelToken,
    TrainingConfig,
    TrainingFramework,
    TrainingJobRunner,
    TrainingStatus,
)
from src.shared.python.training.metrics import MetricKind, TrainingMetric
from src.shared.python.training.runtime import (
    InMemoryProgressSink,
    NoRunnerAvailableError,
    PyTorchCVAERunner,
    RunnerRegistry,
)
from src.shared.python.training.runtime.adapters.neural_motion import (
    KNOWN_NEURAL_MOTION_ENTRY_POINTS,
    NeuralMotionRunner,
)

pytestmark = pytest.mark.unit


def _make_config(
    entry_point: str = "neural_motion:train_masked_proposals",
    dataset_id: str | None = None,
    output_dir: Path | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point=entry_point,
        dataset_id=dataset_id,
        output_dir=output_dir or Path("/tmp/nm11_test_output"),
        hyperparameters=hyperparameters
        if hyperparameters is not None
        else {
            "model_id": "driven_double_pendulum",
            "epochs": 2,
            "lr": 0.001,
            "batch_size": 16,
            "latent_dim": 8,
        },
    )


def test_neural_motion_runner_conforms_to_protocol() -> None:
    """NeuralMotionRunner satisfies TrainingJobRunner protocol structurally."""
    runner = NeuralMotionRunner()
    assert isinstance(runner, TrainingJobRunner)
    assert runner.framework is TrainingFramework.PYTORCH


def test_runner_registry_multi_runner_resolution() -> None:
    """RunnerRegistry resolves different runners sharing TrainingFramework.PYTORCH."""
    registry = RunnerRegistry()
    cvae_runner = PyTorchCVAERunner()
    nm_runner = NeuralMotionRunner()

    registry.register(cvae_runner)
    registry.register(nm_runner)

    # CVAE config resolves to cvae_runner
    cvae_config = TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="motion_matching.inverse:train_inverse_cvae",
        output_dir=Path("/tmp/cvae"),
    )
    resolved_cvae = registry.resolve(cvae_config)
    assert isinstance(resolved_cvae, PyTorchCVAERunner)

    # Neural motion config resolves to nm_runner
    nm_config = _make_config()
    resolved_nm = registry.resolve(nm_config)
    assert isinstance(resolved_nm, NeuralMotionRunner)

    # Unknown entry point raises NoRunnerAvailableError
    unknown_config = TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="unknown:train",
        output_dir=Path("/tmp/unknown"),
    )
    with pytest.raises(NoRunnerAvailableError):
        registry.resolve(unknown_config)


def test_can_run_entry_point_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    """can_run accepts known neural motion entry points and checks torch spec."""
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "torch" else None
    )
    runner = NeuralMotionRunner()

    for ep in KNOWN_NEURAL_MOTION_ENTRY_POINTS:
        cfg = _make_config(entry_point=ep)
        assert runner.can_run(cfg) is True

    # Rejects foreign entry points
    assert (
        runner.can_run(
            _make_config(entry_point="motion_matching.inverse:train_inverse_cvae")
        )
        is False
    )
    assert runner.can_run(_make_config(entry_point="other:point")) is False

    # Rejects foreign frameworks
    non_pytorch = TrainingConfig(
        framework=TrainingFramework.GYMNASIUM,
        entry_point="neural_motion:train_masked_proposals",
        output_dir=Path("/tmp/gym"),
    )
    assert runner.can_run(non_pytorch) is False


def test_can_run_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """can_run returns False cleanly without raising when torch is not installed."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    runner = NeuralMotionRunner()
    assert runner.can_run(_make_config()) is False


def test_prepare_validates_hyperparameters(tmp_path: Path) -> None:
    """prepare rejects malformed or missing critical hyperparameters."""
    runner = NeuralMotionRunner()

    # Missing model_id
    bad_cfg = _make_config(output_dir=tmp_path / "out", hyperparameters={"epochs": 5})
    with pytest.raises(ValueError, match="model_id"):
        runner.prepare(bad_cfg)

    # Negative epochs
    bad_cfg2 = _make_config(
        output_dir=tmp_path / "out", hyperparameters={"model_id": "test", "epochs": -1}
    )
    with pytest.raises(ValueError, match="epochs"):
        runner.prepare(bad_cfg2)

    # Zero or negative learning rate
    bad_cfg3 = _make_config(
        output_dir=tmp_path / "out",
        hyperparameters={"model_id": "test", "epochs": 2, "lr": 0.0},
    )
    with pytest.raises(ValueError, match="lr"):
        runner.prepare(bad_cfg3)


def test_prepare_resolves_dataset(tmp_path: Path) -> None:
    """prepare validates that dataset_id exists in DatasetRegistry."""
    dataset_registry = DatasetRegistry()
    runner = NeuralMotionRunner(dataset_registry=dataset_registry)

    # Missing dataset_id raises KeyError
    missing_cfg = _make_config(
        output_dir=tmp_path / "out", dataset_id="nonexistent_dataset"
    )
    with pytest.raises(KeyError, match="nonexistent_dataset"):
        runner.prepare(missing_cfg)


def test_run_streams_metrics_and_emits_card(tmp_path: Path) -> None:
    """run executes training loop, pushes metrics to sink, and outputs model card."""
    out_dir = tmp_path / "training_run"
    runner = NeuralMotionRunner()
    cfg = _make_config(output_dir=out_dir)

    sink = InMemoryProgressSink()
    cancel = ThreadingCancelToken()

    result = runner.run(cfg, progress=sink, cancel=cancel)
    assert result.status == TrainingStatus.COMPLETED
    assert result.duration_s >= 0.0

    # Verify metrics stream
    metrics = sink.metrics
    assert len(metrics) > 0
    kinds = {m.kind for m in metrics}
    assert MetricKind.LOSS in kinds

    # Verify artifacts emitted
    card_path = out_dir / "model_card.json"
    assert card_path.exists()
    assert (out_dir / "model_checkpoint.pt").exists() or (
        out_dir / "weights.json"
    ).exists()


def test_run_honors_cancellation(tmp_path: Path) -> None:
    """run honors CancelToken cooperative cancellation promptly."""
    out_dir = tmp_path / "cancel_run"
    runner = NeuralMotionRunner()
    cfg = _make_config(
        output_dir=out_dir,
        hyperparameters={
            "model_id": "driven_double_pendulum",
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 16,
        },
    )

    sink = InMemoryProgressSink()
    cancel = ThreadingCancelToken()
    cancel.request_cancel()

    result = runner.run(cfg, progress=sink, cancel=cancel)
    assert result.status == TrainingStatus.CANCELLED
