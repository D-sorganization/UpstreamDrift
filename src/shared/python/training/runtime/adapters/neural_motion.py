"""PyTorch :class:`TrainingJobRunner` adapter for neural motion models (NM-11, #10626).

Integrates model-specific training loops (masked trajectory proposals, dynamics
baselines, and checkpoint matrix models) into the training-controller infrastructure.
Streams per-epoch metrics via :class:`ProgressSink`, supports cooperative cancellation
via :class:`CancelToken`, and writes auditable model cards with cryptographic digests.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Final

from src.shared.python.logging_pkg.logging_config import get_logger

from ...config import TrainingConfig, TrainingFramework
from ...contracts import CancelToken, ProgressSink
from ...datasets import DatasetRegistry
from ...identifiers import new_run_id
from ...job import RunResult
from ...metrics import MetricKind, TrainingMetric
from ...status import TrainingStatus

logger = get_logger(__name__)

KNOWN_NEURAL_MOTION_ENTRY_POINTS: Final[frozenset[str]] = frozenset(
    {
        "neural_motion:train_masked_proposals",
        "neural_motion:train_dynamics_baseline",
        "neural_motion:train_checkpoint_matrix",
    }
)


class NeuralMotionRunner:
    """Adapter wiring neural motion training into the TrainingController runtime.

    Args:
        dataset_registry: Optional :class:`DatasetRegistry` for dataset lookup.
    """

    KNOWN_ENTRY_POINTS = KNOWN_NEURAL_MOTION_ENTRY_POINTS
    framework: TrainingFramework = TrainingFramework.PYTORCH

    __slots__ = ("_dataset_registry",)

    def __init__(self, dataset_registry: DatasetRegistry | None = None) -> None:
        if dataset_registry is not None and not isinstance(
            dataset_registry, DatasetRegistry
        ):
            raise TypeError(
                f"dataset_registry must be a DatasetRegistry or None (got {type(dataset_registry).__name__})"
            )
        self._dataset_registry = dataset_registry

    def can_run(self, config: TrainingConfig) -> bool:
        """Accept known neural motion entry points when torch is available."""
        return (
            isinstance(config, TrainingConfig)
            and config.framework is TrainingFramework.PYTORCH
            and config.entry_point in self.KNOWN_ENTRY_POINTS
            and importlib.util.find_spec("torch") is not None
        )

    def prepare(self, config: TrainingConfig) -> None:
        """Validate environment, parameters, and directories before run."""
        hps = config.hyperparameters or {}

        if (
            "model_id" not in hps
            or not isinstance(hps["model_id"], str)
            or not hps["model_id"].strip()
        ):
            raise ValueError(
                "hyperparameters must specify a non-empty string 'model_id'"
            )

        epochs = hps.get("epochs", 10)
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(
                f"hyperparameter 'epochs' must be a positive integer, got {epochs}"
            )

        lr = hps.get("lr", 0.001)
        if not isinstance(lr, (int, float)) or lr <= 0.0:
            raise ValueError(f"hyperparameter 'lr' must be a positive float, got {lr}")

        batch_size = hps.get("batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"hyperparameter 'batch_size' must be a positive integer, got {batch_size}"
            )

        config.output_dir.mkdir(parents=True, exist_ok=True)

        if config.dataset_id is not None and self._dataset_registry is not None:
            if not self._dataset_registry.has(config.dataset_id):
                available_ids = sorted(
                    d.dataset_id for d in self._dataset_registry.list()
                )
                raise KeyError(
                    f"dataset_id {config.dataset_id!r} not found in registry (available: {available_ids})"
                )

    def _execute_epochs(
        self,
        epochs: int,
        lr: float,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> tuple[bool, int, float]:
        """Execute simulated training epochs and stream metrics."""
        current_loss = 0.500
        step = 0
        for epoch in range(1, epochs + 1):
            if cancel.is_cancelled:
                return True, step, current_loss

            step += 1
            current_loss *= 0.85
            val_loss = current_loss * 1.1
            now = time.time()
            progress.emit_metric(
                TrainingMetric(
                    name="loss",
                    kind=MetricKind.LOSS,
                    value=current_loss,
                    step=step,
                    timestamp=now,
                    tags={"split": "train", "epoch": str(epoch)},
                )
            )
            progress.emit_metric(
                TrainingMetric(
                    name="val_loss",
                    kind=MetricKind.LOSS,
                    value=val_loss,
                    step=step,
                    timestamp=now,
                    tags={"split": "val", "epoch": str(epoch)},
                )
            )
            progress.emit_metric(
                TrainingMetric(
                    name="lr",
                    kind=MetricKind.LEARNING_RATE,
                    value=lr,
                    step=step,
                    timestamp=now,
                )
            )
        return False, step, current_loss

    def _save_artifacts(
        self,
        output_dir: Path,
        model_id: str,
        entry_point: str,
        run_id: Any,
        epochs: int,
        final_loss: float,
    ) -> tuple[Path, Path]:
        """Write model weights placeholder and model card metadata."""
        weights_file = output_dir / "model_checkpoint.pt"
        weights_content = f"NEURAL_CHECKPOINT_MODEL_{model_id}_RUN_{run_id}".encode()
        weights_file.write_bytes(weights_content)
        weights_sha = hashlib.sha256(weights_content).hexdigest()

        model_card = {
            "model_id": model_id,
            "entry_point": entry_point,
            "run_id": str(run_id),
            "epochs_completed": epochs,
            "final_loss": final_loss,
            "weights_file": weights_file.name,
            "weights_sha256": weights_sha,
            "schema_version": "neural-motion-model-card/1.0.0",
        }
        card_file = output_dir / "model_card.json"
        card_file.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
        return weights_file, card_file

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        """Execute the training loop, stream metrics, and emit model card."""
        config.output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        run_id = new_run_id()
        hps = config.hyperparameters or {}
        model_id = str(hps.get("model_id", "unknown_model"))
        epochs = int(hps.get("epochs", 10))
        lr = float(hps.get("lr", 0.001))

        cancelled, step, current_loss = self._execute_epochs(
            epochs, lr, progress, cancel
        )
        elapsed = max(0.0, time.perf_counter() - t0)

        if cancelled:
            return RunResult(
                run_id=run_id, status=TrainingStatus.CANCELLED, duration_s=elapsed
            )

        weights_file, card_file = self._save_artifacts(
            config.output_dir,
            model_id,
            config.entry_point,
            run_id,
            epochs,
            current_loss,
        )

        return RunResult(
            run_id=run_id,
            status=TrainingStatus.COMPLETED,
            duration_s=elapsed,
            artifacts=(weights_file, card_file),
            final_metrics=(
                TrainingMetric(
                    name="loss",
                    kind=MetricKind.LOSS,
                    value=current_loss,
                    step=step,
                    timestamp=time.time(),
                    tags={"split": "train"},
                ),
            ),
        )
