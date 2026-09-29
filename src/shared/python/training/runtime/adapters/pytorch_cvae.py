"""PyTorch :class:`TrainingJobRunner` for the inverse-CVAE swing model.

Wraps :func:`motion_matching.inverse.training.train_inverse_cvae` so the
training-controller can schedule it like any other framework adapter.
Streams per-epoch metrics through :class:`ProgressSink` and honours
cooperative cancellation via :class:`CancelToken`.

The adapter is deliberately thin: all model code lives in the existing
training loop. The two surfaces this module owns are

1. ``hyperparameters`` → ``motion_matching.inverse.training.TrainingConfig``
   translation (with defensive validation of types / ranges), and
2. ``EpochMetrics`` → ``tuple[TrainingMetric, ...]`` fan-out, tagged so
   the dashboard's :func:`filter_by_tags` helper can split train vs val.

PyTorch is imported lazily — :meth:`can_run` uses
:func:`importlib.util.find_spec` so this adapter can be registered in a
torch-less environment without raising. :meth:`prepare` performs the
real ``import torch`` and surfaces the failure as a clean
``ImportError`` (which the :class:`InProcessDriver` converts to a
``FAILED`` :class:`RunResult`).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.logging_pkg.logging_config import get_logger

from ...config import TrainingConfig, TrainingFramework
from ...contracts import CancelToken, ProgressSink
from ...datasets import DatasetRegistry
from ...identifiers import new_run_id
from ...job import RunResult
from ...metrics import MetricKind, TrainingMetric
from ...status import TrainingStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.shared.python.motion_matching.inverse.training import (
        EpochMetrics,
        TrainingResult,
    )

__all__ = ["PyTorchCVAERunner"]


logger = get_logger(__name__)


_KNOWN_ENTRY_POINTS: frozenset[str] = frozenset(
    {"motion_matching.inverse:train_inverse_cvae"}
)
"""Entry-point identifiers this adapter knows how to drive."""


# Hyperparameter keys the adapter accepts. Anything else is ignored with
# a debug log so callers can pass framework-specific extras without the
# adapter erroring — but the *recognised* set is small and explicit.
_HP_EPOCHS = "epochs"
_HP_LR = "lr"
_HP_BATCH_SIZE = "batch_size"
_HP_PATIENCE = "patience"
_HP_KL_ANNEAL = "kl_anneal_epochs"
_HP_MAX_BETA = "max_beta"
_HP_FREE_BITS = "free_bits"
_HP_VAL_FRACTION = "val_fraction"
_HP_WEIGHT_DECAY = "weight_decay"
_HP_GRAD_CLIP = "grad_clip"
_HP_DEVICE = "device"

_RECOGNISED_HPS: frozenset[str] = frozenset(
    {
        _HP_EPOCHS,
        _HP_LR,
        _HP_BATCH_SIZE,
        _HP_PATIENCE,
        _HP_KL_ANNEAL,
        _HP_MAX_BETA,
        _HP_FREE_BITS,
        _HP_VAL_FRACTION,
        _HP_WEIGHT_DECAY,
        _HP_GRAD_CLIP,
        _HP_DEVICE,
    }
)


class PyTorchCVAERunner:
    """Adapter wiring the inverse-CVAE training loop into the controller.

    Args:
        dataset_registry: Optional :class:`DatasetRegistry` for resolving
            ``config.dataset_id`` into a filesystem path. When ``None``,
            the adapter trusts ``config.hyperparameters["dataset_path"]``
            (or fails in :meth:`prepare` if the config sets a
            ``dataset_id`` it cannot resolve).
        dataset_loader: Optional callable ``(Path) -> CompactSwingDataset``
            passed through to the training loop. Lets tests inject an
            in-memory fixture without touching parquet IO. Defaults to
            ``None`` (the loop picks its built-in loader).

    Conforms to :class:`TrainingJobRunner` structurally.
    """

    KNOWN_ENTRY_POINTS = _KNOWN_ENTRY_POINTS
    framework: TrainingFramework = TrainingFramework.PYTORCH

    __slots__ = ("_dataset_loader", "_dataset_registry")

    def __init__(
        self,
        dataset_registry: DatasetRegistry | None = None,
        *,
        dataset_loader: Any = None,
    ) -> None:
        if dataset_registry is not None and not isinstance(
            dataset_registry, DatasetRegistry
        ):
            raise TypeError(
                "dataset_registry must be a DatasetRegistry or None "
                f"(got {type(dataset_registry).__name__})"
            )
        self._dataset_registry = dataset_registry
        self._dataset_loader = dataset_loader

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def can_run(self, config: TrainingConfig) -> bool:
        """Accept only the inverse-CVAE entry point and only when torch is importable."""

        if not isinstance(config, TrainingConfig):
            return False
        if config.framework is not TrainingFramework.PYTORCH:
            return False
        if config.entry_point not in self.KNOWN_ENTRY_POINTS:
            return False
        return importlib.util.find_spec("torch") is not None

    def prepare(self, config: TrainingConfig) -> None:
        """Validate environment + filesystem before :meth:`run`.

        - Confirms ``import torch`` succeeds (surfacing a clear
          ImportError otherwise).
        - Creates ``config.output_dir`` (parents=True, exist_ok=True).
        - Resolves ``config.dataset_id`` against the registry when one
          was supplied; raises :class:`KeyError`-derived error with a
          human-friendly message when the id is unknown.
        """

        importlib.import_module("torch")  # raises ImportError if missing
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if (
            config.dataset_id is not None
            and self._dataset_registry is not None
            and not self._dataset_registry.has(config.dataset_id)
        ):
            raise LookupError(
                f"dataset_id {config.dataset_id!r} is not registered "
                "in the supplied DatasetRegistry; register the "
                "dataset before submitting the job"
            )

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        """Execute the training loop, streaming progress through ``progress``."""

        # Local import so the module remains import-clean when torch is
        # absent — the driver's _RUNNER_FAILURE_TYPES would otherwise
        # see an ImportError at module-load time.
        from src.shared.python.motion_matching.inverse.training import (
            train_inverse_cvae,
        )

        dataset_path = self._resolve_dataset_path(config)
        cvae_cfg = self._build_cvae_config(config.hyperparameters, config)
        run_id = new_run_id()
        emitted: list[TrainingMetric] = []

        def _on_epoch_end(metrics: EpochMetrics) -> None:
            for metric in self._metrics_for_epoch(metrics):
                emitted.append(metric)
                progress.emit_metric(metric)

        progress.emit_status(
            TrainingStatus.RUNNING,
            message=f"starting inverse-CVAE run ({cvae_cfg.epochs} epochs)",
        )
        wall_start = time.monotonic()
        result: TrainingResult = train_inverse_cvae(
            dataset_path,
            epochs=cvae_cfg.epochs,
            batch_size=cvae_cfg.batch_size,
            lr=cvae_cfg.lr,
            device=cvae_cfg.device,
            seed=config.seed if config.seed is not None else cvae_cfg.seed,
            kl_anneal_epochs=cvae_cfg.kl_anneal_epochs,
            max_beta=cvae_cfg.max_beta,
            free_bits=cvae_cfg.free_bits,
            patience=cvae_cfg.patience,
            output_root=config.output_dir,
            val_fraction=cvae_cfg.val_fraction,
            cvae_config=cvae_cfg.cvae,
            dataset_loader=self._dataset_loader,
            on_epoch_end=_on_epoch_end,
            should_stop=lambda: cancel.is_cancelled,
        )
        duration = time.monotonic() - wall_start

        final_metrics = self._final_metrics(result, emitted)
        artifacts: tuple[Path, ...] = (
            result.checkpoint_path,
            result.output_dir / "metrics.json",
        )
        if cancel.is_cancelled:
            status = TrainingStatus.CANCELLED
            message = "cancelled by token; best-so-far checkpoint preserved"
        else:
            status = TrainingStatus.COMPLETED
            message = (
                f"completed {len(result.history)} epoch(s); "
                f"best epoch {result.best_epoch}"
            )
        progress.emit_status(status, message=message)
        return RunResult(
            run_id=run_id,
            status=status,
            duration_s=duration,
            final_metrics=final_metrics,
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dataset_path(self, config: TrainingConfig) -> Path:
        if config.dataset_id is not None and self._dataset_registry is not None:
            dataset = self._dataset_registry.get(config.dataset_id)
            return dataset.path
        # Fall back to an explicit hyperparameter override; tests use the
        # in-memory ``dataset_loader`` injection so the path argument is
        # ignored by the loop but still must be a Path.
        path_hp = config.hyperparameters.get("dataset_path")
        if isinstance(path_hp, (str, Path)):
            return Path(path_hp)
        return config.output_dir

    def _build_cvae_config(self, hyperparameters: Any, config: TrainingConfig) -> Any:
        """Translate ``config.hyperparameters`` into the CVAE TrainingConfig."""

        from src.shared.python.motion_matching.inverse.training import (
            TrainingConfig as CvaeTrainingConfig,
        )

        defaults = CvaeTrainingConfig()
        kwargs: dict[str, Any] = {}
        for key in _RECOGNISED_HPS:
            if key in hyperparameters:
                kwargs[key] = hyperparameters[key]
        # Honour the top-level cap if provided.
        if config.max_epochs is not None:
            requested = kwargs.get(_HP_EPOCHS, defaults.epochs)
            kwargs[_HP_EPOCHS] = min(int(requested), int(config.max_epochs))
        # Unrecognised keys: log once at debug so callers learn they were
        # ignored without polluting INFO.
        ignored = set(hyperparameters) - _RECOGNISED_HPS - {"dataset_path"}
        if ignored:
            logger.debug(
                "ignoring unrecognised cvae hyperparameters: %s",
                sorted(ignored),
            )
        return CvaeTrainingConfig(**kwargs)

    def _metrics_for_epoch(self, metrics: EpochMetrics) -> tuple[TrainingMetric, ...]:
        """Fan one ``EpochMetrics`` out into 6 :class:`TrainingMetric`s."""

        ts = time.time()
        step = int(metrics.epoch)
        train_tags = {"split": "train"}
        val_tags = {"split": "val"}
        return (
            TrainingMetric(
                name="train_recon",
                value=float(metrics.train_recon),
                step=step,
                timestamp=ts,
                kind=MetricKind.LOSS,
                tags=train_tags,
            ),
            TrainingMetric(
                name="train_kl",
                value=float(metrics.train_kl),
                step=step,
                timestamp=ts,
                kind=MetricKind.LOSS,
                tags=train_tags,
            ),
            TrainingMetric(
                name="val_recon",
                value=float(metrics.val_recon),
                step=step,
                timestamp=ts,
                kind=MetricKind.LOSS,
                tags=val_tags,
            ),
            TrainingMetric(
                name="val_kl",
                value=float(metrics.val_kl),
                step=step,
                timestamp=ts,
                kind=MetricKind.LOSS,
                tags=val_tags,
            ),
            TrainingMetric(
                name="beta",
                value=float(metrics.beta),
                step=step,
                timestamp=ts,
                kind=MetricKind.SCALAR,
                tags={"phase": "schedule"},
            ),
            TrainingMetric(
                name="duration_s",
                value=float(metrics.duration_s),
                step=step,
                timestamp=ts,
                kind=MetricKind.SCALAR,
                tags={"phase": "epoch"},
            ),
        )

    def _final_metrics(
        self, result: TrainingResult, emitted: list[TrainingMetric]
    ) -> tuple[TrainingMetric, ...]:
        if result.history:
            return self._metrics_for_epoch(result.history[-1])
        # Defensive fallback: nothing trained — return whatever we
        # already emitted (likely empty) so RunResult invariants hold.
        return tuple(emitted)
