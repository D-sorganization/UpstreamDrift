"""Multi-seed pilot training orchestration for NM-05 baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.neural_motion.episodes import (
    EpisodeStore,
    FamilySplitPlan,
    TrainOnlyNormalizer,
)
from src.shared.python.neural_motion.json_io import SortedJsonWritableMixin

from .classical import ClassicalMethod, fit_classical, predict_classical
from .dataset import TrialMatrixBundle, build_trial_matrices
from .neural import train_small_mlp
from .types import BASELINE_SCHEMA, DynamicsTaskKind, InverseLabelConditioning

__all__ = [
    "DynamicsBaselineTrainer",
    "DynamicsBaselineTrainerConfig",
    "PilotCheckpointCard",
]


@dataclass(frozen=True, slots=True)
class DynamicsBaselineTrainerConfig:
    """Validated inputs for :class:`DynamicsBaselineTrainer`."""

    store: EpisodeStore
    split: FamilySplitPlan
    task: DynamicsTaskKind
    seeds: tuple[int, ...]
    output_dir: Path
    active_dofs: tuple[int, ...] = (0,)
    normalizer: TrainOnlyNormalizer | None = None
    conditioning: InverseLabelConditioning | None = None
    require_mlp: bool = False
    overfit_smoke: bool = False

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        object.__setattr__(
            self,
            "active_dofs",
            tuple(int(i) for i in self.active_dofs),
        )


@dataclass(frozen=True, slots=True)
class PilotCheckpointCard(SortedJsonWritableMixin):
    """Reproducible pilot checkpoint / model card (software-contract receipt)."""

    schema: str
    task: str
    seeds: tuple[int, ...]
    method_order: tuple[str, ...]
    val_mse_by_method: dict[str, float]
    best_seed_metrics: dict[str, float]
    test_untouched: bool
    analytical_beats_mlp: bool
    normalizer_digest: str | None
    content_digest: str
    mlp_skipped_reason: str | None
    limitations: tuple[str, ...]
    checkpoint_dir: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task": self.task,
            "seeds": list(self.seeds),
            "method_order": list(self.method_order),
            "val_mse_by_method": dict(self.val_mse_by_method),
            "best_seed_metrics": dict(self.best_seed_metrics),
            "test_untouched": self.test_untouched,
            "analytical_beats_mlp": self.analytical_beats_mlp,
            "normalizer_digest": self.normalizer_digest,
            "content_digest": self.content_digest,
            "mlp_skipped_reason": self.mlp_skipped_reason,
            "limitations": list(self.limitations),
            "checkpoint_dir": self.checkpoint_dir,
        }


def _load_pilot_matrices(
    cfg: DynamicsBaselineTrainerConfig,
) -> tuple[TrialMatrixBundle, TrialMatrixBundle]:
    train = build_trial_matrices(
        cfg.store,
        cfg.split,
        task=cfg.task,
        split="train",
        active_dofs=cfg.active_dofs,
        conditioning=cfg.conditioning,
    )
    val = build_trial_matrices(
        cfg.store,
        cfg.split,
        task=cfg.task,
        split="val",
        active_dofs=cfg.active_dofs,
        conditioning=cfg.conditioning,
    )
    if train.features.shape[0] == 0:
        raise ValueError("train split is empty; refuse dynamics baseline fit")
    if val.features.shape[0] == 0:
        if not cfg.overfit_smoke:
            raise ValueError("val split is empty; refuse model selection")
        val = train
    return train, val


def _score_classical_methods(
    *,
    task: DynamicsTaskKind,
    train: TrialMatrixBundle,
    val: TrialMatrixBundle,
    active_dofs: tuple[int, ...],
) -> tuple[dict[str, float], list[str]]:
    val_scores: dict[str, float] = {}
    method_order: list[str] = []
    for method in (
        ClassicalMethod.ANALYTICAL_PENDULUM,
        ClassicalMethod.RIDGE,
        ClassicalMethod.NEAREST_NEIGHBOR,
    ):
        model = fit_classical(
            method,
            task=task,
            features=train.features,
            targets=train.targets,
            active_dofs=active_dofs,
        )
        pred = predict_classical(model, val.features)
        mse = float(np.mean((pred - val.targets) ** 2))
        val_scores[method.value] = mse
        method_order.append(method.value)
    return val_scores, method_order


@dataclass(frozen=True, slots=True)
class _MlpPilotState:
    val_scores: dict[str, float]
    method_order: list[str]
    best_train: float
    best_val: float
    best_seed: int
    mlp_reason: str | None


def _run_mlp_seeds(
    cfg: DynamicsBaselineTrainerConfig,
    train: TrialMatrixBundle,
    val: TrialMatrixBundle,
    val_scores: dict[str, float],
    method_order: list[str],
) -> _MlpPilotState:
    mlp_reason: str | None = None
    best_train = float("inf")
    best_val = float("inf")
    best_seed = cfg.seeds[0]
    epochs = 80 if cfg.overfit_smoke else 40

    for seed in cfg.seeds:
        result = train_small_mlp(
            train_x=train.features,
            train_y=train.targets,
            val_x=val.features,
            val_y=val.targets,
            seed=seed,
            epochs=epochs,
        )
        if result.skipped_reason:
            mlp_reason = result.skipped_reason
            if cfg.require_mlp:
                raise RuntimeError(f"MLP required but skipped: {mlp_reason}")
            continue
        val_scores["small_mlp"] = min(
            val_scores.get("small_mlp", float("inf")), result.val_mse
        )
        if "small_mlp" not in method_order:
            method_order.append("small_mlp")
        if result.val_mse < best_val:
            best_val = result.val_mse
            best_train = result.train_mse
            best_seed = seed
            if result.state is not None:
                np.savez(
                    cfg.output_dir / f"mlp_seed_{seed}.npz",
                    **result.state,
                )
    return _MlpPilotState(
        val_scores=val_scores,
        method_order=method_order,
        best_train=best_train,
        best_val=best_val,
        best_seed=best_seed,
        mlp_reason=mlp_reason,
    )


def _content_digest(
    cfg: DynamicsBaselineTrainerConfig,
    *,
    val_scores: dict[str, float],
    method_order: list[str],
    test_trial_count: int,
) -> str:
    payload = {
        "schema": BASELINE_SCHEMA,
        "task": cfg.task.value,
        "seeds": list(cfg.seeds),
        "method_order": method_order,
        "val_mse_by_method": val_scores,
        "test_trial_count_untouched": test_trial_count,
        "active_dofs": list(cfg.active_dofs),
        "normalizer_digest": (
            None if cfg.normalizer is None else cfg.normalizer.stats_digest
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assemble_pilot_card(
    cfg: DynamicsBaselineTrainerConfig,
    *,
    val_scores: dict[str, float],
    method_order: list[str],
    mlp_state: _MlpPilotState,
    test_trial_count: int,
) -> PilotCheckpointCard:
    analytical = val_scores[ClassicalMethod.ANALYTICAL_PENDULUM.value]
    mlp = val_scores.get("small_mlp")
    analytical_beats = mlp is None or analytical <= mlp
    digest = _content_digest(
        cfg,
        val_scores=val_scores,
        method_order=method_order,
        test_trial_count=test_trial_count,
    )
    best_train = mlp_state.best_train
    best_val = mlp_state.best_val
    return PilotCheckpointCard(
        schema=BASELINE_SCHEMA,
        task=cfg.task.value,
        seeds=cfg.seeds,
        method_order=tuple(method_order),
        val_mse_by_method={k: float(v) for k, v in val_scores.items()},
        best_seed_metrics={
            "seed": float(mlp_state.best_seed),
            "train_mse": float(best_train if np.isfinite(best_train) else analytical),
            "val_mse": float(best_val if np.isfinite(best_val) else analytical),
        },
        test_untouched=True,
        analytical_beats_mlp=bool(analytical_beats),
        normalizer_digest=(
            None if cfg.normalizer is None else cfg.normalizer.stats_digest
        ),
        content_digest=digest,
        mlp_skipped_reason=mlp_state.mlp_reason,
        limitations=(
            "software_contract_fixtures_only",
            "not_native_training_success",
            "low_mse_alone_does_not_certify_accelerator",
        ),
        checkpoint_dir=str(cfg.output_dir),
    )


class DynamicsBaselineTrainer:
    """Train classical then optional MLP baselines; checkpoint on validation."""

    def __init__(self, config: DynamicsBaselineTrainerConfig) -> None:
        self._cfg = config

    def run_pilot(self) -> PilotCheckpointCard:
        cfg = self._cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        train, val = _load_pilot_matrices(cfg)
        test_ids = cfg.split.ids_for("test")
        val_scores, method_order = _score_classical_methods(
            task=cfg.task,
            train=train,
            val=val,
            active_dofs=cfg.active_dofs,
        )
        mlp_state = _run_mlp_seeds(cfg, train, val, val_scores, method_order)
        card = _assemble_pilot_card(
            cfg,
            val_scores=mlp_state.val_scores,
            method_order=mlp_state.method_order,
            mlp_state=mlp_state,
            test_trial_count=len(test_ids),
        )
        (cfg.output_dir / "pilot_card.json").write_text(
            json.dumps(card.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return card
