"""Multi-seed pilot training orchestration for NM-05 baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.shared.python.neural_motion.episodes import (
    EpisodeStore,
    FamilySplitPlan,
    TrainOnlyNormalizer,
)
from src.shared.python.neural_motion.json_io import SortedJsonWritableMixin

from .classical import ClassicalMethod, fit_classical, predict_classical
from .dataset import build_trial_matrices
from .neural import train_small_mlp
from .types import BASELINE_SCHEMA, DynamicsTaskKind, InverseLabelConditioning

__all__ = ["DynamicsBaselineTrainer", "PilotCheckpointCard"]


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


class DynamicsBaselineTrainer:
    """Train classical then optional MLP baselines; checkpoint on validation."""

    def __init__(
        self,
        *,
        store: EpisodeStore,
        split: FamilySplitPlan,
        task: DynamicsTaskKind,
        seeds: Sequence[int],
        output_dir: str | Path,
        active_dofs: Sequence[int] = (0,),
        normalizer: TrainOnlyNormalizer | None = None,
        conditioning: InverseLabelConditioning | None = None,
        require_mlp: bool = False,
        overfit_smoke: bool = False,
    ) -> None:
        if not seeds:
            raise ValueError("seeds must be non-empty")
        self._store = store
        self._split = split
        self._task = task
        self._seeds = tuple(int(s) for s in seeds)
        self._output_dir = Path(output_dir)
        self._active_dofs = tuple(int(i) for i in active_dofs)
        self._normalizer = normalizer
        self._conditioning = conditioning
        self._require_mlp = bool(require_mlp)
        self._overfit_smoke = bool(overfit_smoke)

    def run_pilot(self) -> PilotCheckpointCard:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        train = build_trial_matrices(
            self._store,
            self._split,
            task=self._task,
            split="train",
            active_dofs=self._active_dofs,
            conditioning=self._conditioning,
        )
        val = build_trial_matrices(
            self._store,
            self._split,
            task=self._task,
            split="val",
            active_dofs=self._active_dofs,
            conditioning=self._conditioning,
        )
        if train.features.shape[0] == 0:
            raise ValueError("train split is empty; refuse dynamics baseline fit")
        if val.features.shape[0] == 0:
            if not self._overfit_smoke:
                raise ValueError("val split is empty; refuse model selection")
            val = train
        test_ids = self._split.splits.get("test", ())

        val_scores: dict[str, float] = {}
        method_order: list[str] = []
        for method in (
            ClassicalMethod.ANALYTICAL_PENDULUM,
            ClassicalMethod.RIDGE,
            ClassicalMethod.NEAREST_NEIGHBOR,
        ):
            model = fit_classical(
                method,
                task=self._task,
                features=train.features,
                targets=train.targets,
                active_dofs=self._active_dofs,
            )
            pred = predict_classical(model, val.features)
            mse = float(np.mean((pred - val.targets) ** 2))
            val_scores[method.value] = mse
            method_order.append(method.value)

        mlp_reason: str | None = None
        best_train = float("inf")
        best_val = float("inf")
        best_seed = self._seeds[0]
        epochs = 80 if self._overfit_smoke else 40

        for seed in self._seeds:
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
                if self._require_mlp:
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
                        self._output_dir / f"mlp_seed_{seed}.npz",
                        **result.state,
                    )

        analytical = val_scores[ClassicalMethod.ANALYTICAL_PENDULUM.value]
        mlp = val_scores.get("small_mlp")
        analytical_beats = mlp is None or analytical <= mlp

        payload = {
            "schema": BASELINE_SCHEMA,
            "task": self._task.value,
            "seeds": list(self._seeds),
            "method_order": method_order,
            "val_mse_by_method": val_scores,
            "test_trial_count_untouched": len(test_ids),
            "active_dofs": list(self._active_dofs),
            "normalizer_digest": (
                None if self._normalizer is None else self._normalizer.stats_digest
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        card = PilotCheckpointCard(
            schema=BASELINE_SCHEMA,
            task=self._task.value,
            seeds=self._seeds,
            method_order=tuple(method_order),
            val_mse_by_method={k: float(v) for k, v in val_scores.items()},
            best_seed_metrics={
                "seed": float(best_seed),
                "train_mse": float(
                    best_train if np.isfinite(best_train) else analytical
                ),
                "val_mse": float(best_val if np.isfinite(best_val) else analytical),
            },
            test_untouched=True,
            analytical_beats_mlp=bool(analytical_beats),
            normalizer_digest=(
                None if self._normalizer is None else self._normalizer.stats_digest
            ),
            content_digest=digest,
            mlp_skipped_reason=mlp_reason,
            limitations=(
                "software_contract_fixtures_only",
                "not_native_training_success",
                "low_mse_alone_does_not_certify_accelerator",
            ),
            checkpoint_dir=str(self._output_dir),
        )
        (self._output_dir / "pilot_card.json").write_text(
            json.dumps(card.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return card
