"""Evaluate a policy study and write its #14436 run record (NP-01, #14476)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.learning.policy_experiments.metrics import (
    METRIC_NAMES,
    MetricRegistry,
    SwingTrial,
)
from src.learning.policy_experiments.spec import ExperimentSpec
from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import RunProvenance, emit_run_manifest

EVALUATION_FILENAME = "evaluation.json"


@dataclass(frozen=True)
class EvaluationResult:
    """Metric values keyed by registry name, with the registry version used."""

    metrics: dict[str, float]
    registry_version: str

    def as_dict(self) -> dict[str, object]:
        """Plain JSON-ready data with metrics in a fixed (sorted) order."""
        return {
            "metrics": dict(sorted(self.metrics.items())),
            "registry_version": self.registry_version,
        }


def evaluate(
    spec: ExperimentSpec, trial: SwingTrial, registry: MetricRegistry
) -> EvaluationResult:
    """Every :data:`METRIC_NAMES` metric of ``trial``, read from ``registry``.

    ``spec`` (valid by construction) names the study the trial belongs to.
    Postcondition: exactly :data:`METRIC_NAMES`, none computed outside ``registry``.
    """
    del spec
    metrics = {name: registry.compute(name, trial) for name in METRIC_NAMES}
    return EvaluationResult(metrics=metrics, registry_version=registry.version)


def write_run_record(
    run_dir: Path,
    spec: ExperimentSpec,
    result: EvaluationResult,
    *,
    seed: int,
    ud_commit: str,
    created_at: datetime,
) -> RunManifest:
    """Write ``evaluation.json`` and its run manifest into ``run_dir``.

    ``created_at`` is required (not "now") so the same spec, seed, result and
    commit give byte-identical files. Raises :class:`ExperimentSpecError` if
    ``seed`` is not one of ``spec.seeds`` and ``RunManifestError`` for an
    invalid commit. Postcondition: the manifest's output digest covers the
    evaluation file and its parameters carry the spec and its digest.
    """
    spec.require_seed(seed)
    root = Path(run_dir)
    output = root / EVALUATION_FILENAME
    text = json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
    provenance = RunProvenance(
        model_id=spec.model_id,
        revision=spec.model_revision,
        parameters={"spec": spec.as_dict(), "spec_digest": spec.digest()},
        input_sha256=spec.demonstration_source.removeprefix("sha256:"),
        seed=seed,
        engine_name="policy-experiments",
        engine_version="np-01",
        ud_commit=ud_commit,
        input_label=f"demonstrations; registry {result.registry_version}",
    )
    return emit_run_manifest(
        root, outputs=[output], provenance=provenance, created_at=created_at
    )
