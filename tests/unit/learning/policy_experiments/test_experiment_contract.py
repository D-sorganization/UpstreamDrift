"""NP-01 (#14476): experiment contract for neural swing policy studies.

The spec is validated at construction (DbC), the evaluator reads every metric
from a registry (the #14421 stub adapter until GM-1a lands) and the run record
is a #14436 run manifest whose bytes depend only on the spec, seed and inputs.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest

from src.learning.policy_experiments.evaluator import evaluate, write_run_record
from src.learning.policy_experiments.metrics import StubMetricRegistry, SwingTrial
from src.learning.policy_experiments.spec import (
    ComputeBudget,
    EditRecipe,
    ExperimentSpec,
    ExperimentSpecError,
    FineTuneConfig,
    PolicyArch,
)

pytestmark = pytest.mark.unit

REVISION = "sha256:" + "a" * 64
COMMIT = "b" * 40
WHEN = datetime(2026, 11, 6, 6, 0, tzinfo=timezone.utc)


def _spec(**overrides) -> ExperimentSpec:
    fields = {
        "model_id": "mujoco/golfer-v1",
        "model_revision": REVISION,
        "task": "driver-swing",
        "demonstration_source": "sha256:" + "c" * 64,
        "policy": PolicyArch(hidden_sizes=(64, 64), activation="tanh"),
        "seeds": (0, 1, 2),
        "intervention": EditRecipe(layer="hidden.1", rank=1, scale=-1.0),
        "budget": ComputeBudget(max_forward_passes=50, max_wall_s=600.0),
    }
    fields.update(overrides)
    return ExperimentSpec(**fields)


def _trial(offset: float = 0.0) -> SwingTrial:
    t = np.linspace(0.0, 1.0, 11)
    reference = np.column_stack([np.sin(t), np.cos(t)])
    return SwingTrial(
        time_s=t,
        reference_q=reference,
        policy_q=reference + offset,
        reference_clubhead_speed=45.0,
        policy_clubhead_speed=43.5,
        constraint_residual=np.full(11, 1e-3),
    )


def test_spec_accepts_edit_or_fine_tune_or_baseline() -> None:
    assert _spec().intervention.kind == "edit"
    tuned = _spec(intervention=FineTuneConfig(steps=100, lr=1e-3, lora_rank=4))
    assert tuned.intervention.kind == "fine_tune"
    assert _spec(intervention=None).intervention is None


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"seeds": ()}, "seeds"),
        ({"seeds": (1, 1)}, "seeds"),
        ({"seeds": (-1,)}, "seeds"),
        ({"model_revision": ""}, "model_revision"),
        ({"model_revision": "main"}, "model_revision"),
        ({"model_id": " "}, "model_id"),
        ({"task": ""}, "task"),
        ({"demonstration_source": "demos.npz"}, "demonstration_source"),
    ],
)
def test_spec_rejects_missing_or_mutable_provenance(overrides, fragment) -> None:
    with pytest.raises(ExperimentSpecError, match=fragment):
        _spec(**overrides)


@pytest.mark.parametrize(
    "build",
    [
        lambda: PolicyArch(hidden_sizes=(), activation="tanh"),
        lambda: PolicyArch(hidden_sizes=(64, 0), activation="tanh"),
        lambda: PolicyArch(hidden_sizes=(64,), activation="swish?"),
        lambda: EditRecipe(layer="", rank=1, scale=1.0),
        lambda: EditRecipe(layer="hidden.0", rank=0, scale=1.0),
        lambda: FineTuneConfig(steps=0, lr=1e-3),
        lambda: FineTuneConfig(steps=10, lr=0.0),
        lambda: ComputeBudget(max_forward_passes=0, max_wall_s=1.0),
        lambda: ComputeBudget(max_forward_passes=1, max_wall_s=float("nan")),
    ],
)
def test_parts_validate_their_own_preconditions(build) -> None:
    with pytest.raises(ExperimentSpecError):
        build()


def test_evaluator_returns_known_metrics_on_synthetic_trajectories() -> None:
    result = evaluate(_spec(), _trial(offset=0.02), StubMetricRegistry())
    assert result.metrics["tracking_rmse"] == pytest.approx(0.02, abs=1e-15)
    assert result.metrics["clubhead_speed_error"] == pytest.approx(-1.5)
    assert result.metrics["constraint_violation_max"] == pytest.approx(1e-3)
    assert result.registry_version == StubMetricRegistry.version


def test_evaluator_defines_no_metric_locally() -> None:
    class Doubled(StubMetricRegistry):
        version = "test-doubled"

        def compute(self, name: str, trial: SwingTrial) -> float:
            return 2.0 * super().compute(name, trial)

    plain = evaluate(_spec(), _trial(offset=0.02), StubMetricRegistry())
    doubled = evaluate(_spec(), _trial(offset=0.02), Doubled())
    for name, value in plain.metrics.items():
        assert doubled.metrics[name] == pytest.approx(2.0 * value)
    assert doubled.registry_version == "test-doubled"


def test_registry_rejects_unknown_metric_and_bad_trials() -> None:
    with pytest.raises(KeyError, match="no_such_metric"):
        StubMetricRegistry().compute("no_such_metric", _trial())
    good = _trial()
    with pytest.raises(ExperimentSpecError, match="shape"):
        dataclasses.replace(good, policy_q=good.policy_q[:-1])
    with pytest.raises(ExperimentSpecError, match="finite"):
        dataclasses.replace(good, policy_clubhead_speed=float("inf"))


def _record(run_dir, seed=1, offset=0.02):
    run_dir.mkdir()
    result = evaluate(_spec(), _trial(offset=offset), StubMetricRegistry())
    manifest = write_run_record(
        run_dir, _spec(), result, seed=seed, ud_commit=COMMIT, created_at=WHEN
    )
    files = ("run_manifest.json", "evaluation.json")
    return manifest, tuple((run_dir / f).read_bytes() for f in files)


def test_run_record_is_byte_identical_for_same_spec_and_seed(tmp_path) -> None:
    (manifest, first), (_, second) = _record(tmp_path / "a"), _record(tmp_path / "b")
    assert manifest.seed == 1
    assert manifest.model.revision == REVISION
    assert first == second


def test_run_record_changes_with_seed_and_rejects_foreign_seed(tmp_path) -> None:
    zero, two = _record(tmp_path / "0", seed=0), _record(tmp_path / "2", seed=2)
    assert zero[1] != two[1]
    with pytest.raises(ExperimentSpecError, match="seed 7"):
        _record(tmp_path / "7", seed=7)


def test_spec_is_frozen_and_its_digest_stable_and_sensitive() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _spec().task = "iron"  # type: ignore[misc]
    assert _spec().digest() == _spec().digest()
    assert _spec().digest() != _spec(task="iron-swing").digest()
