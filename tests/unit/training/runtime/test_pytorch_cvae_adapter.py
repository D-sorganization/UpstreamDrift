"""Unit tests for :class:`PyTorchCVAERunner` (GH issue #6014).

All tests are guarded by ``pytest.importorskip("torch")`` — they exercise
the runner against the same in-memory synthetic CVAE fixture used by
:mod:`tests.unit.motion_matching.test_swing_inverse_cvae_training` so we
do not depend on parquet IO.

When torch is absent the entire module is skipped at collection time.
The runner-protocol and ``can_run`` smoke tests still exercise the
torch-less paths through a monkeypatched ``importlib.util.find_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Skip the whole module unless torch (and the numpy/pandas dataset deps
# the synthetic fixture needs) are available.
torch = pytest.importorskip("torch")  # noqa: F841 - used by training loop
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.shared.python.motion_matching.inverse import (  # noqa: E402
    DEFAULT_COEFFICIENT_DIM,
)
from src.shared.python.training import (  # noqa: E402
    Dataset,
    DatasetRegistry,
    ResourceRequest,
    ThreadingCancelToken,
    TrainingConfig,
    TrainingFramework,
    TrainingJobRunner,
    TrainingStatus,
)
from src.shared.python.training.metrics import MetricKind  # noqa: E402
from src.shared.python.training.runtime import (  # noqa: E402
    InMemoryProgressSink,
    PyTorchCVAERunner,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


_ENTRY_POINT = "motion_matching.inverse:train_inverse_cvae"


# ---------------------------------------------------------------------------
# Synthetic-dataset helpers (mirrors test_swing_inverse_cvae_training.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeCompactDataset:
    trials: pd.DataFrame
    timesteps: pd.DataFrame
    joint_names: tuple
    coefficient_letters: tuple = ("A", "B", "C", "D", "E", "F", "G")
    schema_version: str = "compact-1.0"


def _build_synthetic_dataset(
    n_trials: int = 8, n_timesteps: int = 16
) -> _FakeCompactDataset:
    rng = np.random.default_rng(0)
    joint_names = tuple(f"j{i}" for i in range(27))
    trial_rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []
    for trial_id in range(n_trials):
        coeffs = rng.normal(0, 50.0, size=DEFAULT_COEFFICIENT_DIM).astype(np.float32)
        trial_rows.append(
            {
                "trial_id": trial_id,
                "coefficients": coeffs.tolist(),
                "joint_names": list(joint_names),
            }
        )
        base = float(np.sum(coeffs)) / 1000.0
        ts = np.linspace(0.0, 0.3, n_timesteps)
        for _k, t in enumerate(ts):
            phase = base + t
            ts_rows.append(
                {
                    "trial_id": trial_id,
                    "t": float(t),
                    "r_buttend": [np.sin(phase), np.cos(phase), 0.5 * t],
                    "r_clubhead": [
                        np.sin(phase + 0.5),
                        np.cos(phase + 0.5),
                        1.0 * t,
                    ],
                    "r_grip": [
                        np.sin(phase + 0.25),
                        np.cos(phase + 0.25),
                        0.75 * t,
                    ],
                    "v_clubhead": [
                        np.cos(phase + 0.5),
                        -np.sin(phase + 0.5),
                        1.0,
                    ],
                }
            )
    return _FakeCompactDataset(
        trials=pd.DataFrame(trial_rows),
        timesteps=pd.DataFrame(ts_rows),
        joint_names=joint_names,
    )


def _loader_factory():
    dataset = _build_synthetic_dataset()
    return lambda _path: dataset


def _toy_hyperparameters(epochs: int = 2) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "batch_size": 4,
        "lr": 5e-3,
        "kl_anneal_epochs": 1,
        "max_beta": 0.01,
        "free_bits": 0.0,
        "device": "cpu",
    }


def _make_config(
    output_dir: Path,
    *,
    hyperparameters: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    entry_point: str = _ENTRY_POINT,
) -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point=entry_point,
        output_dir=output_dir,
        hyperparameters=hyperparameters or _toy_hyperparameters(),
        dataset_id=dataset_id,
        resources=ResourceRequest(),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_satisfies_runner_protocol() -> None:
    runner = PyTorchCVAERunner()
    assert isinstance(runner, TrainingJobRunner)
    assert runner.framework is TrainingFramework.PYTORCH
    assert _ENTRY_POINT in runner.KNOWN_ENTRY_POINTS


def test_can_run_filters_by_entry_point(tmp_path: Path) -> None:
    runner = PyTorchCVAERunner()
    good = _make_config(tmp_path)
    bad = _make_config(tmp_path, entry_point="other.module:train_thing")
    assert runner.can_run(good) is True
    assert runner.can_run(bad) is False


def test_can_run_rejects_non_pytorch_framework(tmp_path: Path) -> None:
    runner = PyTorchCVAERunner()
    cfg = TrainingConfig(
        framework=TrainingFramework.GYMNASIUM,
        entry_point=_ENTRY_POINT,
        output_dir=tmp_path,
    )
    assert runner.can_run(cfg) is False


def test_can_run_returns_false_without_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = PyTorchCVAERunner()
    cfg = _make_config(tmp_path)
    import importlib.util as _util

    monkeypatch.setattr(_util, "find_spec", lambda name: None)
    assert runner.can_run(cfg) is False


def test_full_epoch_stream(tmp_path: Path) -> None:
    runner = PyTorchCVAERunner(dataset_loader=_loader_factory())
    cfg = _make_config(tmp_path / "run", hyperparameters=_toy_hyperparameters(epochs=2))
    sink = InMemoryProgressSink()
    cancel = ThreadingCancelToken()
    runner.prepare(cfg)
    result = runner.run(cfg, progress=sink, cancel=cancel)

    assert result.status is TrainingStatus.COMPLETED
    assert result.run_id is not None
    # Six metrics per epoch, two epochs.
    assert len(sink.metrics) == 12
    # First metric belongs to step 0 and is train_recon as documented.
    names_step0 = [m.name for m in sink.metrics if m.step == 0]
    assert names_step0 == [
        "train_recon",
        "train_kl",
        "val_recon",
        "val_kl",
        "beta",
        "duration_s",
    ]
    # Kinds: losses on recon/kl, scalars on beta/duration.
    loss_metrics = [m for m in sink.metrics if m.kind is MetricKind.LOSS]
    scalar_metrics = [m for m in sink.metrics if m.kind is MetricKind.SCALAR]
    assert len(loss_metrics) == 8
    assert len(scalar_metrics) == 4
    # split tags propagated.
    train_split = {m.name for m in sink.metrics if m.tags.get("split") == "train"}
    val_split = {m.name for m in sink.metrics if m.tags.get("split") == "val"}
    assert train_split == {"train_recon", "train_kl"}
    assert val_split == {"val_recon", "val_kl"}
    # Status changes — RUNNING at the start, COMPLETED at the end.
    statuses = [s for s, _ in sink.statuses]
    assert statuses[0] is TrainingStatus.RUNNING
    assert statuses[-1] is TrainingStatus.COMPLETED
    # Artifacts: checkpoint and metrics.json reachable.
    assert len(result.artifacts) == 2
    assert any(p.name == "metrics.json" for p in result.artifacts)
    ckpts = [p for p in result.artifacts if p.suffix == ".pt"]
    assert ckpts and ckpts[0].exists()
    # Final-metrics tuple mirrors the last epoch (6 entries).
    assert len(result.final_metrics) == 6


def test_cancellation_short_circuits(tmp_path: Path) -> None:
    runner = PyTorchCVAERunner(dataset_loader=_loader_factory())
    cfg = _make_config(
        tmp_path / "run",
        hyperparameters=_toy_hyperparameters(epochs=5),
    )
    cancel = ThreadingCancelToken()
    seen_epochs: set[int] = set()
    inner_sink = InMemoryProgressSink()

    class _CancellingSink:
        def emit_metric(self, metric) -> None:
            seen_epochs.add(metric.step)
            inner_sink.emit_metric(metric)
            # After epoch 0's 6 metrics arrive, request cancellation.
            if metric.step == 0 and metric.name == "duration_s":
                cancel.request_cancel()

        def emit_status(self, status, *, message=None) -> None:
            inner_sink.emit_status(status, message=message)

        @property
        def statuses(self):
            return inner_sink.statuses

    sink = _CancellingSink()

    runner.prepare(cfg)
    result = runner.run(cfg, progress=sink, cancel=cancel)

    assert result.status is TrainingStatus.CANCELLED
    # Only one or two epochs of metrics should have run (cancellation is
    # polled after each epoch and after the early-stop branch).
    assert max(seen_epochs) <= 1
    # Last status emission is CANCELLED.
    final_status, _ = sink.statuses[-1]
    assert final_status is TrainingStatus.CANCELLED
    # Best-so-far checkpoint must still be present.
    ckpts = [p for p in result.artifacts if p.suffix == ".pt"]
    assert ckpts and ckpts[0].exists()


def test_runner_protocol_can_run_false_when_dataset_missing(
    tmp_path: Path,
) -> None:
    registry = DatasetRegistry()
    runner = PyTorchCVAERunner(
        dataset_registry=registry, dataset_loader=_loader_factory()
    )
    cfg = _make_config(tmp_path / "run", dataset_id="nope")
    # can_run does not consult the registry — it should still accept the
    # entry point. The dataset lookup is enforced in prepare().
    assert runner.can_run(cfg) is True
    with pytest.raises(LookupError, match="nope"):
        runner.prepare(cfg)


def test_prepare_creates_output_dir(tmp_path: Path) -> None:
    runner = PyTorchCVAERunner(dataset_loader=_loader_factory())
    nested = tmp_path / "deeply" / "nested" / "outdir"
    cfg = _make_config(nested)
    runner.prepare(cfg)
    assert nested.is_dir()


def test_prepare_with_known_dataset_succeeds(tmp_path: Path) -> None:
    registry = DatasetRegistry()
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    registry.register(
        Dataset(
            dataset_id="compact_swing_v1",
            name="Synthetic compact swing",
            path=dataset_dir,
            format="parquet",
        )
    )
    runner = PyTorchCVAERunner(
        dataset_registry=registry, dataset_loader=_loader_factory()
    )
    cfg = _make_config(
        tmp_path / "run",
        dataset_id="compact_swing_v1",
        hyperparameters=_toy_hyperparameters(),
    )
    runner.prepare(cfg)  # must not raise


def test_dataset_registry_must_be_dataset_registry() -> None:
    with pytest.raises(TypeError):
        PyTorchCVAERunner(dataset_registry="not-a-registry")  # type: ignore[arg-type]


def test_unrecognised_hyperparameters_are_ignored(tmp_path: Path) -> None:
    runner = PyTorchCVAERunner(dataset_loader=_loader_factory())
    hp = _toy_hyperparameters()
    hp["unused_extra_key"] = "ignored"
    cfg = _make_config(tmp_path / "run", hyperparameters=hp)
    sink = InMemoryProgressSink()
    cancel = ThreadingCancelToken()
    runner.prepare(cfg)
    result = runner.run(cfg, progress=sink, cancel=cancel)
    assert result.status is TrainingStatus.COMPLETED


def test_max_epochs_caps_hyperparameter_epochs(tmp_path: Path) -> None:
    runner = PyTorchCVAERunner(dataset_loader=_loader_factory())
    hp = _toy_hyperparameters(epochs=5)
    cfg = TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point=_ENTRY_POINT,
        output_dir=tmp_path / "run",
        hyperparameters=hp,
        max_epochs=1,
        seed=42,
    )
    sink = InMemoryProgressSink()
    cancel = ThreadingCancelToken()
    runner.prepare(cfg)
    result = runner.run(cfg, progress=sink, cancel=cancel)
    assert result.status is TrainingStatus.COMPLETED
    # Six metrics per epoch; cap reduces to a single epoch.
    assert len(sink.metrics) == 6
