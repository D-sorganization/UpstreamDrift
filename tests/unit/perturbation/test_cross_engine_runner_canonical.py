"""Canonical evidence retention and qualification in the cross-engine runner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
)
from src.shared.python.perturbation.cross_engine_runner import (
    CanonicalCrossEngineReport,
    CrossEnginePerturbationRunner,
)
from src.shared.python.perturbation.cross_engine_trial_parity import (
    CrossEngineCompatibilityError,
    CrossEngineTolerances,
)
from src.shared.python.perturbation.perturbation_base import (
    CanonicalPerturbationBatch,
)
from src.shared.python.perturbation.trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    SampledInput,
    TrialTrace,
)

pytestmark = pytest.mark.unit


def _trace(*, frame_id: str = "world-z-up", marker_offset: float = 0.0) -> TrialTrace:
    return TrialTrace(
        times_s=np.array([0.0, 0.01]),
        q=np.array([[0.0], [0.1]]),
        v=np.array([[0.0], [1.0]]),
        coordinate_ids=("club_angle",),
        coordinate_units=("rad",),
        velocity_units=("rad/s",),
        markers_m=np.array([[[0.0, 0.0, 0.0]], [[0.1 + marker_offset, 0.0, 0.0]]]),
        marker_ids=("clubhead",),
        frame_id=frame_id,
        alignment_id="downswing-start/v1",
        complete=True,
    )


def _record(
    engine: str,
    *,
    frame_id: str = "world-z-up",
    marker_offset: float = 0.0,
    outcome: str = "no_impact",
) -> CanonicalTrialEvidence:
    return CanonicalTrialEvidence(
        trial_index=0,
        seed=19,
        plan_sha256="a" * 64,
        scenario_sha256="b" * 64,
        execution_config_sha256=("c" if engine == "mujoco" else "d") * 64,
        tools_revision="e" * 40,
        engine_id=engine,
        engine_revision=("f" if engine == "mujoco" else "0") * 40,
        model_id="logical-double-pendulum/v1",
        sampled_inputs=(SampledInput("torque_offset", 1.0, "N*m"),),
        outcome=outcome,  # type: ignore[arg-type]
        trace=(
            _trace(frame_id=frame_id, marker_offset=marker_offset)
            if outcome == "no_impact"
            else None
        ),
        closest_approach=(
            ClosestApproach(0.01, 0.02, "clubhead", "ball-center", False)
            if outcome == "no_impact"
            else None
        ),
        failure_reason=("solver failed" if outcome == "numerical_failure" else None),
    )


def _summary(engine: str) -> PerturbationSummary:
    return PerturbationSummary(
        engine_name=engine,
        config=PerturbationConfig(n_trials=1, seed=19, noise_amplitude=0.0),
        robustness_score=0.8,
        metrics={},
        success_rate=1.0,
        execution_time_sec=0.01,
    )


def _batch(record: CanonicalTrialEvidence) -> CanonicalPerturbationBatch:
    error = (
        RuntimeError("solver failed") if record.outcome == "numerical_failure" else None
    )
    return CanonicalPerturbationBatch(
        records=(record,),
        engine_results=(object() if error is None else None,),
        errors=(error,),
        legacy_summary=_summary(record.engine_id),
    )


class _CanonicalAnalyzer:
    def __init__(self, batch: CanonicalPerturbationBatch) -> None:
        self.batch = batch
        self.calls = 0

    def run_canonical_batch(self, **kwargs: object) -> CanonicalPerturbationBatch:
        assert kwargs["plan"] is not None
        assert kwargs["gateway"] is not None
        assert kwargs["collector"] is not None
        assert kwargs["row_to_coeffs"] is not None
        self.calls += 1
        return self.batch


class _Gateway:
    def sample_inputs(self, _plan: object) -> np.ndarray:
        return np.array([[1.0]])


class _Collector:
    def collect_success(self, *args: object) -> CanonicalTrialEvidence:
        raise AssertionError("fake analyzer must not call collector")

    def collect_failure(self, *args: object) -> CanonicalTrialEvidence:
        raise AssertionError("fake analyzer must not call collector")


def _runner(
    mujoco: CanonicalPerturbationBatch,
    drake: CanonicalPerturbationBatch,
) -> CrossEnginePerturbationRunner:
    runner = CrossEnginePerturbationRunner(engines=["mujoco", "drake"])
    runner._analyzers["mujoco"] = _CanonicalAnalyzer(mujoco)
    runner._analyzers["drake"] = _CanonicalAnalyzer(drake)
    return runner


def _run(runner: CrossEnginePerturbationRunner) -> CanonicalCrossEngineReport:
    return runner.run_canonical_all(
        plan=SimpleNamespace(n_runs=1, seed=19),
        gateway=_Gateway(),
        collectors={"mujoco": _Collector(), "drake": _Collector()},
        row_to_coeffs={
            "mujoco": lambda row: [[float(row[0])]],
            "drake": lambda row: [[float(row[0])]],
        },
        compatibility_config=PerturbationConfig(
            n_trials=1,
            seed=19,
            noise_amplitude=0.0,
        ),
        tolerances=CrossEngineTolerances(1e-12, (1e-6,), (1e-6,), 1e-5),
        reference_engine="mujoco",
    )


def test_canonical_cross_engine_report_retains_batches_and_qualified_parity() -> None:
    report = _run(_runner(_batch(_record("mujoco")), _batch(_record("drake"))))

    assert report.comparison_qualified is True
    assert report.reference_engine == "mujoco"
    assert set(report.batches) == {"mujoco", "drake"}
    assert report.batches["drake"].records[0].trace is not None
    assert len(report.parity_metrics["drake"]) == 1
    assert report.non_comparable_trials == {"drake": ()}
    assert [entry.engine_name for entry in report.legacy_report.ranking] == [
        "mujoco",
        "drake",
    ]


def test_canonical_cross_engine_report_is_public() -> None:
    assert perturbation.CanonicalCrossEngineReport is CanonicalCrossEngineReport


def test_canonical_cross_engine_runner_rejects_frame_mismatch() -> None:
    runner = _runner(
        _batch(_record("mujoco")),
        _batch(_record("drake", frame_id="club-local")),
    )

    with pytest.raises(CrossEngineCompatibilityError, match="frame"):
        _run(runner)


def test_canonical_cross_engine_runner_rejects_engine_identity_mismatch() -> None:
    runner = _runner(
        _batch(_record("mujoco")),
        _batch(_record("pinocchio")),
    )

    with pytest.raises(CrossEngineCompatibilityError, match="identity"):
        _run(runner)


def test_non_equivalent_geometry_retains_artifacts_but_suppresses_ranking() -> None:
    report = _run(
        _runner(
            _batch(_record("mujoco")),
            _batch(_record("drake", marker_offset=2e-5)),
        )
    )

    assert report.comparison_qualified is False
    assert report.parity_metrics["drake"][0].tolerance_equivalent is False
    assert report.legacy_report.ranking == []
    assert report.legacy_report.consistency == {}


def test_numerical_failure_is_retained_as_non_comparable_and_blocks_ranking() -> None:
    report = _run(
        _runner(
            _batch(_record("mujoco", outcome="numerical_failure")),
            _batch(_record("drake", outcome="numerical_failure")),
        )
    )

    assert report.comparison_qualified is False
    assert report.non_comparable_trials == {"drake": (0,)}
    assert report.batches["mujoco"].records[0].failure_reason == "solver failed"
    assert report.legacy_report.ranking == []


@pytest.mark.parametrize("missing", ["collectors", "row_to_coeffs"])
def test_missing_engine_dependency_fails_before_any_execution(missing: str) -> None:
    runner = _runner(_batch(_record("mujoco")), _batch(_record("drake")))
    collectors = {"mujoco": _Collector(), "drake": _Collector()}
    mappings = {
        "mujoco": lambda row: [[float(row[0])]],
        "drake": lambda row: [[float(row[0])]],
    }
    kwargs = {
        "plan": SimpleNamespace(n_runs=1, seed=19),
        "gateway": _Gateway(),
        "collectors": collectors,
        "row_to_coeffs": mappings,
        "compatibility_config": PerturbationConfig(n_trials=1, seed=19),
        "tolerances": CrossEngineTolerances(1e-12, (1e-6,), (1e-6,), 1e-5),
    }
    kwargs[missing] = {"mujoco": object()}

    with pytest.raises(ValueError, match="drake"):
        runner.run_canonical_all(**kwargs)

    assert runner._analyzers["mujoco"].calls == 0
    assert runner._analyzers["drake"].calls == 0


@pytest.mark.parametrize("invalid", ["gateway", "collector", "row_to_coeffs"])
def test_invalid_dependency_fails_before_any_execution(invalid: str) -> None:
    runner = _runner(_batch(_record("mujoco")), _batch(_record("drake")))
    kwargs = {
        "plan": SimpleNamespace(n_runs=1, seed=19),
        "gateway": _Gateway(),
        "collectors": {"mujoco": _Collector(), "drake": _Collector()},
        "row_to_coeffs": {
            "mujoco": lambda row: [[float(row[0])]],
            "drake": lambda row: [[float(row[0])]],
        },
        "compatibility_config": PerturbationConfig(n_trials=1, seed=19),
        "tolerances": CrossEngineTolerances(1e-12, (1e-6,), (1e-6,), 1e-5),
    }
    if invalid == "gateway":
        kwargs["gateway"] = object()
    elif invalid == "collector":
        kwargs["collectors"] = {"mujoco": _Collector(), "drake": object()}
    else:
        kwargs["row_to_coeffs"] = {
            "mujoco": lambda row: [[float(row[0])]],
            "drake": object(),
        }

    with pytest.raises(TypeError, match=invalid):
        runner.run_canonical_all(**kwargs)

    assert runner._analyzers["mujoco"].calls == 0
    assert runner._analyzers["drake"].calls == 0
