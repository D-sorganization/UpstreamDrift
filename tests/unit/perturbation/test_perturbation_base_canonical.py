"""Canonical Tools-plan integration for the legacy perturbation analyzer base."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.config import PerturbationConfig
from src.shared.python.perturbation.perturbation_base import (
    CanonicalPerturbationBatch,
    PerturbationAnalyzerBase,
)
from src.shared.python.perturbation.trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    ImpactObservation,
    SampledInput,
    TrialTrace,
)

pytestmark = pytest.mark.unit

_PLAN_SHA = "a" * 64
_SCENARIO_SHA = "b" * 64
_CONFIG_SHA = "c" * 64
_TOOLS_REVISION = "d" * 40
_ENGINE_REVISION = "e" * 40


@dataclass
class _Result:
    t: np.ndarray
    q_traj: np.ndarray
    v_traj: np.ndarray
    ee_pos_traj: np.ndarray
    ee_vel_traj: np.ndarray
    kinetic_energy_traj: np.ndarray
    potential_energy_traj: np.ndarray

    @property
    def n_steps(self) -> int:
        return len(self.t)


def _result(value: float) -> _Result:
    times = np.array([0.0, 0.01, 0.02])
    q = np.array([[0.0], [value / 2.0], [value]])
    v = np.array([[0.0], [value], [value]])
    positions = np.column_stack((q[:, 0], np.zeros(3), np.zeros(3)))
    velocities = np.column_stack((v[:, 0], np.zeros(3), np.zeros(3)))
    return _Result(
        t=times,
        q_traj=q,
        v_traj=v,
        ee_pos_traj=positions,
        ee_vel_traj=velocities,
        kinetic_energy_traj=np.array([0.0, 0.25, 0.5]) * value**2,
        potential_energy_traj=np.zeros(3),
    )


class _Analyzer(PerturbationAnalyzerBase):
    ENGINE_NAME = "canonical-stub"

    def _simulate(self, coeffs: list[list[float]]) -> _Result:
        value = float(coeffs[0][0])
        if value < 0.0:
            raise RuntimeError("solver rejected sampled setting")
        return _result(value)

    def _get_q_traj(self, sim_result: _Result) -> np.ndarray:
        return sim_result.q_traj

    def _get_v_traj(self, sim_result: _Result) -> np.ndarray:
        return sim_result.v_traj

    def _validate_sim_result_type(self, sim_result: object) -> None:
        if not isinstance(sim_result, _Result):
            raise ValueError("sim_result must be _Result")


class _Gateway:
    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples

    def sample_inputs(self, _plan: object) -> np.ndarray:
        return self.samples


class _Collector:
    def collect_success(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        result: object,
    ) -> CanonicalTrialEvidence:
        assert isinstance(result, _Result)
        trace = TrialTrace(
            times_s=result.t,
            q=result.q_traj,
            v=result.v_traj,
            coordinate_ids=("club_angle",),
            coordinate_units=("rad",),
            velocity_units=("rad/s",),
            markers_m=result.ee_pos_traj[:, None, :],
            marker_ids=("clubhead",),
            frame_id="world-z-up",
            alignment_id="downswing-start/v1",
            complete=True,
        )
        inputs = (SampledInput("torque_offset", float(sampled_row[0]), "N*m"),)
        if sampled_row[0] > 0.0:
            return CanonicalTrialEvidence(
                **self._identity(trial_index, plan_seed),
                sampled_inputs=inputs,
                outcome="hit",
                trace=trace,
                impact=ImpactObservation(
                    time_s=0.02,
                    state=(SampledInput("clubhead_speed", 1.0, "m/s"),),
                ),
                shot_result=(SampledInput("carry", 2.0, "m"),),
            )
        return CanonicalTrialEvidence(
            **self._identity(trial_index, plan_seed),
            sampled_inputs=inputs,
            outcome="no_impact",
            trace=trace,
            closest_approach=ClosestApproach(
                time_s=0.02,
                distance_m=0.04,
                source_marker_id="clubhead",
                target_id="ball-center",
                contact_observed=False,
            ),
        )

    def collect_failure(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        error: Exception,
    ) -> CanonicalTrialEvidence:
        return CanonicalTrialEvidence(
            **self._identity(trial_index, plan_seed),
            sampled_inputs=(
                SampledInput("torque_offset", float(sampled_row[0]), "N*m"),
            ),
            outcome="numerical_failure",
            trace=None,
            failure_reason=f"{type(error).__name__}: {error}",
        )

    @staticmethod
    def _identity(trial_index: int, seed: int) -> dict[str, object]:
        return {
            "trial_index": trial_index,
            "seed": seed,
            "plan_sha256": _PLAN_SHA,
            "scenario_sha256": _SCENARIO_SHA,
            "execution_config_sha256": _CONFIG_SHA,
            "tools_revision": _TOOLS_REVISION,
            "engine_id": "canonical-stub",
            "engine_revision": _ENGINE_REVISION,
            "model_id": "stub-one-dof/v1",
        }


def test_canonical_batch_type_is_public() -> None:
    assert perturbation.CanonicalPerturbationBatch is CanonicalPerturbationBatch


def test_canonical_batch_retains_engine_results_evidence_and_legacy_projection() -> (
    None
):
    analyzer = _Analyzer()
    plan = SimpleNamespace(n_runs=3, seed=17)
    compatibility_config = PerturbationConfig(
        n_trials=3,
        seed=17,
        noise_amplitude=0.0,
    )

    batch = analyzer.run_canonical_batch(
        plan=plan,
        gateway=_Gateway(np.array([[1.0], [0.0], [-1.0]])),
        collector=_Collector(),
        row_to_coeffs=lambda row: [[float(row[0])]],
        compatibility_config=compatibility_config,
    )

    assert isinstance(batch, CanonicalPerturbationBatch)
    assert [record.outcome for record in batch.records] == [
        "hit",
        "no_impact",
        "numerical_failure",
    ]
    assert batch.records[0].impact is not None
    assert batch.records[0].shot_result == (SampledInput("carry", 2.0, "m"),)
    assert batch.records[1].closest_approach is not None
    assert isinstance(batch.engine_results[0], _Result)
    assert isinstance(batch.engine_results[1], _Result)
    assert batch.engine_results[2] is None
    assert batch.errors[:2] == (None, None)
    assert isinstance(batch.errors[2], RuntimeError)
    assert batch.legacy_summary.config is compatibility_config
    assert batch.legacy_summary.engine_name == "canonical-stub"
    assert batch.legacy_summary.success_rate == pytest.approx(2.0 / 3.0)
    assert batch.legacy_summary.metrics["end_effector_speed_final"].mean == 0.5
    assert batch.legacy_summary.to_dict()["failures"] == [
        {
            "trial_index": 2,
            "seed": 17,
            "stage": "canonical_execution",
            "error_type": "RuntimeError",
            "message": "RuntimeError: solver rejected sampled setting",
        }
    ]


@pytest.mark.parametrize(
    ("plan", "config", "message"),
    [
        (
            SimpleNamespace(n_runs=2, seed=17),
            PerturbationConfig(n_trials=1, seed=17),
            "n_trials",
        ),
        (
            SimpleNamespace(n_runs=1, seed=17),
            PerturbationConfig(n_trials=1, seed=18),
            "seed",
        ),
    ],
)
def test_canonical_batch_rejects_legacy_projection_identity_drift(
    plan: object,
    config: PerturbationConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _Analyzer().run_canonical_batch(
            plan=plan,
            gateway=_Gateway(np.array([[1.0]])),
            collector=_Collector(),
            row_to_coeffs=lambda row: [[float(row[0])]],
            compatibility_config=config,
        )


def test_legacy_run_batch_remains_available_without_canonical_dependencies() -> None:
    analyzer = _Analyzer()
    analyzer.set_base_torque_profile({"coeffs": [[1.0]]})

    summary = analyzer.run_batch(PerturbationConfig(n_trials=2, seed=9))

    assert summary.engine_name == "canonical-stub"
    assert summary.success_rate == 1.0
