"""Tests for plan execution bound directly to durable trial evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.canonical_trial_campaign import (
    CanonicalVariationCampaignResult,
    execute_batched_variation_campaign,
    execute_serial_variation_campaign,
)
from src.shared.python.perturbation.double_pendulum_trial_adapter import (
    DoublePendulumTrialAdapter,
    DoublePendulumTrialConfig,
)
from src.shared.python.perturbation.trial_evidence_bundle import (
    load_trial_evidence_bundle,
)
from src.shared.python.simulation_backends import GolfModelParams, SimState

pytestmark = pytest.mark.unit

_DAMPING = "swing_sim.swing.damping_shoulder"


@dataclass(frozen=True)
class _Spec:
    variable_key: str = _DAMPING
    time_window_s: tuple[float, float] | None = None
    point_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Plan:
    noise: tuple[_Spec, ...] = (_Spec(),)
    n_runs: int = 2
    seed: int = 41
    mode: str = "swing"


class _Gateway:
    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples

    def sample_inputs(self, _plan: object) -> np.ndarray:
        return self._samples


def _adapter(plan: _Plan) -> DoublePendulumTrialAdapter:
    config = DoublePendulumTrialConfig(
        model_params=GolfModelParams.default(),
        initial_state=SimState(
            q=np.radians(np.array([-45.0, -90.0])),
            v=np.array([2.0, 1.0]),
        ),
        duration_s=0.02,
        dt_s=0.01,
        base_torques_nm=(4.0, 0.5),
        target_position_m=(20.0, 0.0, 20.0),
        contact_radius_m=0.01,
        frame_id="pendulum-plane:x-forward-y-out-z-up",
        alignment_id="downswing-start/v1",
    )
    return DoublePendulumTrialAdapter(
        plan=plan,
        config=config,
        plan_sha256="a" * 64,
        scenario_sha256="d" * 64,
        tools_revision="17474249b9267d0e73a779c1d72f231e7b8de39c",
        engine_revision="c" * 40,
    )


def test_campaign_api_is_public() -> None:
    assert (
        perturbation.execute_serial_variation_campaign
        is execute_serial_variation_campaign
    )
    assert (
        perturbation.execute_batched_variation_campaign
        is execute_batched_variation_campaign
    )
    assert perturbation.CanonicalVariationCampaignResult is (
        CanonicalVariationCampaignResult
    )


def test_serial_campaign_retains_miss_and_failure_in_one_qualified_bundle(
    tmp_path: Path,
) -> None:
    plan = _Plan()
    adapter = _adapter(plan)
    destination = tmp_path / "serial-campaign"

    result = execute_serial_variation_campaign(
        plan=plan,
        gateway=_Gateway(np.array([[0.4], [-0.1]])),
        runner=adapter.run,
        collector=adapter,
        destination=destination,
    )

    assert result.bundle.path == destination.resolve()
    assert result.bundle.trial_count == 2
    assert tuple(record.outcome for record in result.records) == (
        "no_impact",
        "numerical_failure",
    )
    loaded = load_trial_evidence_bundle(destination)
    assert tuple(record.outcome for record in loaded) == (
        "no_impact",
        "numerical_failure",
    )


def test_batched_campaign_preserves_canonical_row_order(tmp_path: Path) -> None:
    plan = _Plan()
    adapter = _adapter(plan)
    destination = tmp_path / "batched-campaign"

    result = execute_batched_variation_campaign(
        plan=plan,
        gateway=_Gateway(np.array([[0.4], [1.2]])),
        batch_runner=lambda rows: tuple(adapter.run(row) for row in rows),
        collector=adapter,
        destination=destination,
    )

    assert tuple(record.trial_index for record in result.records) == (0, 1)
    assert tuple(
        record.sampled_inputs[0].value for record in result.records
    ) == pytest.approx((0.4, 1.2))
    loaded = load_trial_evidence_bundle(destination)
    assert tuple(record.trial_index for record in loaded) == (0, 1)
    assert tuple(record.sampled_inputs for record in loaded) == tuple(
        record.sampled_inputs for record in result.records
    )
    assert loaded[0].trace is not None and result.records[0].trace is not None
    np.testing.assert_array_equal(loaded[0].trace.q, result.records[0].trace.q)


def test_existing_destination_blocks_execution_before_model_run(
    tmp_path: Path,
) -> None:
    plan = _Plan()
    adapter = _adapter(plan)
    destination = tmp_path / "existing"
    destination.mkdir()
    calls = 0

    def runner(row: np.ndarray) -> object:
        nonlocal calls
        calls += 1
        return adapter.run(row)

    with pytest.raises(FileExistsError, match="already exists"):
        execute_serial_variation_campaign(
            plan=plan,
            gateway=_Gateway(np.array([[0.4], [1.2]])),
            runner=runner,
            collector=adapter,
            destination=destination,
        )

    assert calls == 0
