"""Unit tests for multi-swing acquisition, event anchoring, and phase registration (MOSAIC-10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.estimation.mosaic.swing_registration import (
    DEFAULT_CANONICAL_EVENT_PHASES,
    MultiSwingTrial,
    PhaseRegistration,
    SwingEvent,
    TrialProvenance,
    align_trials_to_phase,
    evaluate_registration_variance,
    ingest_swing_trial,
)

pytestmark = pytest.mark.unit


def _synthetic_swing_trial(
    trial_id: str,
    t_address: float = 0.0,
    t_takeaway: float = 0.25,
    t_top: float = 0.80,
    t_impact: float = 1.05,
    t_finish: float = 1.40,
    sample_rate_hz: float = 200.0,
) -> MultiSwingTrial:
    """Generate synthetic swing trajectory with known event timings."""
    dt = 1.0 / sample_rate_hz
    times = (np.arange(int(t_finish * sample_rate_hz) + 1) * dt).astype(np.float64)

    # Synthetic torque profiles that peak sharply at impact
    # Bell-shaped curve centered at impact with width ~0.08 s
    torque_impact = 50.0 * np.exp(-((times - t_impact) ** 2) / (2 * 0.04**2))
    # Backswing torque centered between takeaway and top
    t_mid = 0.5 * (t_takeaway + t_top)
    torque_backswing = -20.0 * np.exp(-((times - t_mid) ** 2) / (2 * 0.1**2))
    u1 = torque_impact + torque_backswing
    u2 = 0.5 * torque_impact - 0.3 * torque_backswing
    trajectories = {"inputs": np.stack([u1, u2], axis=-1)}

    events = {
        SwingEvent.ADDRESS: t_address,
        SwingEvent.TAKEAWAY: t_takeaway,
        SwingEvent.TOP: t_top,
        SwingEvent.IMPACT: t_impact,
        SwingEvent.FINISH: t_finish,
    }

    provenance = TrialProvenance(
        subject_id="sub-001",
        session_id="sess-01",
        anonymized=True,
    )

    return MultiSwingTrial(
        trial_id=trial_id,
        times=times,
        trajectories=trajectories,
        events=events,
        provenance=provenance,
    )


def test_phase_registration_monotonic_and_anchors() -> None:
    """Registration warps trial time to [0, 1] and maps event times to canonical phases."""
    trial = _synthetic_swing_trial("trial-1")
    reg = PhaseRegistration.fit(trial)

    # Monotonicity test
    phases = reg.time_to_phase(trial.times)
    assert np.all(np.diff(phases) >= 0.0)
    assert phases[0] == pytest.approx(0.0)
    assert phases[-1] == pytest.approx(1.0)

    # Anchor verification
    for event, expected_phase in DEFAULT_CANONICAL_EVENT_PHASES.items():
        t_event = trial.events[event]
        tau = reg.time_to_phase(np.array([t_event]))[0]
        assert tau == pytest.approx(expected_phase, abs=1e-6)

    # Invertibility: phase_to_time(time_to_phase(t)) == t
    t_recovered = reg.phase_to_time(phases)
    np.testing.assert_allclose(t_recovered, trial.times, atol=1e-6)


def test_synthetic_warped_trials_phase_recovery() -> None:
    """Multi-swing trials with distinct event timings align to uniform canonical phase."""
    trial1 = _synthetic_swing_trial(
        "trial-fast-backswing",
        t_address=0.0,
        t_takeaway=0.20,
        t_top=0.65,
        t_impact=0.90,
        t_finish=1.20,
    )
    trial2 = _synthetic_swing_trial(
        "trial-slow-backswing",
        t_address=0.0,
        t_takeaway=0.30,
        t_top=0.95,
        t_impact=1.20,
        t_finish=1.55,
    )

    reg1 = PhaseRegistration.fit(trial1)
    reg2 = PhaseRegistration.fit(trial2)

    # Verify event anchors match between both registered trials
    for event, expected_tau in DEFAULT_CANONICAL_EVENT_PHASES.items():
        tau1 = reg1.time_to_phase(np.array([trial1.events[event]]))[0]
        tau2 = reg2.time_to_phase(np.array([trial2.events[event]]))[0]
        assert tau1 == pytest.approx(expected_tau, abs=1e-6)
        assert tau2 == pytest.approx(expected_tau, abs=1e-6)


def test_event_anchors_propagate_into_shared_template() -> None:
    """Registered trajectories align peaks and propagate anchors into the template."""
    trials = [
        _synthetic_swing_trial(
            f"trial-{i}",
            t_takeaway=0.22 + 0.05 * i,
            t_top=0.70 + 0.10 * i,
            t_impact=0.95 + 0.12 * i,
            t_finish=1.30 + 0.15 * i,
        )
        for i in range(4)
    ]

    n_bins = 50
    aligned = align_trials_to_phase(
        trials, trajectory_key="inputs", n_phase_bins=n_bins
    )

    # Peak in aligned inputs should align around impact canonical phase (0.75)
    impact_bin = int(DEFAULT_CANONICAL_EVENT_PHASES[SwingEvent.IMPACT] * n_bins)
    for trial_aligned in aligned.per_trial:
        # Peak of first input channel
        peak_bin = int(np.argmax(trial_aligned[:, 0]))
        assert abs(peak_bin - impact_bin) <= 2


def test_explained_variance_improvement() -> None:
    """Quantify time-warp artefacts: registration substantially improves first-mode explained variance."""
    trials = [
        _synthetic_swing_trial(
            f"trial-{i}",
            t_takeaway=0.20 + 0.06 * i,
            t_top=0.65 + 0.12 * i,
            t_impact=0.88 + 0.14 * i,
            t_finish=1.20 + 0.16 * i,
        )
        for i in range(5)
    ]

    var_raw, var_registered = evaluate_registration_variance(
        trials, trajectory_key="inputs", n_phase_bins=50
    )

    # Registration must significantly increase explained variance
    assert var_registered > var_raw
    assert var_registered > 0.90


def test_ingest_path_and_anonymisation() -> None:
    """Ingest path validates contracts, parses events, and enforces anonymisation."""
    raw_data = {
        "trial_id": "swing_trial_42",
        "times": [0.0, 0.2, 0.6, 0.8, 1.0],
        "trajectories": {"markers": np.ones((5, 6, 3))},
        "events": {
            "address": 0.0,
            "takeaway": 0.2,
            "top": 0.6,
            "impact": 0.8,
            "finish": 1.0,
        },
        "subject_name": "Tiger Woods",  # Personal identifiable information
        "session_id": "sess-2026",
    }

    trial = ingest_swing_trial(raw_data, anonymize=True)
    assert trial.trial_id == "swing_trial_42"
    assert trial.provenance.anonymized
    # PII name must not be present in provenance
    assert "Tiger Woods" not in str(trial.provenance)
    assert trial.provenance.subject_id.startswith("anon_")


def test_ingest_rejects_missing_and_mislabelled_events() -> None:
    """Missing or non-chronological events produce explicit ContractViolationError."""
    events_map: dict[str, float] = {
        "address": 0.0,
        "takeaway": 0.2,
        "top": 0.6,
        "impact": 0.8,
        "finish": 1.2,
    }
    base_data: dict[str, Any] = {
        "trial_id": "trial_bad",
        "times": np.linspace(0.0, 1.2, 100),
        "trajectories": {"inputs": np.zeros((100, 2))},
        "events": events_map,
    }

    # Missing event 'impact'
    data_missing = dict(base_data)
    data_missing["events"] = {k: v for k, v in events_map.items() if k != "impact"}
    with pytest.raises(PreconditionError):
        ingest_swing_trial(data_missing)

    # Mislabelled / non-chronological event (impact before top)
    data_misordered = dict(base_data)
    bad_events = dict(events_map)
    bad_events["impact"] = 0.5  # before top at 0.6
    data_misordered["events"] = bad_events
    with pytest.raises(PreconditionError):
        ingest_swing_trial(data_misordered)

    # Non-monotonic times
    data_bad_times = dict(base_data)
    data_bad_times["times"] = np.array([0.0, 0.2, 0.1, 0.4, 0.5])
    data_bad_times["trajectories"] = {"inputs": np.zeros((5, 2))}
    with pytest.raises(PreconditionError):
        ingest_swing_trial(data_bad_times)
