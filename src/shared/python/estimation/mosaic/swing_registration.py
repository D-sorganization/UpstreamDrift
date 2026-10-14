"""Multi-swing acquisition and registration (MOSAIC-10).

This module implements:
1. Swing events (address, takeaway, top, impact, finish) and strict chronological contracts.
2. `MultiSwingTrial` container with trajectories, event anchors, and provenance.
3. `PhaseRegistration`: piecewise monotonic phase warping mapping each trial's time
   grid to canonical phase tau in [0, 1] anchored at swing events.
4. Trajectory alignment and quantification of time-warp artefacts on shared templates
   via SVD explained variance.
5. Ingest validation rejecting missing or mislabelled events and enforcing anonymisation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.torque_template import principal_modes

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


class SwingEvent(str, Enum):
    """Canonical key events of a golf swing in chronological sequence."""

    ADDRESS = "address"
    TAKEAWAY = "takeaway"
    TOP = "top"
    IMPACT = "impact"
    FINISH = "finish"


SWING_EVENT_ORDER: tuple[SwingEvent, ...] = (
    SwingEvent.ADDRESS,
    SwingEvent.TAKEAWAY,
    SwingEvent.TOP,
    SwingEvent.IMPACT,
    SwingEvent.FINISH,
)

DEFAULT_CANONICAL_EVENT_PHASES: dict[SwingEvent, float] = {
    SwingEvent.ADDRESS: 0.0,
    SwingEvent.TAKEAWAY: 0.20,
    SwingEvent.TOP: 0.55,
    SwingEvent.IMPACT: 0.75,
    SwingEvent.FINISH: 1.0,
}


@dataclass(frozen=True)
class TrialProvenance:
    """Provenance, acquisition context, and anonymisation status of a swing capture."""

    subject_id: str
    session_id: str = ""
    collection_date: str = ""
    anonymized: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiSwingTrial:
    """Container for one swing capture with observations, event anchors, and provenance."""

    trial_id: str
    times: FloatArray
    trajectories: Mapping[str, FloatArray]
    events: Mapping[SwingEvent, float]
    provenance: TrialProvenance

    def __post_init__(self) -> None:
        require(bool(self.trial_id), "trial_id must be non-empty")
        require(self.times.ndim == 1 and self.times.size >= 2, "need >= 2 times")
        require(
            bool(np.all(np.diff(self.times) > 0.0)),
            "trial times must be strictly increasing",
        )
        require(
            bool(np.all(np.isfinite(self.times))),
            "trial times must be finite",
        )

        # Enforce presence of all canonical events
        for event in SWING_EVENT_ORDER:
            require(
                event in self.events,
                f"missing required swing event '{event.value}' in trial {self.trial_id}",
            )
            require(
                np.isfinite(self.events[event]),
                f"event time for '{event.value}' must be finite",
            )

        # Enforce strict chronological ordering of events
        event_times = [self.events[e] for e in SWING_EVENT_ORDER]
        for i in range(len(event_times) - 1):
            prev_e = SWING_EVENT_ORDER[i]
            next_e = SWING_EVENT_ORDER[i + 1]
            require(
                event_times[i] < event_times[i + 1],
                f"event '{next_e.value}' (t={event_times[i + 1]}) must occur strictly after "
                f"'{prev_e.value}' (t={event_times[i]})",
            )

        t_min = float(self.times[0])
        t_max = float(self.times[-1])
        require(
            event_times[0] >= t_min - 1e-9,
            f"address time {event_times[0]} before start {t_min}",
        )
        require(
            event_times[-1] <= t_max + 1e-9,
            f"finish time {event_times[-1]} after end {t_max}",
        )

        n_nodes = self.times.size
        for name, traj in self.trajectories.items():
            require(
                traj.shape[0] == n_nodes,
                f"trajectory '{name}' length {traj.shape[0]} != times length {n_nodes}",
            )
            require(
                bool(np.all(np.isfinite(traj))),
                f"trajectory '{name}' must contain only finite numbers",
            )


@dataclass(frozen=True)
class PhaseRegistration:
    """Piecewise monotonic time-to-phase warping anchored at canonical swing events."""

    event_times: tuple[float, ...]
    canonical_phases: tuple[float, ...]

    def __post_init__(self) -> None:
        require(len(self.event_times) >= 2, "need >= 2 event times")
        require(
            len(self.event_times) == len(self.canonical_phases),
            "event_times and canonical_phases lengths must match",
        )
        require(
            all(
                self.event_times[i] < self.event_times[i + 1]
                for i in range(len(self.event_times) - 1)
            ),
            "event_times must be strictly increasing",
        )
        require(
            all(
                self.canonical_phases[i] < self.canonical_phases[i + 1]
                for i in range(len(self.canonical_phases) - 1)
            ),
            "canonical_phases must be strictly increasing",
        )
        require(
            self.canonical_phases[0] == 0.0 and self.canonical_phases[-1] == 1.0,
            "canonical_phases must span [0, 1]",
        )

    @classmethod
    def fit(
        cls,
        trial: MultiSwingTrial,
        canonical_event_phases: Mapping[SwingEvent, float] | None = None,
    ) -> PhaseRegistration:
        """Construct phase registration mapping from a trial's event anchors."""
        anchors = (
            DEFAULT_CANONICAL_EVENT_PHASES
            if canonical_event_phases is None
            else canonical_event_phases
        )
        for event in SWING_EVENT_ORDER:
            require(
                event in anchors,
                f"missing canonical phase for '{event.value}'",
            )

        t_events = tuple(float(trial.events[e]) for e in SWING_EVENT_ORDER)
        tau_events = tuple(float(anchors[e]) for e in SWING_EVENT_ORDER)
        return cls(event_times=t_events, canonical_phases=tau_events)

    def time_to_phase(self, times: FloatArray | float) -> FloatArray:
        """Map trial time(s) to canonical phase tau in [0, 1]."""
        t_arr = np.asarray(times, dtype=np.float64)
        phases = np.interp(
            t_arr,
            self.event_times,
            self.canonical_phases,
            left=self.canonical_phases[0],
            right=self.canonical_phases[-1],
        )
        return np.asarray(phases, dtype=np.float64)

    def phase_to_time(self, phases: FloatArray | float) -> FloatArray:
        """Map canonical phase tau in [0, 1] back to trial time."""
        tau_arr = np.asarray(phases, dtype=np.float64)
        times = np.interp(
            tau_arr,
            self.canonical_phases,
            self.event_times,
            left=self.event_times[0],
            right=self.event_times[-1],
        )
        return np.asarray(times, dtype=np.float64)

    def resample_trajectory(
        self,
        trial_times: FloatArray,
        trajectory: FloatArray,
        phase_grid: FloatArray,
    ) -> FloatArray:
        """Resample a trajectory onto the specified canonical phase grid."""
        require(phase_grid.ndim == 1, "phase_grid must be 1-D")
        require(
            trajectory.shape[0] == trial_times.size,
            "trajectory length must match trial times",
        )
        # Map target phase grid to trial times
        sample_times = self.phase_to_time(phase_grid)

        orig_shape = trajectory.shape[1:]
        flat_traj = trajectory.reshape(trajectory.shape[0], -1)
        resampled_cols = [
            np.interp(sample_times, trial_times, flat_traj[:, col])
            for col in range(flat_traj.shape[1])
        ]
        resampled = np.column_stack(resampled_cols).reshape(
            phase_grid.size, *orig_shape
        )
        return np.asarray(resampled, dtype=np.float64)


@dataclass(frozen=True)
class AlignedTrials:
    """Collection of trials registered onto a shared canonical phase grid."""

    trials: tuple[MultiSwingTrial, ...]
    phase_grid: FloatArray
    per_trial: FloatArray
    template: FloatArray


def align_trials_to_phase(
    trials: Sequence[MultiSwingTrial],
    trajectory_key: str = "inputs",
    n_phase_bins: int = 50,
    canonical_event_phases: Mapping[SwingEvent, float] | None = None,
) -> AlignedTrials:
    """Warp trials to a uniform canonical phase grid anchored at swing events."""
    require(len(trials) >= 1, "at least one trial required")
    require(n_phase_bins >= 2, "n_phase_bins must be >= 2", n_phase_bins)

    phase_grid = np.linspace(0.0, 1.0, n_phase_bins, dtype=np.float64)
    resampled_trials: list[FloatArray] = []

    for trial in trials:
        require(
            trajectory_key in trial.trajectories,
            f"trajectory '{trajectory_key}' missing in trial {trial.trial_id}",
        )
        reg = PhaseRegistration.fit(trial, canonical_event_phases)
        traj = trial.trajectories[trajectory_key]
        resampled = reg.resample_trajectory(trial.times, traj, phase_grid)
        resampled_trials.append(resampled)

    per_trial = np.stack(resampled_trials, axis=0)
    template = per_trial.mean(axis=0)

    return AlignedTrials(
        trials=tuple(trials),
        phase_grid=phase_grid,
        per_trial=per_trial,
        template=template,
    )


def evaluate_registration_variance(
    trials: Sequence[MultiSwingTrial],
    trajectory_key: str = "inputs",
    n_phase_bins: int = 50,
    n_modes: int = 1,
) -> tuple[float, float]:
    """Compute first-mode explained variance before and after event-anchored registration.

    Returns:
        (variance_raw, variance_registered)
    """
    require(len(trials) >= 2, "need >= 2 trials to evaluate variance")
    n_trials = len(trials)

    # 1. Raw linear time normalization (no event anchoring)
    raw_resampled = []
    norm_grid = np.linspace(0.0, 1.0, n_phase_bins, dtype=np.float64)
    for trial in trials:
        traj = trial.trajectories[trajectory_key]
        t0, t_end = trial.times[0], trial.times[-1]
        norm_times = (trial.times - t0) / (t_end - t0)
        flat_traj = traj.reshape(traj.shape[0], -1)
        resampled = np.column_stack(
            [
                np.interp(norm_grid, norm_times, flat_traj[:, c])
                for c in range(flat_traj.shape[1])
            ]
        ).reshape(n_phase_bins, *traj.shape[1:])
        raw_resampled.append(resampled)

    per_trial_raw = np.stack(raw_resampled, axis=0)
    pm_raw = principal_modes(per_trial_raw, n_modes=n_modes)
    var_raw = float(pm_raw.explained_variance_ratio[0])

    # 2. Event-anchored registration
    aligned = align_trials_to_phase(
        trials, trajectory_key=trajectory_key, n_phase_bins=n_phase_bins
    )
    pm_registered = principal_modes(aligned.per_trial, n_modes=n_modes)
    var_registered = float(pm_registered.explained_variance_ratio[0])

    return var_raw, var_registered


def ingest_swing_trial(
    raw_data: Mapping[str, Any],
    anonymize: bool = True,
) -> MultiSwingTrial:
    """Validate, anonymise, and construct a `MultiSwingTrial` from raw ingest data.

    Enforces that all events are present and strictly chronological; rejects
    missing or mislabelled events via `require`.
    """
    require("trial_id" in raw_data, "raw_data must contain 'trial_id'")
    require("times" in raw_data, "raw_data must contain 'times'")
    require("trajectories" in raw_data, "raw_data must contain 'trajectories'")
    require("events" in raw_data, "raw_data must contain 'events'")

    trial_id = str(raw_data["trial_id"])
    times = np.asarray(raw_data["times"], dtype=np.float64)

    raw_trajectories = raw_data["trajectories"]
    require(isinstance(raw_trajectories, Mapping), "trajectories must be a mapping")
    trajectories = {
        str(k): np.asarray(v, dtype=np.float64) for k, v in raw_trajectories.items()
    }

    raw_events = raw_data["events"]
    require(isinstance(raw_events, Mapping), "events must be a mapping")

    parsed_events: dict[SwingEvent, float] = {}
    for event in SWING_EVENT_ORDER:
        val = raw_events.get(event.value, raw_events.get(event))
        require(
            val is not None,
            f"ingest data missing required event '{event.value}' in trial {trial_id}",
        )
        parsed_events[event] = float(val)

    # Anonymisation handling
    subject_raw = str(
        raw_data.get(
            "subject_name",
            raw_data.get("subject_id", "anonymous_subject"),
        )
    )
    session_id = str(raw_data.get("session_id", ""))
    collection_date = str(raw_data.get("collection_date", ""))

    if anonymize:
        # Create deterministic salted pseudonymous hash
        salt = "upstreamdrift_mosaic_anonymize"
        digest = hashlib.sha256(f"{salt}_{subject_raw}".encode()).hexdigest()[:8]
        subject_id = f"anon_{digest}"
    else:
        subject_id = subject_raw

    # Filter out direct PII from metadata
    pii_keys = {"subject_name", "name", "email", "phone", "dob", "address"}
    meta = {
        str(k): v
        for k, v in raw_data.items()
        if str(k).lower() not in pii_keys
        and str(k) not in {"trial_id", "times", "trajectories", "events"}
    }

    provenance = TrialProvenance(
        subject_id=subject_id,
        session_id=session_id,
        collection_date=collection_date,
        anonymized=anonymize,
        metadata=meta,
    )

    return MultiSwingTrial(
        trial_id=trial_id,
        times=times,
        trajectories=trajectories,
        events=parsed_events,
        provenance=provenance,
    )
