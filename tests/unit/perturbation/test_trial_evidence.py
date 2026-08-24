"""Tests for complete, typed UpstreamDrift variation-trial evidence."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    ImpactObservation,
    SampledInput,
    TrialTrace,
)

pytestmark = pytest.mark.unit

_PLAN_SHA = "a" * 64
_TOOLS_REVISION = "b" * 40
_ENGINE_REVISION = "c" * 40


def test_trial_evidence_is_exposed_by_public_perturbation_package() -> None:
    assert perturbation.CanonicalTrialEvidence is CanonicalTrialEvidence
    assert perturbation.TrialTrace is TrialTrace


def _trace(*, complete: bool = True) -> TrialTrace:
    return TrialTrace(
        times_s=np.array([0.0, 0.01, 0.02]),
        q=np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]]),
        v=np.array([[0.0, 0.0], [1.0, 2.0], [1.5, 1.0]]),
        coordinate_ids=("shoulder_angle", "club_angle"),
        coordinate_units=("rad", "rad"),
        velocity_units=("rad/s", "rad/s"),
        markers_m=np.array(
            [
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [0.6, 0.1, 0.0]],
                [[0.2, 0.0, 0.0], [0.7, 0.2, 0.0]],
            ]
        ),
        marker_ids=("lead_hand", "clubhead"),
        frame_id="world-z-up",
        alignment_id="downswing-start/v1",
        complete=complete,
    )


def _common() -> dict[str, object]:
    return {
        "trial_index": 4,
        "seed": 123,
        "plan_sha256": _PLAN_SHA,
        "tools_revision": _TOOLS_REVISION,
        "engine_id": "mujoco",
        "engine_revision": _ENGINE_REVISION,
        "model_id": "two-dof-double-pendulum/v1",
        "sampled_inputs": (
            SampledInput("proximal_torque_peak", 42.0, "N*m"),
            SampledInput("distal_torque_peak", -3.0, "N*m"),
        ),
    }


def test_hit_retains_complete_trace_and_impact_without_copying_semantics() -> None:
    record = CanonicalTrialEvidence(
        **_common(),
        outcome="hit",
        trace=_trace(),
        impact=ImpactObservation(
            time_s=0.02,
            state=(
                SampledInput("clubhead_speed", 51.2, "m/s"),
                SampledInput("face_angle", 0.01, "rad"),
            ),
        ),
        shot_result=(SampledInput("carry", 248.0, "m"),),
    )

    assert record.outcome == "hit"
    assert record.trace is not None and record.trace.complete is True
    assert record.trace.marker_ids == ("lead_hand", "clubhead")
    assert record.impact is not None and record.impact.time_s == 0.02
    assert record.failure_reason is None
    assert record.trace.markers_m.flags.writeable is False


def test_no_impact_is_valid_evidence_with_null_shot_and_closest_approach() -> None:
    record = CanonicalTrialEvidence(
        **_common(),
        outcome="no_impact",
        trace=_trace(),
        closest_approach=ClosestApproach(
            time_s=0.02,
            distance_m=0.014,
            source_marker_id="clubhead",
            target_id="ball-center",
            contact_observed=False,
        ),
    )

    assert record.impact is None
    assert record.shot_result is None
    assert record.closest_approach is not None
    assert record.closest_approach.distance_m == 0.014


def test_partial_valid_trace_requires_failure_reason_and_incomplete_trace() -> None:
    record = CanonicalTrialEvidence(
        **_common(),
        outcome="partial_valid_trace",
        trace=_trace(complete=False),
        failure_reason="Integrator diverged after the retained finite prefix.",
    )

    assert record.trace is not None and record.trace.complete is False
    assert record.failure_reason.startswith("Integrator diverged")

    with pytest.raises(ValueError, match="incomplete trace"):
        CanonicalTrialEvidence(
            **_common(),
            outcome="partial_valid_trace",
            trace=_trace(complete=True),
            failure_reason="Integrator diverged after the retained finite prefix.",
        )


def test_numerical_failure_never_fabricates_impact_or_shot_data() -> None:
    with pytest.raises(ValueError, match="must not contain impact or shot"):
        CanonicalTrialEvidence(
            **_common(),
            outcome="numerical_failure",
            trace=None,
            failure_reason="Engine rejected the initial state as infeasible.",
            impact=ImpactObservation(time_s=0.0, state=()),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"marker_ids": ("clubhead", "clubhead")}, "marker IDs must be unique"),
        ({"coordinate_ids": ("only_one",)}, "coordinate IDs"),
        ({"frame_id": ""}, "frame_id"),
        ({"times_s": np.array([0.0, 0.01, 0.005])}, "strictly increasing"),
        (
            {"markers_m": np.array([[[np.nan, 0.0, 0.0]]] * 3)},
            "markers_m must be finite",
        ),
    ],
)
def test_trace_rejects_anonymous_misaligned_or_nonfinite_geometry(
    override: dict[str, object], message: str
) -> None:
    values = {
        "times_s": np.array([0.0, 0.01, 0.02]),
        "q": np.zeros((3, 2)),
        "v": np.zeros((3, 2)),
        "coordinate_ids": ("q0", "q1"),
        "coordinate_units": ("rad", "rad"),
        "velocity_units": ("rad/s", "rad/s"),
        "markers_m": np.zeros((3, 2, 3)),
        "marker_ids": ("hand", "clubhead"),
        "frame_id": "world-z-up",
        "alignment_id": "downswing-start/v1",
        "complete": True,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        TrialTrace(**values)  # type: ignore[arg-type]


def test_trial_rejects_incoherent_outcome_and_provenance() -> None:
    with pytest.raises(ValueError, match="40-character"):
        CanonicalTrialEvidence(
            **{**_common(), "tools_revision": "main"},
            outcome="no_impact",
            trace=_trace(),
            closest_approach=ClosestApproach(
                0.02, 0.01, "clubhead", "ball-center", False
            ),
        )

    with pytest.raises(ValueError, match="requires impact"):
        CanonicalTrialEvidence(
            **_common(),
            outcome="hit",
            trace=_trace(),
        )

    with pytest.raises(ValueError, match="requires closest approach"):
        CanonicalTrialEvidence(
            **_common(),
            outcome="no_impact",
            trace=_trace(),
        )
