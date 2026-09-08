"""Phase 3: keypoint tracking recovers a dynamically consistent trajectory."""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.shared.python.optimization.ocp import _compat

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_bioptim,
    pytest.mark.skipif(not _compat.bioptim_available(), reason="bioptim not installed"),
]

os.environ.setdefault("MPLBACKEND", "Agg")

from src.shared.python.motion_pipeline.contracts import (  # noqa: E402
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)
from src.shared.python.optimization._swing_models import (  # noqa: E402
    ClubModel,
    GolferModel,
)
from src.shared.python.optimization.ocp.symbolic_model import (  # noqa: E402
    MARKER_NAMES,
    SymbolicSwingModel,
)
from src.shared.python.optimization.ocp.tracking_ocp import (  # noqa: E402
    MarkerTargets,
    TrackingWeights,
    keypoints_to_targets,
    solve_tracking_ocp,
)

_EMPTY = np.zeros(0)


def _ground_truth(
    n_frames: int = 9, duration: float = 0.4
) -> tuple[np.ndarray, np.ndarray, SymbolicSwingModel, np.ndarray]:
    """A smooth, physically plausible joint trajectory and its markers."""
    model = SymbolicSwingModel(GolferModel(), ClubModel())
    times = np.linspace(0.0, duration, n_frames)
    amplitude = np.array([0.3, 0.5, 0.2, 0.6, 0.4, -0.3, 0.15])
    phase = np.linspace(0.0, np.pi, n_frames)
    q = amplitude[:, None] * np.sin(phase)[None, :]
    markers = np.stack(
        [np.asarray(model.markers(q[:, k], _EMPTY)) for k in range(n_frames)], axis=2
    )
    return times, q, model, markers


def _sequence(
    times: np.ndarray,
    markers: np.ndarray,
    *,
    noise_m: float = 0.0,
    drop: tuple[int, ...] = (),
) -> KeypointSequence:
    rng = np.random.default_rng(3)
    frames = []
    for index, timestamp in enumerate(times):
        keypoints = []
        for column, name in enumerate(MARKER_NAMES):
            position = markers[:, column, index]
            if noise_m:
                position = position + rng.normal(0.0, noise_m, 3)
            confidence = 0.0 if index in drop else 0.9
            keypoints.append(
                Keypoint(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    confidence=confidence,
                    name=name,
                )
            )
        frames.append(
            KeypointFrame(
                timestamp=float(timestamp),
                keypoints=keypoints,
                schema_name="custom",
                frame_index=index,
            )
        )
    return KeypointSequence(id="synthetic-swing", frames=frames)


def test_keypoints_to_targets_maps_markers_and_confidence() -> None:
    times, _q, _model, markers = _ground_truth()
    targets = keypoints_to_targets(_sequence(times, markers, drop=(2,)))
    assert targets.marker_names == MARKER_NAMES
    assert targets.positions.shape == (3, len(MARKER_NAMES), times.size)
    assert targets.weights.shape == (len(MARKER_NAMES), times.size)
    np.testing.assert_allclose(targets.positions, markers, atol=1e-12)
    assert np.all(targets.weights[:, 2] == 0.0)
    assert np.all(targets.weights[:, 0] == pytest.approx(0.9))
    assert targets.n_frames == times.size
    assert targets.duration == pytest.approx(times[-1] - times[0])


def test_targets_reject_non_uniform_and_2d_input() -> None:
    times, _q, _model, markers = _ground_truth(n_frames=5)
    uneven = times.copy()
    uneven[2] += 0.01
    with pytest.raises(ValueError, match="uniformly spaced"):
        MarkerTargets(
            times=uneven,
            positions=markers,
            weights=np.ones((len(MARKER_NAMES), times.size)),
            marker_names=MARKER_NAMES,
        )
    flat = _sequence(times, markers)
    flat.frames[0].keypoints[0].z = None
    with pytest.raises(ValueError, match="no z"):
        keypoints_to_targets(flat)


def test_marker_set_leaves_the_shaft_roll_unobservable() -> None:
    """A documented limitation, reported rather than hidden (#9758).

    Every marker except ``clubface`` sits on a joint origin, and even with
    ``clubface`` the seven-DOF chain has a one-dimensional null direction at
    a mid-swing pose. The probe must name it.
    """
    from src.shared.python.optimization.ocp.tracking_ocp import (
        probe_marker_identifiability,
    )

    _times, q_true, model, _markers = _ground_truth()
    report = probe_marker_identifiability(model, q_true[:, 4])
    assert report.rank < len(report.dof_names)
    assert not report.is_full_rank
    # hip_rotation and trunk_rotation are both Z rotations about Z offsets:
    # turning one and counter-turning the other moves no marker.
    assert set(report.unobservable_dofs) == {"hip_rotation", "trunk_rotation"}
    # The terminal shaft roll is only weakly seen, via the clubface offset.
    assert "wrist_rotation" in report.weakly_observable_dofs
    assert report.condition_number > 100.0
    assert set(report.reliable_dofs) <= set(report.dof_names)
    assert report.to_dict()["n_dof"] == 7


def test_marker_set_is_ill_conditioned_for_reading_joint_angles() -> None:
    """The singular spectrum spans two orders of magnitude at a mid pose."""
    from src.shared.python.optimization.ocp.tracking_ocp import (
        probe_marker_identifiability,
    )

    _times, q_true, model, _markers = _ground_truth()
    report = probe_marker_identifiability(model, q_true[:, 4])
    finite = report.singular_values[report.singular_values > 1e-9]
    assert float(finite[0] / finite[-1]) > 50.0
    # Most of the chain is therefore not reliably readable joint by joint.
    assert len(report.reliable_dofs) < len(report.dof_names) - 2


def test_tracking_fits_the_markers_and_reports_what_it_cannot_see() -> None:
    times, q_true, model, markers = _ground_truth()
    targets = keypoints_to_targets(_sequence(times, markers, noise_m=0.005))
    result = solve_tracking_ocp(
        targets,
        model.golfer,
        model.club,
        weights=TrackingWeights(marker=1.0e4, torque=1.0e-6, qdot_derivative=1.0e-3),
        q_guess=np.zeros_like(q_true),
        max_iterations=300,
    )
    assert result.success, result.status
    assert result.q.shape == q_true.shape
    assert result.tau.shape == (7, times.size - 1)
    # Marker residuals sit at the 5 mm noise level, not at model error.
    for name, rms in result.marker_rms_m.items():
        assert rms < 0.03, (name, rms)
    # Joint-angle accuracy is NOT asserted, and that is the point: this
    # marker set determines the marker trajectory, not the joint angles
    # behind it. The result says so instead of implying a precision it does
    # not have. The two worst-recovered joints here are exactly the ones the
    # probe declines to call reliable.
    assert result.identifiability is not None
    rms_deg = np.rad2deg(np.sqrt(np.mean((result.q - q_true) ** 2, axis=1)))
    worst = {
        result.identifiability.dof_names[index] for index in np.argsort(rms_deg)[-2:]
    }
    assert not worst & set(result.identifiability.reliable_dofs), (
        worst,
        result.identifiability.to_dict(),
    )


def test_recovered_trajectory_is_dynamically_consistent() -> None:
    """Whatever the marker set cannot see, the output still obeys the ODE."""
    from src.shared.python.optimization._swing_models import OptimizationConfig
    from src.shared.python.optimization.casadi_backend import dynamics_defect

    times, q_true, model, markers = _ground_truth()
    targets = keypoints_to_targets(_sequence(times, markers))
    result = solve_tracking_ocp(
        targets,
        model.golfer,
        model.club,
        weights=TrackingWeights(marker=1.0e4, torque=1.0e-6, qdot_derivative=1.0e-3),
        q_guess=q_true,
        max_iterations=300,
    )
    assert result.success, result.status
    config = OptimizationConfig(
        n_nodes=times.size, swing_duration=float(times[-1] - times[0])
    )
    x = np.concatenate([result.q.flatten(), result.qdot.flatten()])
    # Re-integrate on the transcription's own grid (RK4, 2 substeps).
    report = dynamics_defect(
        model.golfer, model.club, config, x, torques=result.tau, n_substeps=2
    )
    assert report.max_defect < 1e-4, report.to_dict()


def test_dropped_frames_are_masked_not_tracked() -> None:
    times, q_true, model, markers = _ground_truth()
    corrupted = markers.copy()
    corrupted[:, :, 3] += 5.0  # a frame that is nonsense
    targets = keypoints_to_targets(_sequence(times, corrupted, drop=(3,)))
    assert np.all(targets.weights[:, 3] == 0.0)
    result = solve_tracking_ocp(
        targets,
        model.golfer,
        model.club,
        weights=TrackingWeights(marker=1.0e4, torque=1.0e-6, qdot_derivative=1.0e-3),
        q_guess=np.zeros_like(q_true),
        max_iterations=300,
    )
    assert result.success, result.status
    # The masked frame contributes nothing, so the 5 m outlier does not drag
    # the fit: residuals on the observed frames stay at model accuracy.
    for name, rms in result.marker_rms_m.items():
        assert rms < 0.05, (name, rms)


def test_tracking_weights_contracts() -> None:
    with pytest.raises(ValueError, match="marker weight"):
        TrackingWeights(marker=0.0)
    with pytest.raises(ValueError, match="regulariser"):
        TrackingWeights(torque=-1.0)
    with pytest.raises(ValueError, match="min_confidence"):
        TrackingWeights(min_confidence=1.5)
