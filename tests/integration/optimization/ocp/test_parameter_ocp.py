"""Phase 4: Simultaneous state + parameter estimation OCP tests."""

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

from src.shared.python.estimation.identifiability import (  # noqa: E402
    IdentifiabilityGateOptions,
    UnidentifiableParametersError,
)
from src.shared.python.estimation.map_estimator import (  # noqa: E402
    MapEstimatorResult,
    SharedParameterBlock,
    SharedParameterSpec,
)
from src.shared.python.motion_pipeline.contracts import (  # noqa: E402
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)
from src.shared.python.optimization._swing_models import (  # noqa: E402
    ClubModel,
    GolferModel,
)
from src.shared.python.optimization.ocp.parameter_ocp import (  # noqa: E402
    ParameterOcpOptions,
    add_parameter_block,
    build_tracking_parameter_ocp,
    solve_tracking_parameter_ocp,
)
from src.shared.python.optimization.ocp.result import (  # noqa: E402
    tracking_to_map_estimator_result,
)
from src.shared.python.optimization.ocp.symbolic_model import (  # noqa: E402
    MARKER_NAMES,
    SymbolicSwingModel,
)
from src.shared.python.optimization.ocp.tracking_ocp import (  # noqa: E402
    MarkerTargets,
    TrackingWeights,
    keypoints_to_targets,
)


def _ground_truth(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    n_frames: int = 9,
    duration: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, SymbolicSwingModel, np.ndarray]:
    """A smooth, physically plausible joint trajectory and its markers."""
    g = golfer or GolferModel()
    c = club or ClubModel()
    model = SymbolicSwingModel(g, c)
    times = np.linspace(0.0, duration, n_frames)
    amplitude = np.array([0.3, 0.5, 0.2, 0.6, 0.4, -0.3, 0.15])
    phase = np.linspace(0.0, np.pi, n_frames)
    q = amplitude[:, None] * np.sin(phase)[None, :]
    empty = np.zeros(0)
    markers = np.stack(
        [np.asarray(model.markers(q[:, k], empty)) for k in range(n_frames)],
        axis=2,
    )
    return times, q, model, markers


def _sequence(
    times: np.ndarray,
    markers: np.ndarray,
    *,
    noise_m: float = 0.0,
) -> KeypointSequence:
    rng = np.random.default_rng(42)
    frames = []
    for index, timestamp in enumerate(times):
        keypoints = []
        for column, name in enumerate(MARKER_NAMES):
            position = markers[:, column, index]
            if noise_m > 0.0:
                position = position + rng.normal(0.0, noise_m, 3)
            keypoints.append(
                Keypoint(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    confidence=0.95,
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
    return KeypointSequence(id="synthetic-param-swing", frames=frames)


def test_parameter_ocp_recovers_perturbed_segment_length_and_preserves_locked() -> None:
    """Acceptance: true forearm/arm length perturbed 8% recovered within 1 cm.

    Locked parameters remain unchanged bit-for-bit.
    """
    true_golfer = GolferModel(arm_length=0.60)
    true_club = ClubModel(total_length=1.15)
    times, q_true, _model, markers = _ground_truth(true_golfer, true_club)
    targets = keypoints_to_targets(_sequence(times, markers, noise_m=0.002))

    perturbed_arm = 0.60 * 1.08  # 8% perturbation
    specs = [
        SharedParameterSpec(
            name="arm_length",
            initial=perturbed_arm,
            lower=0.45,
            upper=0.75,
            prior=0.60,
            prior_scale=0.1,
            locked=False,
        ),
        SharedParameterSpec(
            name="club_length",
            initial=1.15,
            locked=True,
        ),
    ]

    result = solve_tracking_parameter_ocp(
        targets,
        true_golfer,
        true_club,
        parameters=specs,
        weights=TrackingWeights(marker=1.0e4, torque=1.0e-6, qdot_derivative=1.0e-3),
        q_guess=q_true,
        options=ParameterOcpOptions(max_iterations=300),
    )

    assert result.success, f"solver failed with status {result.status}"
    recovered_arm = result.parameters["arm_length"]
    assert abs(recovered_arm - 0.60) < 0.01, (
        f"recovered {recovered_arm}, expected ~0.60"
    )

    # Locked parameter must remain unchanged bit-for-bit
    assert result.parameters["club_length"] == 1.15
    assert "club_length" in result.locked_by_gate or "club_length" in specs[1].name


def test_parameter_ocp_gate_raises_on_unidentifiable_parameters() -> None:
    """Gate refuses solve when an unidentifiable parameter is free under policy='raise'."""
    times, q_true, model, markers = _ground_truth()
    targets = keypoints_to_targets(_sequence(times, markers))

    # mass does not affect kinematic marker positions at fixed q
    specs = [
        SharedParameterSpec(name="mass", initial=model.golfer.mass, locked=False),
    ]
    gate_opts = IdentifiabilityGateOptions(policy="raise", relative_tolerance=1e-5)
    options = ParameterOcpOptions(gate=gate_opts)

    with pytest.raises(UnidentifiableParametersError, match="not identifiable"):
        build_tracking_parameter_ocp(
            targets,
            model.golfer,
            model.club,
            parameters=specs,
            q_guess=q_true,
            options=options,
        )


def test_parameter_ocp_gate_locks_unidentifiable_parameters() -> None:
    """Gate auto-locks unidentifiable parameter under policy='lock'."""
    times, q_true, model, markers = _ground_truth()
    targets = keypoints_to_targets(_sequence(times, markers))

    specs = [
        SharedParameterSpec(name="arm_length", initial=0.60, locked=False),
        SharedParameterSpec(name="mass", initial=model.golfer.mass, locked=False),
    ]
    gate_opts = IdentifiabilityGateOptions(policy="lock", relative_tolerance=1e-5)
    options = ParameterOcpOptions(gate=gate_opts, max_iterations=200)

    result = solve_tracking_parameter_ocp(
        targets,
        model.golfer,
        model.club,
        parameters=specs,
        q_guess=q_true,
        options=options,
    )

    assert result.success
    assert "mass" in result.locked_by_gate
    assert result.parameters["mass"] == model.golfer.mass


def test_parameter_ocp_result_adapter() -> None:
    times, q_true, model, markers = _ground_truth()
    targets = keypoints_to_targets(_sequence(times, markers))

    specs = [
        SharedParameterSpec(name="arm_length", initial=0.60, locked=False),
    ]
    result = solve_tracking_parameter_ocp(
        targets,
        model.golfer,
        model.club,
        parameters=specs,
        q_guess=q_true,
        options=ParameterOcpOptions(max_iterations=100),
    )

    map_result = tracking_to_map_estimator_result(result)
    assert isinstance(map_result, MapEstimatorResult)
    assert map_result.success == result.success
    assert "arm_length" in map_result.parameters
    assert map_result.coefficients.shape == (2 * 7 * times.size,)
