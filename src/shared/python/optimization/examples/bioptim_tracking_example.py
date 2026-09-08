"""Recover a dynamically consistent swing from 3D keypoints (epic #9762).

Runs the Phase 3 tracking OCP end to end on a synthetic swing and prints
what it recovered, including what the marker set could *not* observe. No
plots, no external data: it is meant to be runnable in CI and readable as
documentation of the API.

    MPLBACKEND=Agg python -m src.shared.python.optimization.examples.bioptim_tracking_example

Requires the bioptim extra::

    pip install -e '.[bioptim]'      # Debian/Ubuntu also need python3-tk
"""

from __future__ import annotations

import logging

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)
from src.shared.python.optimization._swing_models import ClubModel, GolferModel

logger = logging.getLogger(__name__)

N_FRAMES = 9
DURATION_S = 0.4
MARKER_NOISE_M = 0.005


def synthetic_keypoints(
    golfer: GolferModel, club: ClubModel
) -> tuple[KeypointSequence, np.ndarray]:
    """A smooth swing projected onto the model's markers, plus 5 mm noise."""
    from src.shared.python.optimization.ocp.symbolic_model import (
        MARKER_NAMES,
        SymbolicSwingModel,
    )

    model = SymbolicSwingModel(golfer, club)
    empty = np.zeros(0)
    times = np.linspace(0.0, DURATION_S, N_FRAMES)
    amplitude = np.array([0.3, 0.5, 0.2, 0.6, 0.4, -0.3, 0.15])
    q_true = amplitude[:, None] * np.sin(np.linspace(0.0, np.pi, N_FRAMES))[None, :]

    rng = np.random.default_rng(0)
    frames = []
    for index, timestamp in enumerate(times):
        markers = np.asarray(model.markers(q_true[:, index], empty))
        keypoints = [
            Keypoint(
                x=float(markers[0, column] + rng.normal(0.0, MARKER_NOISE_M)),
                y=float(markers[1, column] + rng.normal(0.0, MARKER_NOISE_M)),
                z=float(markers[2, column] + rng.normal(0.0, MARKER_NOISE_M)),
                confidence=0.9,
                name=name,
            )
            for column, name in enumerate(MARKER_NAMES)
        ]
        frames.append(
            KeypointFrame(
                timestamp=float(timestamp),
                keypoints=keypoints,
                schema_name="custom",
                frame_index=index,
            )
        )
    return KeypointSequence(id="example-swing", frames=frames), q_true


def main() -> int:
    """Track the synthetic keypoints and report the fit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.shared.python.optimization.ocp._compat import (
        BIOPTIM_INSTALL_HINT,
        bioptim_available,
    )

    if not bioptim_available():
        print(BIOPTIM_INSTALL_HINT)
        return 1

    from src.shared.python.optimization.ocp.tracking_ocp import (
        TrackingWeights,
        keypoints_to_targets,
        solve_tracking_ocp,
    )

    golfer, club = GolferModel(), ClubModel()
    sequence, q_true = synthetic_keypoints(golfer, club)
    targets = keypoints_to_targets(sequence)

    result = solve_tracking_ocp(
        targets,
        golfer,
        club,
        weights=TrackingWeights(marker=1.0e4, torque=1.0e-6, qdot_derivative=1.0e-3),
        q_guess=np.zeros_like(q_true),
        max_iterations=300,
    )

    print(
        f"solver status      : {result.status} ({'ok' if result.success else 'failed'})"
    )
    print(f"iterations         : {result.iterations}")
    print(f"wall time          : {result.wall_time_s:.1f} s")
    print(f"marker noise added : {MARKER_NOISE_M * 1000:.0f} mm")
    print(
        "marker RMS [mm]    : "
        + ", ".join(
            f"{name}={rms * 1000:.1f}" for name, rms in result.marker_rms_m.items()
        )
    )
    print(f"peak torque [N*m]  : {np.abs(result.tau).max():.1f}")

    report = result.identifiability
    if report is not None:
        print(f"observable DOFs    : {report.rank} of {len(report.dof_names)}")
        print(f"condition number   : {report.condition_number:.3g}")
        print(f"unobservable       : {list(report.unobservable_dofs) or 'none'}")
        print(f"weakly observable  : {list(report.weakly_observable_dofs) or 'none'}")
        print(
            "Joint angles along those directions fit the markers equally well "
            "at many values; read the marker trajectory, not the angles."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
