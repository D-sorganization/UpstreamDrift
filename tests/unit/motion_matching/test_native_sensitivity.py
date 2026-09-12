"""Native sensitivity replay preserves physical effort units and clock."""

import hashlib
import json

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativeAccelerationDerivatives,
    NativeMarkerDerivatives,
)
from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
    replay_marker_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

pytestmark = pytest.mark.unit


def test_free_mass_matches_exact_sextic_force_response() -> None:
    names = ["x", "y", "z"]
    raw = json.dumps(
        {
            "coordinate_order": names,
            "joints": [
                {
                    "parent": "world",
                    "parent_to_base": np.eye(4).tolist(),
                    "primitives": [
                        {"primitive": p, "coordinate": name}
                        for p, name in zip(("Px", "Py", "Pz"), names, strict=True)
                    ],
                }
            ],
        }
    ).encode()
    doc = {
        "schema_version": 1,
        "coordinate_names": names,
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "coefficient_order": "highest-power-first",
        "time_basis": "absolute-seconds",
        "force_frame": "world",
        "duration_s": 0.8,
        "q0": [0.0] * 3,
        "qd0": [0.0] * 3,
        "coefficients": [[0.0] * 7 for _ in names],
        "marker_labels": ["com"],
        "marker_bodies": ["body"],
        "marker_offsets_m": [[0.0, 0.0, 0.0]],
    }
    candidate = NativeReplayCandidate.from_document(
        doc, names, hashlib.sha256(raw).hexdigest()
    )

    class Mass:
        def __init__(self, spec):
            pass

        def accelerations(self, q, v, effort):
            return {k: value / 2 for k, value in effort.items()}

        def acceleration_derivatives(self, q, v, effort):
            return NativeAccelerationDerivatives(
                tuple(q), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3) / 2
            )

        def frame_poses(self, q):
            frame = np.eye(4)
            frame[:3, 3] = list(q.values())
            return {"body": frame}

        def closure_errors(self):
            return np.zeros(6), np.zeros(6)

        def marker_derivatives(self, q, bodies, offsets):
            return NativeMarkerDerivatives(
                tuple(q), np.array([list(q.values())]), np.eye(3)[None]
            )

    result = replay_marker_sensitivities(
        raw, candidate, np.array([0.0, 0.4, 0.8]), model_factory=Mass, max_step=0.02
    )
    assert result.marker_jacobian.shape == (3, 1, 3, 9)
    np.testing.assert_allclose(
        result.marker_jacobian[-1, 0, :, 2], [0.8**2 / 112, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        result.marker_jacobian[-1, 0, :, 5], [0, 0.8**2 / 112, 0], atol=1e-12
    )
    assert not result.marker_jacobian.flags.writeable
    assert result.primal_marker_max_abs_difference_m == 0

    directions = np.eye(6)[:, [0, 3]]
    window = replay_marker_sensitivities(
        raw,
        candidate,
        np.array([0.6, 0.7, 0.8]),
        model_factory=Mass,
        max_step=0.02,
        initial_state=np.zeros(6),
        initial_sensitivity=directions,
    )
    assert window.marker_jacobian.shape == (3, 1, 3, 11)
    np.testing.assert_allclose(window.marker_jacobian[:, 0, 0, -2], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        window.marker_jacobian[:, 0, 0, -1], [0.0, 0.1, 0.2], atol=1e-12
    )
    np.testing.assert_array_equal(window.replay.integration.time, [0.6, 0.7, 0.8])
    np.testing.assert_allclose(window.state_jacobian[-1, 0, -1], 0.2, atol=1e-12)
    assert not window.state_jacobian.flags.writeable
    with pytest.raises(ValueError, match="initial"):
        replay_marker_sensitivities(
            raw,
            candidate,
            np.array([0.0, 0.8]),
            model_factory=Mass,
            initial_sensitivity=directions,
        )
