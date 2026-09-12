"""Adapter contracts use analytic fake dynamics; native qualification is separate."""

import hashlib
import json

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

pytestmark = pytest.mark.unit


class FakeEngine:
    def __init__(self, spec: dict) -> None:
        self.names = spec["coordinate_order"]

    def accelerations(self, q: dict, v: dict, effort: dict) -> dict:
        return effort

    def closure_errors(self) -> tuple:
        return np.zeros(6), np.zeros(6)

    def frame_poses(self, q: dict) -> dict:
        frame = np.eye(4)
        frame[:3, 3] = [q[n] for n in self.names[:3]]
        return {"club": frame}


@pytest.fixture
def package() -> tuple:
    names = ["x", "y", "z"]
    spec = {
        "coordinate_order": names,
        "joints": [
            {
                "parent": "world",
                "parent_to_base": np.eye(4).tolist(),
                "primitives": [
                    {"coordinate": n, "primitive": p}
                    for n, p in zip(names, ["Px", "Py", "Pz"], strict=True)
                ],
            }
        ],
    }
    raw = json.dumps(spec).encode()
    digest = hashlib.sha256(raw).hexdigest()
    doc = {
        "schema_version": 1,
        "coordinate_names": names,
        "model_sha256": digest,
        "capture_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "coefficient_order": "highest-power-first",
        "time_basis": "absolute-seconds",
        "force_frame": "world",
        "duration_s": 0.8,
        "q0": [0.0] * 3,
        "qd0": [0.0] * 3,
        "coefficients": [[1.0, 0, 0, 0, 0, 0, 0], [0.0] * 7, [0.0] * 7],
        "marker_labels": ["tip"],
        "marker_bodies": ["club"],
        "marker_offsets_m": [[0.0, 0, 1.0]],
    }
    return raw, NativeReplayCandidate.from_document(doc, names, digest)


def test_one_initial_state_and_absolute_sextic(package: tuple) -> None:
    raw, candidate = package
    clock = np.array([0.0, 0.4, 0.8])
    result = replay_candidate(raw, candidate, clock, model_factory=FakeEngine)
    np.testing.assert_allclose(result.markers_m[:, 0, 0], clock**8 / 56, atol=1e-11)
    assert result.candidate_sha256 == candidate.sha256
    assert not result.markers_m.flags.writeable


def test_initial_closure_failure_rejected(package: tuple) -> None:
    class Broken(FakeEngine):
        def closure_errors(self) -> tuple:
            return np.ones(6), np.zeros(6)

    with pytest.raises(ValueError, match="closure"):
        replay_candidate(*package, np.array([0.0, 0.8]), model_factory=Broken)


def test_partial_coverage_rejected(package: tuple) -> None:
    with pytest.raises(ValueError, match="coverage"):
        replay_candidate(*package, np.array([0.0, 0.7]), model_factory=FakeEngine)


def test_changed_model_bytes_rejected(package: tuple) -> None:
    raw, candidate = package
    with pytest.raises(ValueError, match="model_sha256"):
        replay_candidate(
            raw + b" ", candidate, np.array([0.0, 0.8]), model_factory=FakeEngine
        )


def test_window_uses_absolute_effort_and_exact_endpoint(package: tuple) -> None:
    from src.engines.physics_engines.pinocchio.python.native_replay import replay_window

    raw, candidate = package
    clock = np.array([0.6, 0.7, 0.8])
    initial = np.array([0.6**8 / 56, 0.0, 0.0, 0.6**7 / 7, 0.0, 0.0])
    result = replay_window(raw, candidate, clock, initial, model_factory=FakeEngine)
    np.testing.assert_array_equal(result.integration.time, clock)
    np.testing.assert_allclose(result.markers_m[:, 0, 0], clock**8 / 56, atol=1e-11)
    np.testing.assert_array_equal(result.integration.state[0], initial)
    assert not result.integration.time.flags.writeable


@pytest.mark.parametrize(
    "clock,state",
    [
        ([-0.1, 0.8], np.zeros(6)),
        ([0.6, 0.9], np.zeros(6)),
        ([0.6, 0.8], np.zeros(5)),
        ([0.6, 0.8], np.full(6, np.nan)),
    ],
)
def test_invalid_window_rejected(package: tuple, clock, state) -> None:
    from src.engines.physics_engines.pinocchio.python.native_replay import replay_window

    with pytest.raises(ValueError):
        replay_window(*package, np.array(clock), state, model_factory=FakeEngine)


def test_window_rejects_unassembled_node(package: tuple) -> None:
    from src.engines.physics_engines.pinocchio.python.native_replay import replay_window

    class BrokenNode(FakeEngine):
        def closure_errors(self) -> tuple:
            return np.ones(6), np.zeros(6)

    with pytest.raises(ValueError, match="closure"):
        replay_window(
            *package, np.array([0.6, 0.8]), np.zeros(6), model_factory=BrokenNode
        )
