"""Trusted native worker requests validate and retain solver-relevant outputs."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python import (
    native_sensitivity_batch as batch,
)
from src.shared.python.motion_matching.native_window_executor import (
    NativeWindowExecutor,
)

pytestmark = pytest.mark.unit


def _request() -> batch.NativeSensitivityWindowRequest:
    names = ["x", "y", "z"]
    raw = json.dumps(
        {
            "coordinate_order": names,
            "joints": [{"parent": "world", "parent_to_base": np.eye(4).tolist()}],
        }
    ).encode()
    document = {
        "schema_version": 1,
        "coordinate_names": names,
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "coefficient_order": "highest-power-first",
        "time_basis": "absolute-seconds",
        "force_frame": "world",
        "duration_s": 0.8,
        "q0": [0.0] * 3,
        "qd0": [0.0] * 3,
        "coefficients": [[0.0] * 7 for _ in names],
        "marker_labels": ["marker"],
        "marker_bodies": ["body"],
        "marker_offsets_m": [[0.0, 0.0, 0.0]],
    }
    return batch.NativeSensitivityWindowRequest(
        raw, document, np.array([0.0, 0.4]), np.zeros(6), np.zeros((6, 2)), 0, 0.8
    )


def _fake_worker_result(_payload: bytes) -> batch.NativeSensitivityWindowResult:
    values = np.ones((2, 1, 3))
    return batch.NativeSensitivityWindowResult(
        values,
        np.ones((2, 6)),
        np.ones(6),
        np.ones((2, 1, 3, 2)),
        np.ones((2, 6, 2)),
        0.1,
        1,
        0.0,
    )


def test_trusted_worker_reconstructs_validated_candidate_and_returns_arrays(
    monkeypatch,
):
    request = _request()
    called = {}

    def fake_replay(raw, candidate, clock, **kwargs):
        called.update(raw=raw, candidate=candidate, clock=clock.copy(), kwargs=kwargs)
        return SimpleNamespace(
            replay=SimpleNamespace(
                markers_m=np.ones((2, 1, 3)),
                integration=SimpleNamespace(
                    state=np.arange(12, dtype=float).reshape(2, 6)
                ),
            ),
            marker_jacobian=np.ones((2, 1, 3, 11)),
            state_jacobian=np.ones((2, 6, 11)),
            sensitivity_elapsed_s=1.5,
            sensitivity_evaluations=17,
            primal_marker_max_abs_difference_m=2e-9,
        )

    monkeypatch.setattr(batch, "replay_marker_sensitivities", fake_replay)
    result = batch.evaluate_trusted_native_sensitivity_request(
        batch.serialize_trusted_native_sensitivity_request(request)
    )
    assert called["candidate"].document == request.candidate_document
    np.testing.assert_array_equal(called["clock"], request.time_s)
    assert called["kwargs"]["first_control"] == 0
    assert result.endpoint_state.tolist() == [6, 7, 8, 9, 10, 11]
    assert result.states.shape == (2, 6)
    assert result.marker_jacobian.shape == (2, 1, 3, 11)
    assert not result.markers_m.flags.writeable


def test_request_contracts_and_wrong_payload_are_rejected():
    request = _request()
    with pytest.raises(ValueError, match="strictly increasing"):
        batch.NativeSensitivityWindowRequest(
            request.model_bytes,
            request.candidate_document,
            np.array([0.4, 0.0]),
            request.initial_state,
            request.initial_sensitivity,
            0,
            0.8,
        )
    with pytest.raises(ValueError, match="payload type"):
        batch.evaluate_trusted_native_sensitivity_request(b"not a request")
    with pytest.raises(TypeError, match="immutable bytes"):
        batch.evaluate_trusted_native_sensitivity_request(bytearray(b"x"))


def test_parent_adapter_preserves_order_and_returns_solver_arrays():
    request = _request()
    with NativeWindowExecutor(_fake_worker_result) as executor:
        adapter = batch.NativeSensitivityBatchAdapter(
            lambda theta, clock, state: request, executor
        )
        output = adapter(
            [
                (np.zeros(1), np.array([0.0, 0.4]), None),
                (np.zeros(1), np.array([0.4, 0.8]), np.zeros(6)),
            ]
        )
    assert len(output) == 2
    assert output[0][0].shape == (2, 1, 3)
    assert output[1][1].shape == (6,)
