"""Engine-free tests of the same-input integrator, bundle and scoring (#11607)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    ROOT_COORDINATES,
    SCHEMA,
    InputBundle,
    growth_rate,
    integrate,
    open_loop,
    score_replay,
    zoh_rk4_step,
)

pytestmark = pytest.mark.unit

ORDER = (*ROOT_COORDINATES, "Joint")
SPEC = json.dumps({"coordinate_order": list(ORDER)}).encode()


class _Oscillator:
    """Unit oscillators a = -q + tau; no closure; identity frame per coordinate."""

    coordinate_order = ORDER

    def acceleration(self, q, v, tau):
        return -q + tau

    def closure_pose_residual(self, q):
        return np.zeros(1)

    def closure_rate_matrix(self, q):
        return np.zeros((1, q.size))

    def kinematic_frames(self, q):
        pose = np.eye(4)
        pose[0, 3] = q[-1]
        return {"tip": pose}


def _bundle(steps: int = 20, dt_s: float = 0.01) -> InputBundle:
    q0, v0 = np.zeros(7), np.zeros(7)
    q0[-1] = 1.0
    efforts = np.zeros((steps, 7))
    efforts[:, -1] = 0.5
    rollout = open_loop(_Oscillator(), q0, v0, efforts, dt_s=dt_s, project=False)
    return InputBundle(
        spec_bytes=SPEC,
        coordinate_order=ORDER,
        dt_s=dt_s,
        q0=q0,
        v0=v0,
        efforts=efforts,
        reference_q=rollout.q,
        reference_v=rollout.v,
        reference_engine="toy",
        provenance={"source": "unit test"},
    )


def test_rk4_step_is_fourth_order() -> None:
    plant, q0, v0, tau = _Oscillator(), np.ones(7), np.zeros(7), np.zeros(7)
    errors = []
    for dt in (0.1, 0.05):
        q, v = q0, v0
        for _ in range(round(1.0 / dt)):
            q, v = zoh_rk4_step(plant, q, v, tau, dt)
        errors.append(abs(q[-1] - np.cos(1.0)))
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.1)


def test_integrate_holds_effort_and_zeroes_root() -> None:
    calls = []

    def source(k, t, q, v):
        calls.append(k)
        return np.ones(7)

    rollout = integrate(
        _Oscillator(),
        np.zeros(7),
        np.zeros(7),
        source,
        steps=3,
        dt_s=0.1,
        project=False,
    )
    assert calls == [0, 1, 2]
    assert np.all(rollout.efforts[:, :6] == 0.0)
    assert np.all(rollout.q[:, :6] == 0.0)
    assert rollout.q.shape == (4, 7)


def test_substeps_match_a_finer_step() -> None:
    plant, q0, v0 = _Oscillator(), np.ones(7), np.zeros(7)
    efforts = np.zeros((10, 7))
    coarse = open_loop(plant, q0, v0, efforts, dt_s=0.1, substeps=4, project=False)
    fine = open_loop(
        plant, q0, v0, np.zeros((40, 7)), dt_s=0.025, substeps=1, project=False
    )
    np.testing.assert_allclose(coarse.q[-1], fine.q[-1], atol=1e-15)


def test_open_loop_rejects_root_efforts_and_bad_shapes() -> None:
    efforts = np.zeros((2, 7))
    efforts[0, 0] = 1.0
    with pytest.raises(ValueError, match="root"):
        open_loop(_Oscillator(), np.zeros(7), np.zeros(7), efforts, dt_s=0.1)
    with pytest.raises(ValueError):
        open_loop(_Oscillator(), np.zeros(7), np.zeros(7), np.zeros((2, 3)), dt_s=0.1)
    with pytest.raises(ValueError):
        integrate(_Oscillator(), np.zeros(7), np.zeros(7), None, steps=0, dt_s=0.1)


def test_bundle_round_trip(tmp_path: Path) -> None:
    bundle = _bundle()
    path = tmp_path / "bundle.npz"
    bundle.save(path)
    loaded = InputBundle.load(path)
    assert loaded.manifest() == bundle.manifest()
    assert loaded.manifest()["schema"] == SCHEMA
    assert loaded.spec_bytes == SPEC
    np.testing.assert_array_equal(loaded.efforts, bundle.efforts)
    np.testing.assert_array_equal(loaded.reference_q, bundle.reference_q)


def test_bundle_rejects_tampered_spec(tmp_path: Path) -> None:
    bundle = _bundle()
    path = tmp_path / "bundle.npz"
    bundle.save(path)
    with np.load(path) as data:
        arrays = dict(data)
    arrays["spec"] = np.frombuffer(SPEC.replace(b"Joint", b"Jxint"), dtype=np.uint8)
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        InputBundle.load(path)


def test_bundle_validates_shapes() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError):
        InputBundle(
            spec_bytes=SPEC,
            coordinate_order=ORDER,
            dt_s=bundle.dt_s,
            q0=bundle.q0,
            v0=bundle.v0,
            efforts=bundle.efforts,
            reference_q=bundle.reference_q[:-1],
            reference_v=bundle.reference_v,
            reference_engine="toy",
        )


def test_growth_rate_recovers_an_exponent() -> None:
    t = np.linspace(0.0, 1.0, 50)
    assert growth_rate(t, 1e-14 * np.exp(25.0 * t), 1e-12, 1e-3) == pytest.approx(25.0)
    assert np.isnan(growth_rate(t, np.zeros(50), 1e-12, 1e-3))


def test_score_replay_of_the_reference_is_exact() -> None:
    bundle = _bundle()
    rollout = open_loop(
        _Oscillator(),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
        project=False,
    )
    score = score_replay(_Oscillator(), bundle, rollout, frame_stride=5)
    assert score.coordinate_error.max() == 0.0
    assert score.frame_error_m.max() == 0.0
    assert score.horizon_s == pytest.approx(bundle.steps * bundle.dt_s)
    summary = score.summary()
    assert summary["growth_rate_per_s"] is None  # no growth band: strict JSON null
    json.dumps(summary, allow_nan=False)


def test_stop_on_failure_returns_the_states_reached() -> None:
    class _Blowup(_Oscillator):
        def acceleration(self, q, v, tau):
            if q[-1] > 1.5:
                raise FloatingPointError("nonfinite")
            return np.full_like(q, 0.0) + tau

    efforts = np.zeros((10, 7))
    efforts[:, -1] = 100.0
    plant, q0 = _Blowup(), np.zeros(7)
    with pytest.raises(FloatingPointError):
        open_loop(plant, q0, np.zeros(7), efforts, dt_s=0.1, project=False)
    rollout = open_loop(
        plant, q0, np.zeros(7), efforts, dt_s=0.1, project=False, stop_on_failure=True
    )
    assert rollout.failure is not None and "FloatingPointError" in rollout.failure
    assert rollout.q.shape[0] == rollout.efforts.shape[0] + 1 < 11
