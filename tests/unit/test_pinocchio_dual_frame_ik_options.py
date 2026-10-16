"""Characterization of ``solve_dual_frame_ik`` with a numpy kinematics fake.

Real-pinocchio convergence lives in
``tests/heavy_integration/test_pinocchio_diff_ik.py``; this pins the option
handling and iteration contract without the robotics library (issue #12845).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.pinocchio_golf import diff_ik

JAC_A = np.diag([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])[:, :2]
JAC_B = np.array([[0.0, 1.0], [1.0, 0.0]] + [[0.0, 0.0]] * 4)


class _Model:
    nq = 2

    def existFrame(self, name: str) -> bool:  # noqa: N802
        return name in {"a", "b"}

    def getFrameId(self, name: str) -> int:  # noqa: N802
        return {"a": 0, "b": 1}[name]


@pytest.fixture
def fake_pin(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"fk": 0}

    def fk(model, data, q):
        counts["fk"] += 1
        data.oMf[0] = JAC_A @ np.asarray(q)
        data.oMf[1] = JAC_B @ np.asarray(q)

    fake = SimpleNamespace(
        LOCAL=0,
        forwardKinematics=fk,
        updateFramePlacement=lambda *a: None,
        computeJointJacobians=lambda *a: None,
        getFrameJacobian=lambda m, d, fid, ref: (JAC_A, JAC_B)[fid],
        integrate=lambda m, q, dq: q + dq,
    )
    monkeypatch.setattr(diff_ik, "pin", fake)
    monkeypatch.setattr(diff_ik, "PINOCCHIO_AVAILABLE", True)
    monkeypatch.setattr(diff_ik, "se3_log6", lambda target, cur: target - cur)
    return counts


def _data() -> SimpleNamespace:
    return SimpleNamespace(oMf={})


def _targets() -> tuple[np.ndarray, np.ndarray]:
    q_true = np.array([0.3, -0.2])
    return JAC_A @ q_true, JAC_B @ q_true


@pytest.mark.unit
def test_converges_with_defaults(fake_pin) -> None:
    ta, tb = _targets()
    q, ok = diff_ik.solve_dual_frame_ik(
        _Model(), _data(), "a", ta, "b", tb, np.zeros(2)
    )
    assert ok is True
    assert q == pytest.approx([0.3, -0.2], abs=1e-3)
    assert q.dtype == np.float64


@pytest.mark.unit
def test_max_iters_zero_does_not_converge(fake_pin) -> None:
    ta, tb = _targets()
    q, ok = diff_ik.solve_dual_frame_ik(
        _Model(), _data(), "a", ta, "b", tb, np.zeros(2), max_iters=0
    )
    assert ok is False
    assert fake_pin["fk"] == 0
    assert q.tolist() == [0.0, 0.0]


@pytest.mark.unit
def test_max_iters_bounds_iterations(fake_pin) -> None:
    ta, tb = _targets()
    _, ok = diff_ik.solve_dual_frame_ik(
        _Model(), _data(), "a", ta, "b", tb, np.zeros(2), max_iters=1, tol=1e-12
    )
    assert ok is False
    assert fake_pin["fk"] == 1


@pytest.mark.unit
def test_all_keyword_options_accepted(fake_pin) -> None:
    ta, tb = _targets()
    q, ok = diff_ik.solve_dual_frame_ik(
        _Model(),
        _data(),
        "a",
        ta,
        "b",
        tb,
        np.zeros(2),
        max_iters=400,
        damping=1e-5,
        tol=1e-5,
        weight_a=2.0,
        weight_b=0.5,
    )
    assert ok
    assert q == pytest.approx([0.3, -0.2], abs=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["weight_a", "weight_b"])
def test_non_positive_weights_rejected(fake_pin, name: str) -> None:
    ta, tb = _targets()
    with pytest.raises(ValueError, match="frame weights must be positive"):
        diff_ik.solve_dual_frame_ik(
            _Model(), _data(), "a", ta, "b", tb, np.zeros(2), **{name: 0.0}
        )


@pytest.mark.unit
def test_unknown_frame_and_bad_seed_rejected(fake_pin) -> None:
    ta, tb = _targets()
    with pytest.raises(ValueError, match="frame 'zz' not found"):
        diff_ik.solve_dual_frame_ik(_Model(), _data(), "zz", ta, "b", tb, np.zeros(2))
    with pytest.raises(ValueError, match="frame 'zz' not found"):
        diff_ik.solve_dual_frame_ik(_Model(), _data(), "a", ta, "zz", tb, np.zeros(2))
    with pytest.raises(ValueError, match="q0 has shape"):
        diff_ik.solve_dual_frame_ik(_Model(), _data(), "a", ta, "b", tb, np.zeros(3))


@pytest.mark.unit
def test_unknown_option_is_type_error(fake_pin) -> None:
    ta, tb = _targets()
    with pytest.raises(TypeError):
        diff_ik.solve_dual_frame_ik(
            _Model(), _data(), "a", ta, "b", tb, np.zeros(2), bogus=1
        )


@pytest.mark.unit
def test_requires_pinocchio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diff_ik, "PINOCCHIO_AVAILABLE", False)
    with pytest.raises(ImportError, match="pinocchio is required"):
        diff_ik.solve_dual_frame_ik(None, None, "a", None, "b", None, np.zeros(2))
