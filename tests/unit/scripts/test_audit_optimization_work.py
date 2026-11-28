"""Analytic work fixtures for total applied joint torque (#15493)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import ArrayLike

from scripts.audit_optimization_work import audit_joint_work

pytestmark = pytest.mark.unit


def test_braking_does_not_hide_positive_work() -> None:
    result = audit_joint_work([0, 2], [[2, -2], [-2, 2]], [[1, 1], [1, 1]])
    np.testing.assert_allclose(result.signed_work_j, [0, 0])
    np.testing.assert_allclose(result.positive_work_j, [1, 1])
    np.testing.assert_allclose(result.negative_work_j, [-1, -1])
    assert result.total_positive_work_j == pytest.approx(2)


def test_nonuniform_time_and_unequal_crossing() -> None:
    result = audit_joint_work([0, 4, 5], [[3], [-1], [-1]], [[1], [1], [1]])
    np.testing.assert_allclose(result.signed_work_j, [3])
    np.testing.assert_allclose(result.positive_work_j, [4.5])
    np.testing.assert_allclose(result.negative_work_j, [-1.5])


def test_velocity_sign_and_total_applied_torque() -> None:
    # Feedforward is zero; all applied power here could come from feedback.
    result = audit_joint_work([0, 2], [[3, 4], [3, 4]], [[2, -1], [2, -1]])
    np.testing.assert_allclose(result.signed_work_j, [12, -8])
    np.testing.assert_allclose(result.positive_work_j, [12, 0])
    np.testing.assert_allclose(result.negative_work_j, [0, -8])


def test_zero_power_and_zero_endpoint() -> None:
    result = audit_joint_work([0, 1, 2], [[0, 0], [2, 0], [0, 0]], np.ones((3, 2)))
    np.testing.assert_allclose(result.positive_work_j, [2, 0])
    np.testing.assert_allclose(result.negative_work_j, [0, 0])


@pytest.mark.parametrize(
    ("time", "torque", "velocity"),
    [
        ([0], [[1]], [[1]]),
        ([0, 0], [[1], [1]], [[1], [1]]),
        ([1, 0], [[1], [1]], [[1], [1]]),
        ([0, np.inf], [[1], [1]], [[1], [1]]),
        ([[0, 1]], [[1], [1]], [[1], [1]]),
        ([0, 1], [1, 1], [[1], [1]]),
        ([0, 1], [[1], [1]], [[1, 1], [1, 1]]),
        ([0, 1], [[], []], [[], []]),
        ([0, 1], [[np.nan], [1]], [[1], [1]]),
        ([0, 1], [[1], [1]], [[1], [np.inf]]),
    ],
)
def test_rejects_invalid_inputs(
    time: ArrayLike, torque: ArrayLike, velocity: ArrayLike
) -> None:
    with pytest.raises(ValueError):
        audit_joint_work(time, torque, velocity)


def test_rejects_overflow_in_derived_power() -> None:
    with pytest.raises(ValueError, match="finite"):
        audit_joint_work([0, 1], [[1e308], [1e308]], [[2], [2]])


def test_work_is_additive_after_linear_power_refinement() -> None:
    coarse = audit_joint_work([0, 3], [[4], [-2]], [[1], [1]])
    fine = audit_joint_work([0, 1, 2, 3], [[4], [2], [0], [-2]], np.ones((4, 1)))
    np.testing.assert_allclose(coarse.positive_work_j, fine.positive_work_j)
    np.testing.assert_allclose(coarse.negative_work_j, fine.negative_work_j)
    np.testing.assert_allclose(
        fine.signed_work_j, fine.positive_work_j + fine.negative_work_j
    )


def test_rejects_overflow_in_work() -> None:
    with pytest.raises(ValueError, match="finite"):
        audit_joint_work([0, 1e308], [[4], [4]], [[1], [1]])


def test_cli_uses_applied_torque_and_rollout_velocity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import audit_optimization_work as cli

    def load(path: Path) -> SimpleNamespace:
        assert path == Path("example-bundle")
        return SimpleNamespace(
            solution_id="example",
            joint_order=("wrist",),
            arrays={
                "time": [0, 2],
                "tau_applied": [[3], [3]],
                "qd_rollout": [[2], [2]],
                "tau_cmd": [[0], [0]],
                "tau_ff_ref": [[0], [0]],
            },
        )

    monkeypatch.setattr(cli, "load_solution_bundle", load)
    monkeypatch.setattr("sys.argv", ["audit", "example-bundle"])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["solution_id"] == "example"
    assert report["joint_order"] == ["wrist"]
    assert report["positive_work_j"] == [12]
    assert report["total_positive_work_j"] == 12
    assert report["qualification"] == "sampled_work_only"
