"""Weld-consistent tracked reference (GCV-20, #11767).

The low-passed tracked reference opens the dual-grip weld by up to 16 mm on
the driver capture; the KKT replay then spends trail-arm effort closing it
during the release. Projecting each tracked sample back onto the weld over
the trail-arm coordinates removes that inconsistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.constants import (
    TRAIL_ARM_WELD_COORDINATES,
)
from src.shared.python.motion_matching.pipeline.weld_projection import (
    project_onto_closure,
    weld_consistent_track,
)

TARGET = np.array([1.2, 0.7])


def _two_link(q: np.ndarray) -> np.ndarray:
    """Planar two-link tip (unit links on columns 1 and 2) minus a target."""
    a, b = q[1], q[1] + q[2]
    return np.array([np.cos(a) + np.cos(b), np.sin(a) + np.sin(b)]) - TARGET


def test_projection_closes_the_residual_on_the_given_columns_only() -> None:
    q = np.array([5.0, 0.3, 0.4, -2.0])
    out = project_onto_closure(q, _two_link, [1, 2])
    assert np.linalg.norm(_two_link(out)) < 1e-7
    assert out[0] == q[0] and out[3] == q[3]


def test_a_consistent_sample_is_returned_unchanged() -> None:
    q = project_onto_closure(np.array([0.0, 0.3, 0.4, 0.0]), _two_link, [1, 2])
    assert np.array_equal(project_onto_closure(q, _two_link, [1, 2]), q)


def test_projection_does_not_mutate_its_input() -> None:
    q = np.array([0.0, 0.3, 0.4, 0.0])
    keep = q.copy()
    project_onto_closure(q, _two_link, [1, 2])
    assert np.array_equal(q, keep)


@pytest.mark.parametrize(
    ("q", "columns"),
    [
        (np.array([0.0, np.nan, 0.4]), [1, 2]),
        (np.zeros((2, 3)), [1, 2]),
        (np.zeros(3), []),
        (np.zeros(3), [1, 3]),
        (np.zeros(3), [1, 1]),
    ],
)
def test_projection_contracts(q: np.ndarray, columns: list[int]) -> None:
    with pytest.raises(ValueError):
        project_onto_closure(q, _two_link, columns)


def test_a_projection_that_does_not_converge_is_reported() -> None:
    def unreachable(q: np.ndarray) -> np.ndarray:
        return np.array([np.cos(q[1]) + 3.0])

    with pytest.raises(ValueError, match="did not close"):
        project_onto_closure(np.zeros(2), unreachable, [1])


class _FakeKin:
    """Body 'hand' at the planar two-link tip, 'club' fixed at the target."""

    def __init__(self, order: list[str]) -> None:
        self.coordinate_order = tuple(order)
        self.cols = [order.index(TRAIL_ARM_WELD_COORDINATES[k]) for k in (0, 1)]

    def body_poses(self, q, bodies):  # noqa: ANN001 - test double
        a, b = q[self.cols[0]], q[self.cols[0]] + q[self.cols[1]]
        tip = np.array([np.cos(a) + np.cos(b), np.sin(a) + np.sin(b), 0.0])
        return {
            "hand": (np.eye(3), tip),
            "club": (np.eye(3), np.r_[TARGET, 0.0]),
        }


def _spec() -> dict:
    return {
        "closure": {
            "body_a": "hand",
            "body_b": "club",
            "placement_a": np.eye(4).tolist(),
            "placement_b": np.eye(4).tolist(),
        }
    }


def test_weld_consistent_track_closes_every_sample_and_reports_it() -> None:
    order = ["root", *TRAIL_ARM_WELD_COORDINATES, "LWInputX"]
    kin = _FakeKin(order)
    q = np.zeros((5, len(order)))
    q[:, kin.cols[0]] = np.linspace(0.2, 0.4, 5)
    q[:, kin.cols[1]] = 0.5
    out, report = weld_consistent_track(kin, _spec(), q)
    assert out.shape == q.shape
    assert report["applied"] is True
    assert report["max_position_mm_before"] > 1.0
    assert report["max_position_mm_after"] < 1e-3
    assert report["coordinates"] == list(TRAIL_ARM_WELD_COORDINATES)
    untouched = [i for i, n in enumerate(order) if n not in TRAIL_ARM_WELD_COORDINATES]
    assert np.array_equal(out[:, untouched], q[:, untouched])


def test_weld_consistent_track_skips_a_model_without_the_weld() -> None:
    order = ["root", *TRAIL_ARM_WELD_COORDINATES]
    q = np.zeros((3, len(order)))
    out, report = weld_consistent_track(_FakeKin(order), {}, q)
    assert out is q
    assert report == {"applied": False, "source": "no closure in spec"}


def test_weld_consistent_track_skips_a_model_without_the_trail_arm() -> None:
    order = ["root", *TRAIL_ARM_WELD_COORDINATES]
    kin = _FakeKin(order)
    kin.coordinate_order = ("root", "x", "y")
    q = np.zeros((3, 3))
    out, report = weld_consistent_track(kin, _spec(), q)
    assert out is q
    assert report["applied"] is False
    assert report["source"].startswith("missing trail-arm coordinates")


def test_trail_arm_coordinates_are_the_right_scapula_shoulder_elbow_wrist() -> None:
    assert TRAIL_ARM_WELD_COORDINATES == (
        "RScapInputX",
        "RScapInputY",
        "RSInputX",
        "RSInputY",
        "RSInputZ",
        "REInput",
        "RFInput",
        "RWInputX",
        "RWInputY",
    )
