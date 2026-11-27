"""Shaft shape at impact in the club frame and through the sweep JSON (#15150 C5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.club_fitting.bend_trace import bend_trace
from src.shared.python.club_fitting.grip_driven_shaft import HeadGeometry, ShaftRun
from src.shared.python.club_fitting.shaft_shape import (
    ShaftShape,
    shaft_shape,
    shaft_shape_from_run,
)

pytestmark = pytest.mark.unit

HEAD = HeadGeometry(np.zeros(3), np.array([0.0, 0.0, 1.0]))
# Club +x points down in the world (droop), the face normal club +z is lead.
ROT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
Y = np.array([-1.0, -0.7, -0.4, -0.1, 0.0])
H = 0.03


def _points(lead_m: float, droop_m: float) -> np.ndarray:
    """Grip-to-tip chain points bent quadratically, tip at ``(lead, droop)``."""
    f = ((Y - Y[0]) / (Y[-1] - Y[0])) ** 2
    return np.column_stack([droop_m * f, Y, H + lead_m * f])


def _run(lead_m: float, droop_m: float) -> ShaftRun:
    """A three-record run ending at impact with matching tip deflection."""
    times = np.array([-0.02, -0.01, 0.0])
    defl = np.tile(ROT @ np.array([droop_m, 0.0, lead_m]), (3, 1))
    rot, vec, zero = np.tile(ROT, (3, 1, 1)), np.zeros((3, 3)), np.zeros(3)
    return ShaftRun(
        times, np.zeros((3, 8)), defl, vec, vec, vec, vec, rot, rot,
        zero, zero, zero, np.array([]), _points(lead_m, droop_m),
    )  # fmt: skip


def test_straight_shaft_has_zero_deflection() -> None:
    shape = shaft_shape(_points(0.0, 0.0), HEAD, ROT)
    assert shape.points_m.shape == (len(Y), 3)
    np.testing.assert_allclose(shape.lead_m, 0.0, atol=1e-15)
    np.testing.assert_allclose(shape.droop_m, 0.0, atol=1e-15)
    np.testing.assert_allclose(shape.along_m, Y - Y[0])


def test_known_bend_gives_lead_and_droop_at_the_tip() -> None:
    """Lead + with the head ahead, droop + toe-down (the #15777 convention)."""
    shape = shaft_shape(_points(0.05, 0.02), HEAD, ROT)
    assert shape.lead_m[-1] == pytest.approx(0.05)
    assert shape.droop_m[-1] == pytest.approx(0.02)
    assert shape.lead_m[0] == pytest.approx(0.0)
    run = _run(0.05, 0.02)
    lead_mm, droop_mm = bend_trace(run, HEAD, 0.0).at_impact()
    from_run = shaft_shape_from_run(run, HEAD, 0.0)
    assert from_run is not None
    assert from_run.lead_m[-1] * 1e3 == pytest.approx(lead_mm)
    assert from_run.droop_m[-1] * 1e3 == pytest.approx(droop_mm)


def test_run_without_chain_points_has_no_shape() -> None:
    run = _run(0.0, 0.0)
    bare = ShaftRun(*(getattr(run, f) for f in run.__dataclass_fields__ if
                      f != "impact_chain_club_m"))  # fmt: skip
    assert shaft_shape_from_run(bare, HEAD, 0.0) is None


def test_shape_contracts() -> None:
    with pytest.raises(ValueError, match="points_m"):
        ShaftShape(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="points_m"):
        ShaftShape(np.array([[0.0, np.nan, 0.0], [1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="missing"):
        ShaftShape.from_dict({})
    with pytest.raises(ValueError, match="chain points"):
        shaft_shape(np.zeros((1, 3)), HEAD, ROT)


def test_shape_round_trips_through_the_sweep_json(tmp_path) -> None:
    from src.shared.python.club_fitting.__main__ import write_outputs
    from src.shared.python.club_fitting.flex_sweep import _row
    from src.shared.python.club_fitting.shaft_report import load_sweep_result

    row = _row("Stiff", 280.0, _run(0.05, 0.02), 0.0, None, HEAD)
    assert row.shape is not None and "shape" not in row.record()
    json_path, _ = write_outputs([row], tmp_path)
    loaded = load_sweep_result(json_path).shapes["Stiff"]
    np.testing.assert_allclose(loaded.points_m, row.shape.points_m)
    data = json.loads(json_path.read_text())
    del data["shapes"]
    json_path.write_text(json.dumps(data))
    assert load_sweep_result(json_path).shapes == {}
