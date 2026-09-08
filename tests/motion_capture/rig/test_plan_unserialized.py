"""A unit without a USB serial is bound as "the one without a serial"."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.motion_capture.rig.plan import (
    UNSERIALIZED_IDENTITY,
    CameraBinding,
    RigPlan,
    check_plan,
)
from src.motion_capture.rig.topology import CameraLocation

pytestmark = pytest.mark.unit


def _cam(serial: str | None, port: int, tail: str) -> CameraLocation:
    return CameraLocation(
        camera=f"USB\\VID_32E4&PID_5234&MI_00\\{tail}&0&0000",
        composite=None,
        serial=serial,
        identity=serial or f"path_{tail}-0-0000",
        root_hub="hub",
        root_port=port,
        host="host",
        hub_depth=0,
    )


PLAN = RigPlan(
    name="lab",
    cameras=(
        CameraBinding(view="a", serial="1"),
        CameraBinding(view="b", serial="2"),
        CameraBinding(view="c", unserialized=True),
    ),
)


def test_binding_needs_exactly_one_identity_kind() -> None:
    assert CameraBinding(view="c", unserialized=True).identity == UNSERIALIZED_IDENTITY
    with pytest.raises(ValidationError, match="serial, a port_path or unserialized"):
        CameraBinding(view="x")
    with pytest.raises(ValidationError, match="not both"):
        CameraBinding(view="x", serial="1", unserialized=True)


def test_unserialized_binding_resolves_when_exactly_one_unit_lacks_a_serial() -> None:
    check = check_plan(PLAN, [_cam("1", 6, "A"), _cam("2", 5, "B"), _cam(None, 4, "C")])
    assert check.ok
    assert check.matched["c"].endswith("C&0&0000")
    assert check.unplanned == ()


def test_unserialized_binding_follows_the_unit_to_another_port() -> None:
    moved = check_plan(PLAN, [_cam("1", 6, "A"), _cam("2", 5, "B"), _cam(None, 2, "Z")])
    assert moved.ok and moved.matched["c"].endswith("Z&0&0000")


def test_two_serial_less_units_are_ambiguous_not_guessed() -> None:
    check = check_plan(
        PLAN, [_cam("1", 6, "A"), _cam(None, 5, "B"), _cam(None, 4, "C")]
    )
    assert check.missing == ("b", "c")
    assert set(check.unplanned) == {"path_B-0-0000", "path_C-0-0000"}


def test_serial_less_unit_already_bound_by_port_path_is_not_double_counted() -> None:
    plan = RigPlan(
        name="lab",
        cameras=(
            CameraBinding(view="c", port_path="path_C-0-0000"),
            CameraBinding(view="d", unserialized=True),
        ),
    )
    check = check_plan(plan, [_cam(None, 4, "C"), _cam(None, 3, "D")])
    assert check.ok
    assert check.matched["c"].endswith("C&0&0000")
    assert check.matched["d"].endswith("D&0&0000")


def test_plan_round_trips_the_unserialized_flag(tmp_path) -> None:
    path = tmp_path / "plan.json"
    PLAN.save(path)
    assert RigPlan.load(path) == PLAN
