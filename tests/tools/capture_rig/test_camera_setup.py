"""Player camera plans use the rig's actual schema and stable identities."""

from pathlib import Path
from threading import Event

import pytest

from src.motion_capture.rig.plan import CaptureMode, RigPlan
from src.motion_capture.rig.topology import CameraLocation
from src.tools.capture_rig import camera_setup as setup

pytestmark = pytest.mark.unit


def camera(identity: str, *, serial: bool = True) -> CameraLocation:
    return CameraLocation(
        camera=f"USB\\{identity}",
        composite=None,
        serial=identity if serial else None,
        identity=identity,
        root_hub=f"hub-{identity}",
        root_port=1,
        host="host",
        hub_depth=1,
        index=0,
    )


def test_plan_round_trip_preserves_stable_identity_and_existing_file(
    tmp_path: Path,
) -> None:
    bindings = [
        setup.bind_camera("face_on", camera("serial-1"), CaptureMode()),
        setup.bind_camera("down_line", camera("path_2", serial=False), CaptureMode()),
    ]
    plan = setup.create_plan("Lesson", bindings, "Keep camera positions fixed")
    first = setup.save_revision(plan, tmp_path)
    original = first.read_bytes()
    second = setup.save_revision(plan, tmp_path)
    assert first != second and first.read_bytes() == original
    assert RigPlan.load(first) == plan
    assert setup.load_plan(second) == plan
    assert plan.cameras[0].serial == "serial-1"
    assert plan.cameras[1].port_path == "path_2"
    assert setup.connection_status(
        plan, [camera("serial-1"), camera("path_2", serial=False)]
    )[0]


@pytest.mark.parametrize("view", ["../video", "a/b", "", "white space", "a\\b"])
def test_invalid_view_names_are_rejected(view: str) -> None:
    with pytest.raises(ValueError, match="view"):
        setup.create_plan(
            "Lesson", [setup.bind_camera(view, camera("one"), CaptureMode())]
        )


def test_duplicate_devices_and_missing_connections_are_actionable() -> None:
    binding = setup.bind_camera("face_on", camera("one"), CaptureMode())
    other = setup.bind_camera("down_line", camera("one"), CaptureMode())
    with pytest.raises(ValueError, match="duplicate"):
        setup.create_plan("Lesson", [binding, other])
    ready, reason = setup.connection_status(setup.create_plan("Lesson", [binding]), [])
    assert not ready and "face_on" in reason


def test_cancelled_discovery_does_not_touch_hardware(monkeypatch) -> None:
    cancelled = Event()
    cancelled.set()
    monkeypatch.setattr(
        setup, "query_topology", lambda **kwargs: pytest.fail("hardware queried")
    )
    assert setup.discover_cameras(cancelled) == []
