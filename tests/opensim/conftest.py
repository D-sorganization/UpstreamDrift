"""Native OpenSim fixtures shared by geometry and registration tests."""

from pathlib import Path

import pytest


@pytest.fixture
def native_pin(tmp_path: Path) -> Path:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    body = osim.Body("segment", 1.0, osim.Vec3(0), osim.Inertia(0.02))
    model.addBody(body)
    joint = osim.PinJoint(
        "pin",
        model.getGround(),
        osim.Vec3(1, 0, 0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    joint.updCoordinate().setName("angle")
    joint.updCoordinate().setRangeMin(-1.5)
    joint.updCoordinate().setRangeMax(1.5)
    model.addJoint(joint)
    frame = osim.PhysicalOffsetFrame(
        "skin_frame", body, osim.Transform(osim.Vec3(0.2, 0, 0))
    )
    body.addComponent(frame)
    model.finalizeConnections()
    path = tmp_path / "pin.osim"
    model.printToXML(str(path))
    return path
