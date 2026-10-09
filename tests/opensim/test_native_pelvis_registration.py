"""Native gauge reference and explicit donor direction evidence."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
    apply_frozen_registration,
    pelvis_coordinate_gauge,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TourCapture,
)

pytest_plugins = ("tests.opensim.test_native_marker_geometry",)
pytestmark = pytest.mark.unit


def test_native_reference_gauge_recovers_independently_generated_marker_motion(
    native_pin: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_marker_geometry import (
        NativeMarkerGeometry,
    )

    native = NativeMarkerGeometry(native_pin, ("/jointset/pin/angle",))
    labels = MARKER_SEGMENTS["pelvis"]
    offsets = [[0.05, 0, -0.1], [0.05, 0, 0.1], [-0.05, 0, -0.1], [-0.05, 0, 0.1]]
    bindings = {
        label: ("/bodyset/segment", offset)
        for label, offset in zip(labels, offsets, strict=True)
    }
    actual = np.array(
        [native.marker_positions(np.array([q]), bindings) for q in [0, 0.2, 0.4]]
    )
    raw_rotation = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    observations = actual @ raw_rotation.T + [3, 0.5, 4]
    capture = TourCapture(
        np.array([0, 0.01, 0.02]), labels, observations, np.ones((3, 4), bool), "a" * 64
    )
    pose = native.frame_poses(bindings, native.initial_coordinates)["/bodyset/segment"]
    frozen = pelvis_coordinate_gauge(
        capture,
        pose,
        target_left_axis_local=np.array([0, 0, -1.0]),
        target_geometry_sha256=native.identity_sha256,
        training_frame=0,
    )
    registered = apply_frozen_registration(
        capture, frozen, source_frame=frozen.transform.source_frame
    )
    np.testing.assert_allclose(registered.capture.points_m, actual, atol=1e-12)
    assert not frozen.anatomically_qualified


@pytest.mark.parametrize("right_frame", ["/bodyset/pelvis", "/bodyset/other"])
def test_source_marker_direction_is_explicit_and_same_body(
    tmp_path: Path, right_frame: str
) -> None:
    from scripts.diagnostics.native_pelvis_registration import source_lateral_axis

    model = tmp_path / "markers.osim"
    model.write_text(f"""<OpenSimDocument><Model><MarkerSet><objects>
    <Marker name="lasi"><socket_parent_frame>/bodyset/pelvis</socket_parent_frame><location>0 0 -0.1</location></Marker>
    <Marker name="rasi"><socket_parent_frame>{right_frame}</socket_parent_frame><location>0 0 0.1</location></Marker>
    </objects></MarkerSet></Model></OpenSimDocument>""")
    if right_frame.endswith("other"):
        with pytest.raises(ValueError, match="body"):
            source_lateral_axis(model)
    else:
        np.testing.assert_array_equal(source_lateral_axis(model), [0, 0, -1])
