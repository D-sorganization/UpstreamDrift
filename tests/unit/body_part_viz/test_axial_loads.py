"""Axial reaction projection and frame capability contracts."""

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadFrame,
    axial_force_from_proximal_reaction,
    read_axial_load_frame,
)

pytestmark = pytest.mark.unit


def test_hanging_inverted_and_rotated_reactions():
    assert axial_force_from_proximal_reaction((0, 10), (0, 0), (0, -1)) == 10
    assert axial_force_from_proximal_reaction((0, 10), (0, 0), (0, 1)) == -10
    assert axial_force_from_proximal_reaction((10, 0, 0), (1, 2, 3), (0, 2, 3)) == 10
    assert axial_force_from_proximal_reaction((10, 0), (0, 0), (0, 1)) == 0


def test_projection_does_not_mutate_and_rejects_bad_geometry():
    force = np.array([0.0, 10.0])
    axial_force_from_proximal_reaction(force, (0, 0), (0, -1))
    np.testing.assert_array_equal(force, [0, 10])
    for end in [(0, 0), (0, float("nan")), (0, 0, 1)]:
        with pytest.raises(ValueError):
            axial_force_from_proximal_reaction(force, (0, 0), end)


def test_frame_is_immutable_and_json_safe():
    values = {"link": 10.0, "gap": float("nan")}
    frame = AxialLoadFrame(1.0, values, "parent-on-child at proximal section")
    values["link"] = -10
    assert frame.to_dict()["values_n"] == {"link": 10.0, "gap": None}
    assert frame.to_dict()["sign_convention"] == "tension-positive"
    with pytest.raises(ValueError):
        AxialLoadFrame(float("nan"), {}, "source")


def test_optional_provider_and_timestamp_validation():
    class Provider:
        def get_segment_axial_loads(self):
            return AxialLoadFrame(1.0, {"link": 5.0}, "fixture")

    assert read_axial_load_frame(Provider(), 1.0)["values_n"] == {"link": 5.0}
    assert read_axial_load_frame(Provider(), 2.0) is None
    assert read_axial_load_frame(object(), 1.0) is None
