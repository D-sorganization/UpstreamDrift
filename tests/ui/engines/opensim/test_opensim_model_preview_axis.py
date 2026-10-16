"""OpenSim preview stands the golfer upright and capsules only missing meshes (#12925)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import geometry_resolver, model_preview

Z_UP_GRAVITY = (0.0, 0.0, -9.81)
Y_UP_GRAVITY = (0.0, -9.81, 0.0)


@pytest.mark.unit
def test_up_axis_is_read_from_gravity() -> None:
    assert model_preview.up_axis_from_gravity(Z_UP_GRAVITY) == 2
    assert model_preview.up_axis_from_gravity(Y_UP_GRAVITY) == 1
    with pytest.raises(ValueError):
        model_preview.up_axis_from_gravity((0.0, 0.0, 0.0))


@pytest.mark.unit
def test_y_up_points_are_rotated_to_z_up() -> None:
    out = model_preview.to_plot_frame(np.array([[0.1, 1.7, 0.0]]), Y_UP_GRAVITY)
    assert out[0, 2] == pytest.approx(1.7)


@pytest.mark.unit
def test_z_up_points_are_left_alone() -> None:
    pts = np.array([[0.1, 0.2, 1.7]])
    np.testing.assert_allclose(model_preview.to_plot_frame(pts, Z_UP_GRAVITY), pts)


@pytest.mark.unit
def test_default_anthro_pelvis_is_above_ankles() -> None:
    pytest.importorskip("opensim")
    segments = model_preview.body_segments(geometry_resolver.DEFAULT_MODEL)
    by_name = {s.name: s for s in segments}
    pelvis_z = by_name["joint_femur_r"].start[2]
    ankle_z = by_name["joint_talus_r"].end[2]
    assert pelvis_z > ankle_z + 0.5


@pytest.mark.unit
def test_capsules_only_for_bodies_with_missing_meshes() -> None:
    pytest.importorskip("opensim")
    everything = model_preview.body_segments(geometry_resolver.DEFAULT_MODEL)
    assert everything
    none_missing = model_preview.body_segments(
        geometry_resolver.DEFAULT_MODEL, only_bodies=frozenset()
    )
    assert none_missing == []
    one = everything[0].body
    only_one = model_preview.body_segments(
        geometry_resolver.DEFAULT_MODEL, only_bodies=frozenset({one})
    )
    assert only_one and all(s.body == one for s in only_one)


@pytest.mark.unit
def test_drawn_golfer_stands_upright() -> None:
    pytest.importorskip("opensim")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure()
    segments = model_preview.body_segments(geometry_resolver.DEFAULT_MODEL)
    assert model_preview.draw_segments(fig, segments) == len(segments)
    lines = fig.axes[0].lines
    zs = [z for line in lines for z in line.get_data_3d()[2]]
    head = max(zs)
    feet = min(zs)
    assert head - feet > 1.2
    # The vertical extent must dominate the horizontal one (not lying down).
    xs = [x for line in lines for x in line.get_data_3d()[0]]
    ys = [y for line in lines for y in line.get_data_3d()[1]]
    assert (head - feet) > max(max(xs) - min(xs), max(ys) - min(ys)) * 0.9
