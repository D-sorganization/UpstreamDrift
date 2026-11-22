"""Roll segments and landing labels of the hole figure (#15443, child 6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.tools.golf_environment import shots as sh
from tests.tools.golf_environment.test_shots import _Hole, _flight, _place

pytestmark = pytest.mark.unit

ROLL = (21.0, -4.0)  # (down-range, left) metres


def _hf():
    try:
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib not loadable: {exc}")
    from src.tools.golf_environment import hole_figure

    return hole_figure


def _rolled(carry: float, lateral: float, roll=ROLL) -> sh.HoleShot:
    (shot,) = _place([_flight(carry, lateral)])
    total = carry + float(np.hypot(*roll))
    return replace(
        shot, roll_m=float(np.hypot(*roll)), total_m=total, roll_offset_m=roll
    )


def _build(shots, label=True):
    hf = _hf()
    return hf.build_hole_figure(
        shots, _Hole(), sh.load_shots_settings(), label_at_landing=label
    )


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def test_shot_without_roll_keeps_its_fields_unset() -> None:
    (shot,) = _place([_flight(200.0, 5.0)])
    assert shot.roll_m is None and shot.total_m is None
    assert shot.roll_offset_m is None


def test_roll_fields_come_together_and_are_valid() -> None:
    (shot,) = _place([_flight(200.0, 5.0)])
    with pytest.raises(ValueError, match="together"):
        replace(shot, roll_m=3.0)
    with pytest.raises(ValueError, match="roll"):
        replace(shot, roll_m=-1.0, total_m=199.0, roll_offset_m=(1.0, 0.0))


def test_roll_is_a_ground_segment_from_landing_to_rest_in_both_views() -> None:
    fig, plan, side = _build([_rolled(200.0, 10.0)])
    yd = 1.0 / 0.9144
    rolls = [ln for ax in (plan, side) for ln in ax.lines if ln.get_gid() == "roll"]
    assert len(rolls) == 2
    px, py = rolls[0].get_data()
    assert px[0] == pytest.approx(200.0 * yd) and px[-1] == pytest.approx(221.0 * yd)
    assert py[0] == pytest.approx(-10.0 * yd)
    assert py[-1] == pytest.approx((-10.0 - 4.0) * yd)
    _, sz = rolls[1].get_data()
    assert list(sz) == [0.0, 0.0]


def test_no_roll_lines_or_labels_without_roll_or_option() -> None:
    (plain,) = _place([_flight(200.0, 5.0)])
    _, plan, side = _build([plain])
    assert not [ln for ax in (plan, side) for ln in ax.lines if ln.get_gid() == "roll"]
    texts = [t.get_text() for t in plan.texts if t.get_gid() == "landing_label"]
    assert texts == ["Carry 219 yd, Side 5 yd R"]  # carry and side, no roll line
    _, plan, _ = _build([_rolled(200.0, 5.0)], label=False)
    assert not [t for t in plan.texts if t.get_gid() == "landing_label"]


def test_label_states_carry_side_roll_and_total_in_display_units() -> None:
    _, plan, _ = _build([_rolled(200.0, -10.0)])
    (text,) = [t.get_text() for t in plan.texts if t.get_gid() == "landing_label"]
    assert text.splitlines()[0] == "Carry 219 yd, Side 11 yd L"
    assert text.splitlines()[1].startswith("Roll 23 yd, Total 242 yd")
    assert text.endswith("(model)")


@pytest.mark.parametrize("carry", [60.0, 200.0, 370.0, 395.0])
@pytest.mark.parametrize("lateral", [-30.0, 0.0, 29.0])
def test_label_boxes_stay_inside_the_axes_and_clear_of_the_legend(
    carry, lateral
) -> None:
    shots = [_rolled(carry, lateral), _rolled(carry - 25.0, lateral / 2 + 3.0)]
    fig, plan, _ = _build(shots)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = plan.get_window_extent(renderer)
    legend = plan.get_legend().get_window_extent(renderer)
    boxes = []
    for t in plan.texts:
        if t.get_gid() == "landing_label":
            bb = t.get_window_extent(renderer)
            boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
    assert len(boxes) == 2
    for b in boxes:
        assert axes.x0 <= b[0] and b[2] <= axes.x1, b
        assert axes.y0 <= b[1] and b[3] <= axes.y1, b
        assert not _overlap(b, (legend.x0, legend.y0, legend.x1, legend.y1))
    assert not _overlap(boxes[0], boxes[1])


def test_existing_figure_output_is_unchanged_without_roll(tmp_path) -> None:
    hf = _hf()
    shots = _place([_flight(166.0, -11.0)])
    fig, plan, side = hf.build_hole_figure(shots, _Hole(), sh.load_shots_settings())
    assert not [t for t in plan.texts if t.get_gid() == "landing_label"]
