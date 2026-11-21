"""Stored kinetic-energy surface shading acceptance (#15396)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.body_part_viz.segment_ke_settings import (
    dump_segment_ke_state,
    parse_segment_ke_state,
)
from src.shared.python.segment_kinetic_energy.contracts import (
    SegmentKineticEnergyFrame,
    SegmentKineticEnergySample,
)
from src.shared.python.segment_kinetic_energy.shading import (
    KE_PROXY_GLOBE_LABEL,
    REFERENCE_SWING_PEAK_SEGMENT_KE_J,
    component_values_j,
    default_stored_ke_color_scale,
    format_stored_ke_legend,
    ui_strings_for_ke_view,
)

pytestmark = pytest.mark.unit


def _frame(total_scale: float = 1.0) -> SegmentKineticEnergyFrame:
    trans, rot = 30.0 * total_scale, 20.0 * total_scale
    sample = SegmentKineticEnergySample(
        body="upper",
        body_index=1,
        com_m=(0.0, 0.0, 1.0),
        translational_j=trans,
        rotational_j=rot,
        total_j=trans + rot,
    )
    return SegmentKineticEnergyFrame(
        schema_version=1, time_s=0.1, engine="mujoco", segments=(sample,)
    )


def test_translational_plus_rotational_equals_total_per_frame() -> None:
    frame = _frame()
    total = component_values_j(frame, "total")["upper"]
    trans = component_values_j(frame, "translational")["upper"]
    rot = component_values_j(frame, "rotational")["upper"]
    assert trans is not None and rot is not None and total is not None
    assert total == pytest.approx(trans + rot)


def test_constant_ke_same_colour_across_frames_fixed_limits() -> None:
    scale = default_stored_ke_color_scale()
    scale = replace(scale, enabled=True)
    frame_a = _frame(1.0)
    frame_b = replace(_frame(1.0), time_s=0.2)
    color_a = scale.evaluate(component_values_j(frame_a, "total")["upper"], "#101010")
    color_b = scale.evaluate(component_values_j(frame_b, "total")["upper"], "#101010")
    assert color_a == color_b


def test_legend_reports_units_limits_transform_and_saturation_note() -> None:
    scale = replace(default_stored_ke_color_scale(), enabled=True)
    line = format_stored_ke_legend(scale, mode="total")
    assert "J" in line
    assert "cividis" in line
    assert "saturat" in line.lower()
    assert "sqrt" in line or "linear" in line
    beyond = scale.evaluate(scale.high_limit * 1.5, "#ffffff")
    assert beyond.saturated is True


def test_settings_persistence_round_trip() -> None:
    scale = replace(
        default_stored_ke_color_scale(), enabled=True, display_transform="sqrt"
    )
    text = dump_segment_ke_state(
        scale, mode="rotational", toggles={"show_ke_proxy_globes": True}
    )
    loaded_scale, mode, toggles = parse_segment_ke_state(text)
    assert loaded_scale == scale
    assert mode == "rotational"
    assert toggles["show_ke_proxy_globes"] is True


def test_high_limit_j_persists_round_trip() -> None:
    custom = 12345.6
    scale = replace(default_stored_ke_color_scale(), enabled=True, high_limit=custom)
    text = dump_segment_ke_state(scale, mode="total", toggles={})
    loaded, _, _ = parse_segment_ke_state(text)
    assert loaded.high_limit == pytest.approx(custom)


def test_legend_lists_clipped_segments() -> None:
    scale = replace(default_stored_ke_color_scale(), enabled=True)
    line = format_stored_ke_legend(
        scale, mode="total", clipped_segments=frozenset({"a"})
    )
    assert "clipped" in line.lower()
    assert "a" in line


def test_default_high_limit_matches_reference_swing() -> None:
    scale = default_stored_ke_color_scale()
    assert scale.high_limit == pytest.approx(REFERENCE_SWING_PEAK_SEGMENT_KE_J)


def test_ke_proxy_globe_label_exposed() -> None:
    strings = ui_strings_for_ke_view()
    assert strings["proxy_globe_label"] == KE_PROXY_GLOBE_LABEL


def test_ke_view_strings_never_use_transfer_wording() -> None:
    strings = " ".join(ui_strings_for_ke_view().values()).lower()
    legend = format_stored_ke_legend(
        replace(default_stored_ke_color_scale(), enabled=True), mode="total"
    ).lower()
    assert "transfer" not in strings
    assert "transfer" not in legend
