"""Overlaying an open-fp swing file on a ground-reaction series (OF-07, #13087)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
)
from src.shared.python.force_plates import (
    PlateTimeAlignment,
    PlateUncertainty,
    placement_for_target,
)
from src.shared.python.force_plates.open_fp_format import PlateDeclarations
from src.shared.python.force_plates.open_fp_format.overlay_file import (
    overlay_swing_file,
)
from src.shared.python.impact_parameters.target_frame import TargetFrame
from tests.unit.force_plates import _open_fp_writer as w

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

TIMES = (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030)


def declared() -> PlateDeclarations:
    return PlateDeclarations(
        uncertainty=PlateUncertainty(
            force_floor_n=2.0,
            force_fraction=0.01,
            cop_m=0.010,
            sensing_depth_sigma_m=0.002,
            source="declared hypothesis for tests",
        ),
        sensing_depth_m=0.015,
        capacity_n=3923.0,
    )


def alignment() -> PlateTimeAlignment:
    return PlateTimeAlignment(
        offset_s=0.0, sigma_offset_s=0.001, kind="manual", note="test clock"
    )


def model_series() -> GroundReactionPlotSeries:
    flat = {a: [0.0] * len(TIMES) for a in ("x", "y", "z", "magnitude")}
    return GroundReactionPlotSeries(
        time_s=TIMES,
        feet=("left", "right"),
        traces={"left_force_n": flat},
        load_share={},
        available=True,
    )


def targeted(handedness: str = "RH"):
    frame = TargetFrame(handedness="right" if handedness == "RH" else "left")
    return placement_for_target(handedness, frame, origin_m=(0.0, 0.0, 0.0))


def test_file_overlay_reads_loads_and_flags_unqualified(tmp_path: Path) -> None:
    path = w.write_swing(tmp_path)
    out = overlay_swing_file(
        model_series(),
        path,
        declared=declared(),
        placement=targeted(),
        alignment=alignment(),
    )
    left = out.traces["measured_left_force_n"]["z"]
    assert left[:6] == pytest.approx(list(w.FZ_L))
    assert left[6] is None  # beyond the last plate sample: never extrapolated
    assert out.measured_qualified is False
    assert "force.impact" in out.events


def test_handedness_disagreement_is_refused(tmp_path: Path) -> None:
    path = w.write_swing(tmp_path, golfer="RH")
    with pytest.raises(ValueError, match="handedness"):
        overlay_swing_file(
            model_series(),
            path,
            declared=declared(),
            placement=targeted("LH"),
            alignment=alignment(),
        )


def test_placement_type_is_checked(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="TargetedPlacement"):
        overlay_swing_file(
            model_series(),
            w.write_swing(tmp_path),
            declared=declared(),
            placement=object(),  # type: ignore[arg-type]
            alignment=alignment(),
        )
