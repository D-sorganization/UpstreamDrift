"""Tension and compression by part: ZTCF and ZVCF section loads (#15150 Z4b).

The owner asked to see which tension and compression come from the ZTCF
(drift: gravity, momentum and passive springs, ``tau = 0``) and which from
the ZVCF (control: the applied torques alone, at zero velocity, no gravity).
Section loads are linear in the acceleration, the external wrenches and
gravity, so each part is the inverse dynamics of its own acceleration and
reactions; the ZVCF drops the gravity term by subtracting the unloaded,
still pose, and the parts add to the total.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.force_overlay.contracts import ForceTorqueFrame
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.overlay import OverlayOptions, part_shading
from src.tools.native_viewer_export.overlay_feed import OverlayFeed
from src.tools.native_viewer_export.ztcf_axial import (
    SHADING_NOTES,
    SHADING_PARTS,
    PartLoad,
    part_axial_frames,
    with_part_shading,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
GRAVITY = 7.0


def _linear(rates, accel, ground, closure) -> AxialLoadFrame:
    """A section load linear in a, ground and closure; gravity and v^2 terms."""
    value = (
        2.0 * float(np.sum(accel))
        + 3.0 * float(sum(ground))
        + 5.0 * float(closure)
        + GRAVITY
        + 11.0 * float(np.sum(rates)) ** 2
    )
    return AxialLoadFrame(0.5, {"femur": value, "hand": None}, "fake")


DRIFT = PartLoad(np.array([1.0, 2.0]), (4.0,), 0.5)
CONTROL = PartLoad(np.array([-3.0, 1.0]), (-1.0,), 2.0)
V = np.array([0.5, 0.5])


def test_each_part_is_its_own_inverse_and_the_parts_add_to_the_total() -> None:
    frames = part_axial_frames(_linear, V, DRIFT, CONTROL, unloaded=0.0)
    assert set(frames) == set(SHADING_PARTS) - {"recorded"}
    ztcf = 2 * 3 + 3 * 4 + 5 * 0.5 + GRAVITY + 11 * 1.0
    zvcf = 2 * -2 + 3 * -1 + 5 * 2.0  # no gravity, no velocity term
    assert frames["ztcf"].values_n["femur"] == pytest.approx(ztcf)
    assert frames["zvcf"].values_n["femur"] == pytest.approx(zvcf)
    assert frames["total"].values_n["femur"] == pytest.approx(ztcf + zvcf)
    assert all(f.values_n["hand"] is None for f in frames.values())
    assert "ZVCF" in frames["zvcf"].source and "ZTCF" in frames["ztcf"].source


def test_an_unsupported_model_gives_no_part_frames() -> None:
    frames = part_axial_frames(lambda *_: None, V, DRIFT, CONTROL, unloaded=0.0)
    assert all(f is None for f in frames.values())


def _frame(axial: AxialLoadFrame | None) -> ForceTorqueFrame:
    return ForceTorqueFrame(0.5, "mujoco", (), axial_loads=axial)


def test_a_part_replaces_the_shading_and_keeps_the_impact_blackout() -> None:
    parts = part_axial_frames(_linear, V, DRIFT, CONTROL, unloaded=0.0)
    recorded = AxialLoadFrame(0.5, {"femur": -900.0, "hand": None}, "recorded")
    blackout = AxialLoadFrame(0.5, {"femur": None, "hand": None}, "unavailable")
    frames = {0: _frame(recorded), 1: _frame(blackout), 2: _frame(None)}
    shaded = with_part_shading(frames.__getitem__, lambda _i: parts, "zvcf")
    assert shaded(0).axial_loads == parts["zvcf"]
    assert shaded(1).axial_loads.values_n["femur"] is None  # ball contact
    assert shaded(2).axial_loads is None
    same = with_part_shading(frames.__getitem__, lambda _i: parts, "recorded")
    assert same(0) is frames[0]
    with pytest.raises(ValueError, match="part"):
        with_part_shading(frames.__getitem__, lambda _i: parts, "sideways")


def test_the_shading_part_flag_defaults_to_the_recorded_loads() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(BASE).segment_shading_part == "recorded"
    args = parser.parse_args([*BASE, "--segment-shading-part", "zvcf"])
    assert args.segment_shading_part == "zvcf"
    with pytest.raises(SystemExit):
        parser.parse_args([*BASE, "--segment-shading-part", "sideways"])


def test_the_hud_names_the_shading_part_after_the_forces_note() -> None:
    feed = OverlayFeed(
        lambda _i: _frame(None),
        forces_note="forces",
        shading_note=SHADING_NOTES["zvcf"],
    )
    assert feed.hud_lines_at(0) == ("forces", SHADING_NOTES["zvcf"])
    assert OverlayFeed(lambda _i: _frame(None)).hud_lines_at(0) == ()


def test_recorded_shading_keeps_the_frames_and_adds_no_note() -> None:
    frame_at = {0: _frame(None)}.__getitem__
    opts = OverlayOptions(shading_part="recorded")
    assert part_shading(frame_at, None, True, opts) == (frame_at, "")
    unshaded = OverlayOptions(shading_part="ztcf")  # no axial loads computed
    assert part_shading(frame_at, None, False, unshaded) == (frame_at, "")
    with pytest.raises(ValueError, match="part"):
        part_shading(frame_at, None, True, OverlayOptions(shading_part="sideways"))


SEEDS = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


@pytest.mark.requires_pinocchio
def test_live_parts_add_up_on_the_seed_swing() -> None:
    """ZTCF + ZVCF section loads equal the inverse of the summed solve."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.ztcf_axial import part_axial_source

    spec = (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()
    order = tuple(json.loads(spec)["coordinate_order"])
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        q, v, tau = (rec[n][300] for n in ("q", "v", "tau"))
    at = part_axial_source(spec, order)
    frames = at(q, v, tau, 1.25)
    ztcf, zvcf, total = (frames[p].values_n for p in ("ztcf", "zvcf", "total"))
    shown = [b for b, x in total.items() if x is not None]
    assert len(shown) > 10
    for body in shown:
        assert total[body] == pytest.approx(ztcf[body] + zvcf[body], abs=1e-6)
    summed = at.summed(q, v, tau, 1.25).values_n
    for body in shown:
        assert total[body] == pytest.approx(summed[body], rel=1e-6, abs=1e-6)
