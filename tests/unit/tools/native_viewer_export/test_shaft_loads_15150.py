"""Shaft internal loads drawn by the native export (#15150 Track Z, Z9c).

The geometry layer turns a :class:`ShaftSectionLoads` row into one arrow per
shaft station without an engine or GL. A cantilever with a tip demand gives
known loads, so the arrows are checked exactly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.simulation_backends.shaft_load_split import section_loads
from src.tools.native_viewer_export import shaft_loads
from src.tools.native_viewer_export.core import ExportSettings
from src.tools.native_viewer_export.overlay_feed import OverlayFeed

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
N, TIP_N = 5, np.array([0.0, 0.0, -10.0])
SEEDS = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


def _cantilever():
    """Shaft along +y; the last link needs ``TIP_N`` at ``y = 1`` from the
    proximal side, so joint ``i`` passes ``TIP_N`` and ``(r_tip - p_i) x TIP_N``."""
    points = np.zeros((N, 3))
    points[:, 1] = np.arange(N) / N
    forces, moments = np.zeros((N, 3)), np.zeros((N, 3))
    forces[-1], moments[-1] = TIP_N, np.cross([0.0, 1.0, 0.0], TIP_N)
    axes = np.tile([0.0, 1.0, 0.0], (N, 1))
    return points, section_loads(points, axes, forces, moments)


@pytest.fixture
def look():
    return replace(shaft_loads.load_look(), station_stride=1, max_length_m=10.0)


def _tips(arrows):
    return np.array([np.subtract(a.tip_m, a.tail_m) for a in arrows])


def test_settings_refuse_bad_values(tmp_path) -> None:
    raw = json.loads(shaft_loads.SETTINGS_PATH.read_text(encoding="utf-8"))
    bad = (
        ({"station_stride": 0}, "station_stride"),
        ({"parts": {**raw["parts"], "total": {"hex": "red", "text": ""}}}, "RRGGBB"),
        ({"quantities": {"bending": raw["quantities"]["bending"]}}, "quantities"),
    )
    for change, message in bad:
        path = tmp_path / "s.json"
        path.write_text(json.dumps({**raw, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            shaft_loads.load_look(path)


def test_bending_and_shear_arrows_are_the_loads_at_the_stations(look) -> None:
    points, loads = _cantilever()
    bending = shaft_loads.shaft_load_arrows(points, loads, "bending", "total", look)
    arms = 1.0 - points[:, 1]
    moments = np.outer(-10.0 * arms, [1.0, 0.0, 0.0])  # (arm e_y) x (-10 e_z)
    np.testing.assert_allclose([a.tail_m for a in bending], points, atol=1e-12)
    scale = look.quantities["bending"]["scale_m"]
    np.testing.assert_allclose(_tips(bending), scale * moments, atol=1e-12)
    assert [a.magnitude for a in bending] == pytest.approx(10.0 * arms)
    assert {a.units for a in bending} == {"N m"} and not any(a.clamped for a in bending)
    shear = shaft_loads.shaft_load_arrows(points, loads, "shear", "total", look)
    scale = look.quantities["shear"]["scale_m"]
    np.testing.assert_allclose(_tips(shear), np.tile(scale * TIP_N, (N, 1)), atol=1e-12)
    assert {a.units for a in shear} == {"N"}


def test_long_arrows_are_clamped_small_ones_dropped_and_stride_kept(look) -> None:
    points, loads = _cantilever()
    q = {**look.quantities, "bending": {**look.quantities["bending"], "floor": 2.5}}
    tight = replace(look, max_length_m=1e-3, quantities=q)
    arrows = shaft_loads.shaft_load_arrows(points, loads, "bending", "total", tight)
    assert len(arrows) == 4  # the last station's 2 N m is below the floor
    assert np.linalg.norm(_tips(arrows), axis=1) == pytest.approx([1e-3] * 4)
    assert all(a.clamped for a in arrows)
    strided = replace(look, station_stride=2)
    arrows = shaft_loads.shaft_load_arrows(points, loads, "bending", "ztcf", strided)
    np.testing.assert_allclose([a.tail_m[1] for a in arrows], points[::2, 1])
    rgb = tuple(round(255 * c) for c in arrows[0].rgba[:3])
    assert shaft_loads.colour_key("bending", "ztcf", look)[0][1] == rgb
    for quantity, part in (("torsion", "total"), ("bending", "sum")):
        with pytest.raises(ValueError, match="quantity|part"):
            shaft_loads.shaft_load_arrows(points, loads, quantity, part, look)


def test_legend_states_units_sign_convention_and_scale(look) -> None:
    text = shaft_loads.legend("bending", "zvcf", look)
    assert "N m" in text and "right-hand" in text and "ZVCF" in text
    assert f"1 m = {1.0 / look.quantities['bending']['scale_m']:g} N m" in text
    assert "proximal" in shaft_loads.legend("shear", "total", look)


def test_with_shaft_loads_keeps_the_feed_arrows_legend_and_key(look) -> None:
    feed = OverlayFeed(
        lambda i: None,
        extra_arrows=lambda i: ("hand",),
        extra_legend="hands",
        colour_key=(("Hand", (1, 2, 3)),),
    )
    out = shaft_loads.with_shaft_loads(feed, lambda i: (i,), "bending", "total", look)
    assert out.extra_arrows(3) == ("hand", 3)
    assert out.extra_legend.startswith("hands | ") and "N m" in out.extra_legend
    assert out.colour_key[0] == ("Hand", (1, 2, 3)) and len(out.colour_key) == 2
    out = shaft_loads.with_shaft_loads(OverlayFeed(id), id, "shear", "total", look)
    assert out.extra_arrows is id and "hands" not in out.extra_legend


def test_export_settings_and_cli_flags() -> None:
    from src.tools.native_viewer_export.cli import build_parser, build_settings

    assert ExportSettings(shaft_loads="shear").shaft_loads_part == "total"
    for kwargs in (
        {"shaft_loads": "torsion"},
        {"shaft_loads": "bending", "shaft_loads_part": "sum"},
        {"shaft_loads": "bending", "overlays": False},
    ):
        with pytest.raises(ValueError, match="quantity|part|overlays"):
            ExportSettings(**kwargs)
    base = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    flags = ["--shaft-loads", "bending", "--shaft-loads-part", "ztcf"]
    settings = build_settings(build_parser().parse_args([*base, *flags]))
    assert (settings.shaft_loads, settings.shaft_loads_part) == ("bending", "ztcf")
    assert build_settings(build_parser().parse_args(base)).shaft_loads is None
    assert (
        shaft_loads.feed_with_options(id, None, SimpleNamespace(shaft_loads=None)) is id
    )


@pytest.mark.requires_pinocchio
def test_live_glyphs_match_the_shaft_load_split() -> None:
    """On the seed model with a bent chain shaft, each arrow is the split's load."""
    pytest.importorskip("pinocchio")
    from src.shared.python.motion_matching.execution.spec_builder import (
        with_shaft_model,
    )

    doc = json.loads((SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes())
    spec = json.dumps(with_shaft_model(doc, "driver", "chain")).encode()
    order = tuple(json.loads(spec)["coordinate_order"])
    rigid = len(doc["coordinate_order"])
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        pad = ((0, 0), (0, len(order) - rigid))
        q, v, tau = (np.pad(rec[n][399:401], pad) for n in ("q", "v", "tau"))
    q[:, rigid:] = 0.02  # bent: chain bending is the joint spring moment
    bundle = SimpleNamespace(
        spec_bytes=spec, coordinate_order=order, dt_s=0.01, efforts=tau, reference_v=v
    )
    swing = SimpleNamespace(bundle=bundle, q=q, v=v)
    look = replace(shaft_loads.load_look(), station_stride=1, max_length_m=1e6)
    glyphs = shaft_loads.shaft_load_glyphs(swing, "bending", "total", look)
    arrows = glyphs(0)
    assert arrows and glyphs(0) is arrows  # evaluated once per state
    expected = shaft_loads._sampler(swing)(0)[0].total().bending_magnitude_nm[0]
    kept = expected[expected >= look.quantities["bending"]["floor"]]
    np.testing.assert_allclose([a.magnitude for a in arrows], kept, rtol=1e-9)
