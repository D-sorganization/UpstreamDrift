"""Optional colouring of export ellipsoids by kind instead of by point (#15150)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.hand_mobility import HandMobility
from src.shared.python.force_overlay.palette import (
    ELLIPSOID_COLOR_MODES,
    ELLIPSOID_KIND_PALETTE,
    ellipsoid_kind_rgba,
)
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.core import ELLIPSOID_KINDS, ExportSettings
from src.tools.native_viewer_export.overlay import (
    OverlayOptions,
    MobilitySelection,
    _build_hand_ellipsoid_feed,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
KINDS = ("mobility", "effective-mass")


def _swing() -> SimpleNamespace:
    spec = b'{"frames": {"LW": {"anchor": [0.1, 0.2, 0.3]}, "RW": {"anchor": [-0.1, 0.2, 0.3]}}}'
    eye, ones, mask = np.eye(3), np.ones(3), np.zeros(3, bool)

    def seg_mob(idx: int, points: Any = None, **kw: Any) -> dict[str, HandMobility]:
        return {
            p.name: HandMobility(
                p.name, (0.1, 0.2, 0.3), eye, eye, ones, ones, 5.0, mask, {}
            )
            for p in points or []
        }

    return SimpleNamespace(
        bundle=SimpleNamespace(spec_bytes=spec, coordinate_order=("j1",)),
        n_states=10,
        q=np.zeros((10, 1)),
        segment_mobility_at=seg_mob,
    )


def _colours(**extra: Any) -> dict[str, set[tuple[float, ...]]]:
    glyph_fn, _, _ = _build_hand_ellipsoid_feed(
        _swing(),  # type: ignore[arg-type]
        active_ellipsoids=KINDS,
        hand_mobility_mode="constrained",
        ellipsoid_scale="auto",
        selection=MobilitySelection("wrists"),
        **extra,
    )
    glyphs = glyph_fn(0)
    assert len(glyphs) == 4
    return {
        kind: {tuple(g.rgba) for g in glyphs if g.label.endswith(f" {kind}")}
        for kind in KINDS
    }


def test_kind_palette_covers_every_kind_with_distinct_colours() -> None:
    assert set(ELLIPSOID_KIND_PALETTE) == set(ELLIPSOID_KINDS)
    colours = [ellipsoid_kind_rgba(kind)[:3] for kind in ELLIPSOID_KINDS]
    assert len(set(colours)) == len(ELLIPSOID_KINDS)
    for rgba in ELLIPSOID_KIND_PALETTE.values():
        assert all(0.0 <= c <= 1.0 for c in rgba) and 0.0 < rgba[3] < 1.0
    with pytest.raises(ValueError, match="ellipsoid kind"):
        ellipsoid_kind_rgba("stiffness")


def test_default_keeps_point_colours_shared_by_both_kinds() -> None:
    assert ELLIPSOID_COLOR_MODES[0] == "point"
    assert OverlayOptions().ellipsoid_color_by == "point"
    assert ExportSettings().ellipsoid_color_by == "point"
    colours = _colours()
    assert colours["mobility"] == colours["effective-mass"]
    assert _colours(ellipsoid_color_by="point") == colours


def test_colour_by_kind_separates_mobility_from_effective_mass() -> None:
    colours = _colours(ellipsoid_color_by="kind")
    for kind in KINDS:
        assert colours[kind] == {ellipsoid_kind_rgba(kind)}
    assert colours["mobility"] != colours["effective-mass"]


def test_unknown_colour_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="ellipsoid_color_by"):
        _colours(ellipsoid_color_by="rainbow")
    with pytest.raises(ValueError, match="ellipsoid_color_by"):
        ExportSettings(ellipsoid_color_by="rainbow")


def test_cli_flag_reaches_the_settings() -> None:
    parser = cli.build_parser()
    assert "--ellipsoid-color-by" in parser.format_help()
    default = cli.build_settings(parser.parse_args(BASE))
    assert default.ellipsoid_color_by == "point"
    chosen = cli.build_settings(
        parser.parse_args([*BASE, "--ellipsoid-color-by", "kind"])
    )
    assert chosen.ellipsoid_color_by == "kind"
    with pytest.raises(SystemExit):
        parser.parse_args([*BASE, "--ellipsoid-color-by", "rainbow"])
