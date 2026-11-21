"""Ellipsoid scale grouping, nested visibility and display floor (#15150)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.shared.python.force_overlay.glyphs import EllipsoidGlyph
from src.tools.native_viewer_export import ellipsoid_panel as ep
from src.tools.native_viewer_export import ellipsoid_visibility as ev
from src.tools.native_viewer_export.core import ExportSettings

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

CFG = ev.load_visibility_settings()
EYE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
#: a small hand-like series and a large clubhead-like series (physical radii)
SERIES = {"hand": [[1.0, 2.0, 3.0]] * 4, "head": [[10.0, 20.0, 30.0]] * 4}


def _glyph(centre, radii, alpha=0.45, label="g"):
    return EllipsoidGlyph(centre, EYE, radii, (0.2, 0.4, 0.6, alpha), label)


def test_packaged_settings_are_valid_and_every_field_is_checked() -> None:
    assert CFG.scale_grouping in ev.GROUPINGS and CFG.target_radius_m > 0.0
    for bad in (
        {"scale_grouping": "per_swing"},
        {"target_radius_m": 0.0},
        {"outer_alpha": 0.0},
        {"outer_alpha": 1.5},
        {"min_semi_axis_m": -0.1},
        {"min_semi_axis_m": 0.3},
    ):
        with pytest.raises(ValueError):
            replace(CFG, **bad)
    with pytest.raises(TypeError, match="numeric"):
        replace(CFG, target_radius_m="big")


def test_settings_file_must_not_have_unknown_keys(tmp_path) -> None:
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"scale_grouping": "per_point", "bogus": 1}))
    with pytest.raises(ValueError, match="bogus"):
        ev.load_visibility_settings(path)


def test_per_kind_shares_one_scale_and_per_point_gives_each_its_own() -> None:
    shared = ev.group_scales(
        SERIES, "mobility", "auto", replace(CFG, scale_grouping="per_kind")
    )
    own = ev.group_scales(
        SERIES, "mobility", "auto", replace(CFG, scale_grouping="per_point")
    )
    assert shared["hand"].scale_m_per_unit == shared["head"].scale_m_per_unit
    # one point's largest displayed radius is the target at its own scale
    ratio = own["hand"].scale_m_per_unit / own["head"].scale_m_per_unit
    assert ratio == pytest.approx(10.0)
    assert 3.0 * own["hand"].scale_m_per_unit == pytest.approx(
        30.0 * own["head"].scale_m_per_unit
    )


def test_fixed_scale_is_the_same_for_every_point() -> None:
    cfg = replace(CFG, scale_grouping="per_point")
    scales = ev.group_scales(SERIES, "mobility", "fixed:0.01", cfg)
    assert {s.scale_m_per_unit for s in scales.values()} == {0.01}


def test_outer_ellipsoid_of_a_shared_centre_gets_the_outer_alpha() -> None:
    inner, outer = (
        _glyph((0, 0, 0), (0.02, 0.03, 0.04)),
        _glyph((0, 0, 0), (0.05, 0.06, 0.07)),
    )
    far = _glyph((1, 0, 0), (0.5, 0.5, 0.5))
    out = ev.fade_outer([outer, inner, far], 0.2)
    assert [g.rgba[3] for g in out] == [0.2, 0.45, 0.45]  # order and others kept
    assert out[0].radii == outer.radii and out[0].rgba[:3] == outer.rgba[:3]


def test_outer_alpha_one_changes_nothing() -> None:
    glyphs = [_glyph((0, 0, 0), (1, 1, 1)), _glyph((0, 0, 0), (2, 2, 2))]
    assert ev.fade_outer(glyphs, 1.0) == tuple(glyphs)


def test_panel_note_follows_the_grouping() -> None:
    per_point = replace(CFG, scale_grouping="per_point")
    assert ev.panel_note(per_point) == CFG.per_point_note
    assert ev.panel_note(replace(CFG, scale_grouping="per_kind")) == ""
    cfg = replace(ep.load_panel_settings(), note="Scaled Per Point")
    panel = ep.EllipsoidPanel(cfg, ("mobility",), False)
    rows = (ep.PanelRow("lead_hand", 1.0, 2.0, 3.0),)
    assert panel.lines(rows)[-1][0] == "Scaled Per Point"
    plain = ep.EllipsoidPanel(replace(cfg, note=""), ("mobility",), False)
    assert plain.lines(rows)[-1][0] == "Lead Hand"


def test_with_panel_uses_the_packaged_note_unless_set() -> None:
    feed = type("F", (), {"hand_panel_rows": staticmethod(lambda i: ())})()
    export = ExportSettings(interface_ellipsoids=("mobility",))
    overlays = ep.with_panel(None, export, feed, None)
    assert overlays.ellipsoids.settings.note == ev.panel_note(CFG)
    none = replace(ep.load_panel_settings(), note="")
    overlays = ep.with_panel(None, replace(export, ellipsoid_panel=none), feed, None)
    assert overlays.ellipsoids.settings.note == ""


def test_cli_settings_file_sets_the_panel_note(tmp_path) -> None:
    from src.tools.native_viewer_export import cli

    base = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    path = tmp_path / "v.json"
    for grouping, note in (("per_point", "Scaled Per Point"), ("per_kind", "")):
        text = json.dumps(
            {**json.loads(ev.SETTINGS_PATH.read_text()), "scale_grouping": grouping}
        )
        path.write_text(text)
        args = cli.build_parser().parse_args(
            [*base, "--ellipsoid-visibility-settings", str(path)]
        )
        assert cli.build_settings(args).ellipsoid_panel.note == note
    plain = cli.build_parser().parse_args(base)
    assert cli.build_settings(plain).ellipsoid_panel is None
