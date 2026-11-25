"""Midhands equivalent force and couple of both hands on the club (#15720)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    TorqueArcGlyph,
)
from src.shared.python.simulation_backends.hand_force_split import (
    HandWrench,
    HandWrenchSplit,
)
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.midhands import (
    MIDHANDS_SETTINGS_PATH,
    equivalent_wrench,
    load_midhands_settings,
    midhands_glyphs,
    midhands_key,
)
from src.tools.native_viewer_export.ztcf_feed import HAND_CHOICES, hands_of

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


def _random_wrenches(seed: int):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(2, 3)),
        rng.normal(size=(2, 3)) * 100,
        rng.normal(size=(2, 3)) * 10,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_equivalent_wrench_is_the_sum_force_and_the_midpoint_couple(
    seed: int,
) -> None:
    (p_lead, p_trail), (f_lead, f_trail), (m_lead, m_trail) = _random_wrenches(seed)
    mid, force, couple = equivalent_wrench(
        [p_lead, p_trail], [f_lead, f_trail], [m_lead, m_trail]
    )
    p_mid = (p_lead + p_trail) / 2
    np.testing.assert_allclose(mid, p_mid)
    np.testing.assert_allclose(force, f_lead + f_trail)
    lever = np.cross(p_lead - p_mid, f_lead) + np.cross(p_trail - p_mid, f_trail)
    np.testing.assert_allclose(couple, m_lead + m_trail + lever)
    # the same moment about another point as the two hand wrenches
    probe = np.array([0.3, -1.2, 2.0])
    about_probe = (
        m_lead
        + m_trail
        + np.cross(p_lead - probe, f_lead)
        + np.cross(p_trail - probe, f_trail)
    )
    np.testing.assert_allclose(couple + np.cross(mid - probe, force), about_probe)


def test_the_ztcf_and_zvcf_equivalents_add_up_to_the_total_case() -> None:
    from src.tools.native_viewer_export.midhands import case_wrench

    split = _split()
    _, f_z, m_z = case_wrench(split, "ztcf")
    _, f_v, m_v = case_wrench(split, "zvcf")
    _, f_t, m_t = case_wrench(split, "total")
    np.testing.assert_allclose(f_z, [0, 0, 200.0])  # drift incl. gravity only
    np.testing.assert_allclose(f_v, 0.0, atol=1e-12)  # applied torques only
    np.testing.assert_allclose(f_z + f_v, f_t)
    np.testing.assert_allclose(m_z + m_v, m_t)


def test_bad_shapes_or_non_finite_values_are_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        equivalent_wrench(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        equivalent_wrench(np.full((2, 3), np.nan), np.zeros((2, 3)), np.zeros((2, 3)))


def _split() -> HandWrenchSplit:
    def hw(p, f, t):
        return HandWrench(np.array([p]), np.array([f]), np.array([t]))

    lead = {
        "ztcf": hw((0, 0.05, 1), (0, 0, 100.0), (0, 0, 0)),
        "zvcf": hw((0, 0.05, 1), (200.0, 0, 0), (0, 0, 0)),
    }
    trail = {
        "ztcf": hw((0, -0.05, 1), (0, 0, 100.0), (0, 0, 0)),
        "zvcf": hw((0, -0.05, 1), (-200.0, 0, 0), (0, 0, 0)),
    }
    return HandWrenchSplit({"lead": lead, "trail": trail})


STYLE = ForceGlyphStyle(
    force_scale_m_per_n=0.001, torque_scale_m_per_nm=0.01, max_length_m=3.0
)


def test_each_case_draws_its_equivalent_force_and_couple_in_the_settings_colour() -> (
    None
):
    settings = load_midhands_settings()
    ztcf = midhands_glyphs(_split(), "ztcf", STYLE, settings)
    (arrow,) = (g for g in ztcf if isinstance(g, ArrowGlyph))
    np.testing.assert_allclose(np.subtract(arrow.tip_m, arrow.tail_m), [0, 0, 0.2])
    np.testing.assert_allclose(arrow.tail_m, [0, 0, 1])
    assert not any(isinstance(g, TorqueArcGlyph) for g in ztcf)  # no couple
    zvcf = midhands_glyphs(_split(), "zvcf", STYLE, settings)
    assert not any(isinstance(g, ArrowGlyph) for g in zvcf)  # pure push-pull
    (arc,) = (g for g in zvcf if isinstance(g, TorqueArcGlyph))
    assert arc.magnitude == pytest.approx(20.0)
    total = midhands_glyphs(_split(), "total", STYLE, settings)
    assert {type(g) for g in total} == {ArrowGlyph, TorqueArcGlyph}
    for g in total:
        assert g.rgba == pytest.approx(settings.rgba("total"))
    assert arrow.rgba == pytest.approx(settings.rgba("ztcf"))
    assert arc.rgba == pytest.approx(settings.rgba("zvcf"))
    with pytest.raises(ValueError, match="case"):
        midhands_glyphs(_split(), "sideways", STYLE, settings)


def test_the_key_names_the_case_and_the_settings_are_validated(tmp_path) -> None:
    settings = load_midhands_settings()
    ((text, rgb),) = midhands_key("zvcf", settings)
    assert "ZVCF" in text and "midpoint" in text.lower()
    assert settings.key_text in text
    assert rgb == tuple(round(255 * c) for c in settings.rgba("zvcf")[:3])
    doc = json.loads(MIDHANDS_SETTINGS_PATH.read_text(encoding="utf-8"))
    no_key = {k: v for k, v in doc.items() if k != "key_text"}
    (tmp_path / "k.json").write_text(json.dumps(no_key), encoding="utf-8")
    with pytest.raises(ValueError, match="key_text"):
        load_midhands_settings(tmp_path / "k.json")
    doc["case_hex"]["zvcf"] = "yellow"
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="case_hex"):
        load_midhands_settings(bad)


def test_the_hands_flag_offers_the_midhands_equivalent() -> None:
    assert "midhands" in HAND_CHOICES
    assert hands_of("midhands") == ()
    args = cli.build_parser().parse_args([*BASE, "--ztcf-zvcf-hands", "midhands"])
    assert args.ztcf_zvcf_hands == "midhands"


def test_the_feed_draws_extra_couple_arcs_as_torque_arcs() -> None:
    from src.tools.native_viewer_export.overlay_feed import OverlayFeed
    from src.shared.python.force_overlay.contracts import ForceTorqueFrame

    extra = midhands_glyphs(_split(), "total", STYLE)
    feed = OverlayFeed(
        lambda _i: ForceTorqueFrame(0.0, "t", ()), extra_arrows=lambda _i: extra
    )
    glyphs = feed.glyphs_at(0)
    assert len(glyphs.arrows) == 1 and len(glyphs.torque_arcs) == 1


def test_the_overlay_draws_the_midhands_case_with_its_key(monkeypatch) -> None:
    from src.shared.python.force_overlay.contracts import ForceTorqueFrame
    from src.tools.native_viewer_export import overlay, ztcf_feed
    from src.tools.native_viewer_export.overlay_feed import OverlayFeed

    monkeypatch.setattr(
        ztcf_feed, "split_glyphs", lambda _swing, draw: lambda _i: draw(_split())
    )
    feed = OverlayFeed(lambda _i: ForceTorqueFrame(0.0, "t", ()))
    opts = overlay.OverlayOptions(ztcf_hands="midhands", ztcf_view="zvcf")
    out = overlay._with_ztcf_zvcf(feed, object(), STYLE, opts)
    assert out.colour_key == midhands_key("zvcf")
    assert "midpoint" in out.extra_legend
    (arc,) = out.extra_arrows(0)
    assert isinstance(arc, TorqueArcGlyph)
    # the arrow gain lengthens the couple too: it is tens of N m
    gained = overlay.OverlayOptions(
        ztcf_hands="midhands", ztcf_view="zvcf", ztcf_arrow_gain=2.0
    )
    (big,) = overlay._with_ztcf_zvcf(feed, object(), STYLE, gained).extra_arrows(0)
    assert big.radius_m == pytest.approx(2 * arc.radius_m)
    tip_to_tail = overlay.OverlayOptions(ztcf_hands="midhands")
    with pytest.raises(ValueError, match="midhands"):
        overlay._with_ztcf_zvcf(feed, object(), STYLE, tip_to_tail)
