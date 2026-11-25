"""Both hands, part views and the colour key in the export (#15150 Track Z, Z3b).

The owner asked for the ZTCF (drift, with gravity) and ZVCF (control, the
applied torques alone) of both the lead and the trail hand on the same
swing, as separate renders: ZTCF alone, ZVCF alone, total force, total
torque and tip to tail, each with a key that names the arrow colours.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.tip_to_tail import PART_VIEWS, colour_key
from src.shared.python.simulation_backends.hand_force_split import (
    HANDS,
    PARTS,
    HandWrench,
    HandWrenchSplit,
)
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.core import OverlayFeed
from src.tools.native_viewer_export.ztcf_feed import (
    HAND_CHOICES,
    hand_arrows,
    hands_of,
    legend_for,
    ztcf_zvcf_arrows,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
STYLE = ForceGlyphStyle(force_scale_m_per_n=0.001, max_length_m=0.6)


def _split() -> HandWrenchSplit:
    """Lead at x = 0, trail at x = 0.1; distinct force and torque per part."""
    parts = {}
    for h, hand in enumerate(HANDS):
        point = np.array([[0.1 * h, 0.0, 1.0]])
        parts[hand] = MappingProxyType(
            {
                part: HandWrench(
                    point,
                    np.array([[100.0 * (h + 1) * (p + 1), 0.0, 0.0]]),
                    np.array([[0.0, 0.0, 5.0 * (p + 1)]]),
                )
                for p, part in enumerate(PARTS)
            }
        )
    return HandWrenchSplit(MappingProxyType(parts))


def _labels(arrows) -> list[str]:
    return [a.label for a in arrows]


def test_the_view_quantity_and_hand_flags_default_and_parse() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(BASE)
    assert (args.ztcf_zvcf_view, args.ztcf_zvcf_quantity, args.ztcf_zvcf_hands) == (
        "tip_to_tail",
        "force",
        "both",
    )
    args = parser.parse_args(
        [
            *BASE,
            "--ztcf-zvcf-view",
            "zvcf",
            "--ztcf-zvcf-quantity",
            "torque",
            "--ztcf-zvcf-hands",
            "lead",
        ]
    )
    assert (args.ztcf_zvcf_view, args.ztcf_zvcf_quantity, args.ztcf_zvcf_hands) == (
        "zvcf",
        "torque",
        "lead",
    )
    with pytest.raises(SystemExit):
        parser.parse_args([*BASE, "--ztcf-zvcf-view", "sideways"])


def test_hands_of_maps_the_choice() -> None:
    assert hands_of("both") == HANDS
    assert hands_of("lead") == ("lead",)
    assert set(HAND_CHOICES) == {"both", *HANDS, "midhands"}  # midhands: #15720
    with pytest.raises(ValueError, match="hands"):
        hands_of("left")


def test_both_hands_are_drawn_each_from_its_own_point() -> None:
    arrows = hand_arrows(_split(), STYLE, "tip_to_tail", "force", HANDS)
    labels = _labels(arrows)
    assert any(lb.startswith("Lead ") for lb in labels)
    assert any(lb.startswith("Trail ") for lb in labels)
    tails = {a.label.split()[0]: a.tail_m for a in arrows if "total" in a.label}
    assert tails["Lead"][0] == pytest.approx(0.0)
    assert tails["Trail"][0] == pytest.approx(0.1)


@pytest.mark.parametrize("view", PART_VIEWS)
def test_each_view_draws_only_its_parts(view: str) -> None:
    arrows = hand_arrows(_split(), STYLE, view, "force", ("lead",))
    expected = len(colour_key(view))
    assert len(arrows) == expected


def test_the_torque_view_draws_the_torques() -> None:
    arrows = hand_arrows(_split(), STYLE, "total", "torque", ("trail",))
    (arrow,) = arrows
    assert arrow.magnitude == pytest.approx(15.0)  # 5 + 10 N m about z


def test_a_gain_lengthens_without_tripping_the_length_limit() -> None:
    """Gain 8 on a 700 N total would pass 0.6 m; the limit grows with it."""
    from src.tools.native_viewer_export.ztcf_feed import gained_style

    arrows = hand_arrows(
        _split(), gained_style(STYLE, 8.0), "total", "force", ("trail",), gain=8.0
    )
    (arrow,) = arrows
    assert np.linalg.norm(np.subtract(arrow.tip_m, arrow.tail_m)) > STYLE.max_length_m


def test_the_legend_names_hands_view_and_quantity() -> None:
    text = legend_for(1.0, "zvcf", "torque", ("lead", "trail"))
    assert "Lead and trail" in text and "ZVCF" in text and "torque" in text
    assert "12x" in legend_for(12.0, "ztcf", "force", ("trail",))


def _frame(_index: int) -> ForceTorqueFrame:
    wrench = OverlayWrench(
        WrenchKind.CONTACT, "contact:foot", "foot", (0, 0, 0), (0, 0, 400.0), source="t"
    )
    return ForceTorqueFrame(0.0, "mujoco", (wrench,))


def test_the_feed_draws_its_colour_key_inside_the_key_box() -> None:
    from src.tools.native_viewer_export.compositor import colour_key_box

    key = colour_key("tip_to_tail")
    feed = OverlayFeed(_frame, STYLE, colour_key=key)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    out = feed.draw_key(img)
    changed = np.argwhere(np.any(out != img, axis=2))
    assert changed.size
    x0, y0, x1, y1 = colour_key_box(img.shape, key)
    assert changed[:, 0].min() >= y0 and changed[:, 0].max() < y1
    assert changed[:, 1].min() >= x0 and changed[:, 1].max() < x1
    assert OverlayFeed(_frame, STYLE).draw_key(img) is img
    assert feed.held_after(0).colour_key == key


SEEDS = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


@pytest.mark.requires_pinocchio
def test_live_both_hands_on_the_seed_swing() -> None:
    pytest.importorskip("pinocchio")
    spec = (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()
    doc = json.loads(spec)
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        q, v, tau = (rec[n][398:402] for n in ("q", "v", "tau"))
    bundle = SimpleNamespace(
        spec_bytes=spec,
        coordinate_order=tuple(doc["coordinate_order"]),
        dt_s=1.0 / 240.0,
        efforts=tau[:-1],
        reference_v=v,
    )
    swing = SimpleNamespace(bundle=bundle, q=q, v=v)
    style = ForceGlyphStyle(force_scale_m_per_n=1e-4, max_length_m=5.0)
    arrows = ztcf_zvcf_arrows(swing, style, view="tip_to_tail", hands=HANDS)(1)
    labels = _labels(arrows)
    assert any(lb.startswith("Lead ") for lb in labels)
    assert any(lb.startswith("Trail ") for lb in labels)


def test_each_hand_has_its_own_colours() -> None:
    """Review of the first render: both hands in one palette read as one hand."""
    arrows = hand_arrows(_split(), STYLE, "tip_to_tail", "force", HANDS)
    by_label = {a.label: a.rgba for a in arrows}
    for part in ("ztcf:drift", "zvcf:control", "total"):
        assert by_label[f"Lead {part}"] != by_label[f"Trail {part}"]


def test_the_colour_key_names_each_hand_in_its_colours() -> None:
    from src.tools.native_viewer_export.ztcf_feed import hands_colour_key

    key = hands_colour_key("tip_to_tail", HANDS)
    assert len(key) == 2 * len(colour_key("tip_to_tail"))
    texts = [text for text, _ in key]
    assert sum(t.startswith("Lead ") for t in texts) == 3
    assert sum(t.startswith("Trail ") for t in texts) == 3
    assert len({rgb for _, rgb in key}) == len(key)
    (only,) = hands_colour_key("zvcf", ("trail",))
    assert only[0].startswith("Trail ZVCF")


def test_hand_colours_come_from_the_settings_file(tmp_path) -> None:
    from src.tools.native_viewer_export.ztcf_feed import (
        HAND_COLOURS_PATH,
        hand_styles,
    )

    styles = hand_styles()
    assert set(styles) == set(HANDS)
    doc = json.loads(HAND_COLOURS_PATH.read_text(encoding="utf-8"))
    assert styles["trail"].drift_hex == doc["trail"]["drift_hex"]
    del doc["trail"]
    bad = tmp_path / "colours.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="trail"):
        hand_styles(bad)
