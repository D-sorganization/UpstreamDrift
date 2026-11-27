"""Planted feet in the export's ZTCF/ZVCF hand split (#15150 Track Z, Z5).

The split provider of the export re-solves the feet as reactions in each
part, like the hand-club closure, and reports where each foot's reaction
acts. ``split_glyphs`` evaluates any drawing of a state's split lazily.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.simulation_backends import hand_force_split
from src.shared.python.simulation_backends.hand_force_split import HANDS, PARTS
from src.tools.native_viewer_export import ztcf_feed

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
SEEDS = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
ROWS = slice(398, 402)


def _fake_swing(rows: int = 3) -> SimpleNamespace:
    bundle = SimpleNamespace(
        spec_bytes=json.dumps({"closure": {}}).encode(),
        coordinate_order=("a", "b"),
        dt_s=0.01,
        efforts=np.ones((rows - 1, 2)),
        reference_v=np.zeros((rows, 2)),
    )
    return SimpleNamespace(bundle=bundle, q=np.zeros((rows, 2)), v=np.zeros((rows, 2)))


@pytest.fixture
def fake_split(monkeypatch):
    """No engine: a stub provider, and a split that records the state row."""
    seen: list[float] = []

    def fake_provider(spec_bytes, order):
        none = lambda *_a: None  # noqa: E731
        return ztcf_feed.SplitProvider(None, none, none, none, none)

    def fake_hand_wrench_split(provider, frame_at, club_load, lead_point, q, v, tau):
        seen.append(float(q[0]))
        return ("split", float(q[0]))

    monkeypatch.setattr(ztcf_feed, "split_provider", fake_provider)
    monkeypatch.setattr(hand_force_split, "hand_wrench_split", fake_hand_wrench_split)
    return seen


def test_split_glyphs_draws_each_state_once_and_clamps_the_index(fake_split) -> None:
    swing = _fake_swing()
    swing.q[:, 0] = (10.0, 11.0, 12.0)
    calls: list[tuple] = []

    def draw(split):
        calls.append(split)
        return (split,)

    glyphs = ztcf_feed.split_glyphs(swing, draw)
    assert glyphs(1) == (("split", 11.0),)
    assert glyphs(1) is glyphs(1)
    assert glyphs(-5) == (("split", 10.0),)
    assert glyphs(99) == (("split", 12.0),)
    assert glyphs(2) is glyphs(99)
    assert len(calls) == 3 and fake_split == [11.0, 10.0, 12.0]


def test_split_glyphs_refuses_a_kinematic_replay(fake_split) -> None:
    swing = _fake_swing()
    swing.bundle.efforts = np.zeros((2, 2))
    with pytest.raises(ValueError, match="efforts"):
        ztcf_feed.split_glyphs(swing, lambda split: ())


def test_the_hand_arrows_are_a_drawing_of_the_split(fake_split, monkeypatch) -> None:
    def fake_arrows(split, style, view, quantity, hands, colours, gain):
        return (("arrow", split, view, quantity, tuple(hands), gain),)

    monkeypatch.setattr(ztcf_feed, "hand_arrows", fake_arrows)
    style = ForceGlyphStyle(force_scale_m_per_n=1e-3, max_length_m=1.0)
    arrows = ztcf_feed.ztcf_zvcf_arrows(
        _fake_swing(), style, view="zvcf", hands=("lead",), gain=2.0
    )
    assert arrows(0) == (("arrow", ("split", 0.0), "zvcf", "force", ("lead",), 2.0),)
    assert {"split_glyphs", "split_provider"} <= set(ztcf_feed.__all__)


@pytest.fixture(scope="module")
def seed():
    pytest.importorskip("pinocchio")
    spec = (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()
    order = tuple(json.loads(spec)["coordinate_order"])
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        q, v, tau = (rec[n][ROWS].copy() for n in ("q", "v", "tau"))
    return ztcf_feed.split_provider(spec, order), q, v, tau


@pytest.mark.requires_pinocchio
def test_the_split_provider_reports_where_the_feet_react(seed) -> None:
    split, q, _, _ = seed
    assert isinstance(split, ztcf_feed.SplitProvider)
    for row in q:
        points = split.ground_points(row)
        assert tuple(body for body, _ in points) == ("calcn_r", "calcn_l")
        for _, point in points:
            assert np.shape(point) == (3,) and np.all(np.isfinite(point))


@pytest.mark.requires_pinocchio
def test_the_grounded_parts_still_add_to_the_total(seed) -> None:
    """Each hand's ZTCF plus ZVCF is its load under ``tau``, feet planted."""
    from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
        constrained_acceleration_at,
    )

    split, q, v, tau = seed
    provider = split.provider
    assert provider.closure_jacobian(q[0]).shape[0] > 6  # closure + feet
    for k in range(q.shape[0]):
        parts = hand_force_split.hand_wrench_split(
            provider,
            split.frame_at,
            split.club_load,
            split.lead_point,
            q[k],
            v[k],
            tau[k],
        )
        a, lam = constrained_acceleration_at(provider, q[k], v[k], tau[k])
        rot, _ = split.frame_at(q[k])
        trail = -np.asarray(rot) @ lam[:3]
        club_force, _ = split.club_load(q[k], v[k], a, True)
        expected = {"trail": trail, "lead": np.asarray(club_force) - trail}
        for hand in HANDS:
            total = sum(parts.part(hand, p).force_n[0] for p in PARTS)
            np.testing.assert_allclose(total, expected[hand], rtol=1e-7, atol=1e-6)
