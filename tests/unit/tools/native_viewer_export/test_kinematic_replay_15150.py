"""A kinematic replay shows no forces derived from efforts or contact (#15150)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.tools.native_viewer_export import cli, kinematic_replay as kr
from src.tools.native_viewer_export.overlay import OverlayOptions

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[4]


def _bundle(efforts: np.ndarray, holds: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(efforts=efforts, hold_efforts=holds)


def test_only_all_zero_efforts_make_a_kinematic_replay() -> None:
    assert kr.is_kinematic_replay(_bundle(np.zeros((5, 3))))
    efforts = np.zeros((5, 3))
    efforts[4, 2] = 1e-9
    assert not kr.is_kinematic_replay(_bundle(efforts))
    holds = np.zeros((10, 3))
    holds[3, 0] = 0.5  # an odd row is not a step's first hold, so not in efforts
    assert not kr.is_kinematic_replay(_bundle(np.zeros((5, 3)), holds))
    assert kr.is_kinematic_replay(_bundle(np.zeros((5, 3)), np.zeros((10, 3))))
    with pytest.raises(ValueError, match="efforts"):
        kr.is_kinematic_replay(_bundle(np.zeros((0, 3))))


def test_a_provenance_label_does_not_decide() -> None:
    labelled = _bundle(np.ones((2, 3)))
    labelled.provenance = {"kinematic": True}
    assert not kr.is_kinematic_replay(labelled)


def test_vectors_are_hidden_on_a_replay_unless_requested(caplog) -> None:
    kin, dyn = _bundle(np.zeros((3, 2))), _bundle(np.ones((3, 2)))
    with caplog.at_level(logging.WARNING):
        assert kr.vector_visibility(kin, OverlayOptions()) == (False, False, True, True)
    assert sum("kinematic replay" in r.getMessage() for r in caplog.records) == 1
    asked = OverlayOptions(show_torque=False, forces_requested=True)
    assert kr.vector_visibility(kin, asked) == (True, False, False, True)
    assert kr.vector_visibility(dyn, OverlayOptions(show_force=False)) == (
        False,
        True,
        False,
        False,
    )
    assert kr.NOTE == "Kinematic Replay: Forces Not Shown"


def _settings(*flags: str):
    return cli.build_settings(
        cli.build_parser().parse_args(
            ["--bundle", "x", "--out", "o", "--swing", "s", *flags]
        )
    )


def test_only_an_explicit_vector_flag_requests_vectors_on_a_replay() -> None:
    assert not _settings().vectors_requested
    assert _settings("--force-vectors").vectors_requested
    assert _settings("--torque-vectors").vectors_requested
    assert not _settings("--no-force-vectors").vectors_requested
    assert (
        _settings().force_vectors and not _settings("--no-force-vectors").force_vectors
    )


UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
STATES = 9


@pytest.fixture(scope="module")
def zero_effort_swing():
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.full_body_spec import load_full_body_spec
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import SwingInput

    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    spec_bytes = json.dumps(spec).encode("utf-8")
    order = tuple(spec["coordinate_order"])
    end = np.random.default_rng(15150).normal(0.0, 0.2, len(order))
    q = np.linspace(0.0, 1.0, STATES)[:, None] * end[None, :]
    v = np.gradient(q, 1.0 / 360.0, axis=0)
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=order,
        dt_s=1.0 / 360.0,
        q0=q[0],
        v0=v[0],
        efforts=np.zeros((STATES - 1, len(order))),
        reference_q=q,
        reference_v=v,
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "sweep", "Driver", "mujoco", v=v)


def _drawn(feed, index: int) -> int:
    glyphs = feed.glyphs_at(index)
    return len(glyphs.arrows) + len(glyphs.torque_arcs)


def test_a_kinematic_bundle_draws_no_force_or_torque_glyphs(zero_effort_swing) -> None:
    from src.tools.native_viewer_export.overlay import build_overlay_feed

    feed, _ = build_overlay_feed(zero_effort_swing, "mujoco")
    assert all(_drawn(feed, k) == 0 for k in range(STATES))
    assert feed.hud_lines_at(0) == (kr.NOTE,)
    asked, _ = build_overlay_feed(
        zero_effort_swing, "mujoco", options=OverlayOptions(forces_requested=True)
    )
    assert asked.hud_lines_at(0) == (kr.NOTE_DRAWN,)
    assert any(_drawn(asked, k) > 0 for k in range(STATES))  # the control


def test_ellipsoids_stay_on_a_kinematic_replay(zero_effort_swing) -> None:
    from src.tools.native_viewer_export.overlay import build_overlay_feed

    opts = OverlayOptions(hand_mobility="both")
    feed, _ = build_overlay_feed(zero_effort_swing, "mujoco", options=opts)
    assert len(feed.glyphs_at(0).ellipsoids) >= 2
    assert feed.hud_lines_at(0)[-1] == kr.NOTE and len(feed.hud_lines_at(0)) == 3
