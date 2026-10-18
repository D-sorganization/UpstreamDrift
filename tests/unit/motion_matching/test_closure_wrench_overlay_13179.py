"""Grip closure wrench and ground free moment on the real full-body spec (#13179).

The efforts come first: given ``efforts`` the explicit KKT solve fixes the
physical weld multiplier and the contact generalized force. Injecting both into
the prescribed-motion inverse must return the same efforts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.overlay_source import (  # noqa: E402
    MujocoOverlaySource,
)
from src.shared.python.body_part_viz.mujoco_axial_loads import (  # noqa: E402
    BodyWrench,
    GripClosure,
    MujocoAxialLoadSource,
)
from src.shared.python.force_overlay.bundle_provider import (  # noqa: E402
    ClosureAxialLoadSource,
    contact_sample_wrenches,
    sphere_bodies,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    load_full_body_spec,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
TOLERANCE_NM = 1e-6


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    return json.dumps(spec).encode("utf-8")


@pytest.fixture
def case(spec_bytes: bytes):
    source = MujocoOverlaySource(spec_bytes)
    source._torsion_patch_m = 0.06  # exercise the ground free moment
    rng = np.random.default_rng(13179)
    names = source.coordinate_order
    q = dict(zip(names, rng.normal(0.0, 0.05, len(names)), strict=True))
    for name in names:
        if name.startswith("Translation") and name.endswith("Z"):
            q[name] = -0.05  # press the feet into the ground
    v = dict(zip(names, rng.normal(0.0, 0.3, len(names)), strict=True))
    e = dict(zip(names, rng.normal(0.0, 20.0, len(names)), strict=True))
    accel, _ = source.solve_with_multipliers(q, v, e)
    return source, q, v, e, accel, sphere_bodies(json.loads(spec_bytes))


def _site_centre_wrenches(source, q, v, bodies):
    """Ground wrenches on the forward model's own lines of action, with moment."""
    samples = source.evaluate_contact_samples(q, v)
    moments = source.contact_free_moments(q, v)
    source.generalized_forces(q, v)  # leave site positions at this state
    out = []
    for name, moment in moments.items():
        site = source._spheres[name]["site_id"]
        sample = samples[name]
        out.append(
            BodyWrench(
                bodies[name],
                tuple(source.data.site_xpos[site]),
                tuple(sample.normal_force_n + sample.friction_force_n),
                tuple(moment),
            )
        )
    return out


def test_source_offers_the_closure_accessor(case) -> None:
    source, q, v, e, _, _ = case
    assert isinstance(source, ClosureAxialLoadSource)
    closure = source.closure_wrench(q, v, e)
    assert closure.available and len(closure.multiplier_world) == 6
    on_hand, on_club = closure.wrenches
    assert np.abs(np.add(on_hand.force_n, on_club.force_n)).max() <= 1e-9
    assert np.abs(np.add(on_hand.torque_nm, on_club.torque_nm)).max() <= 1e-9


def test_missing_or_incomplete_efforts_make_the_closure_unavailable(case) -> None:
    source, q, v, e, _, _ = case
    none = source.closure_wrench(q, v, None)
    assert not none.available and none.multiplier_world is None and none.reason
    partial = source.closure_wrench(q, v, {k: e[k] for k in list(e)[:3]})
    assert not partial.available and "closure solve failed" in partial.reason


def test_inverse_with_closure_and_free_moment_returns_the_efforts(case) -> None:
    source, q, v, e, accel, bodies = case
    ground = _site_centre_wrenches(source, q, v, bodies)
    assert ground and any(np.abs(w.torque_nm).max() > 0 for w in ground)
    closure = source.closure_wrench(q, v, e)
    axial = MujocoAxialLoadSource(source.model, axis_policy="kinematic")
    vec = source._vector
    required = axial.required_generalized_force(
        vec(q), vec(v), vec(accel), ground, closure
    )
    assert np.abs(required - vec(e)).max() <= TOLERANCE_NM
    # The fix bites: the same inverse without the closure is far from the efforts.
    open_loop = axial.required_generalized_force(
        vec(q), vec(v), vec(accel), ground, GripClosure.unavailable("test")
    )
    assert np.abs(open_loop - vec(e)).max() > 1.0


def test_free_moment_is_not_dropped_from_the_ground_wrench(case) -> None:
    source, q, v, _, _, bodies = case
    samples = source.evaluate_contact_samples(q, v)
    moments = source.contact_free_moments(q, v)
    assert moments
    with_moment = contact_sample_wrenches(
        samples, bodies, source="t", free_moments=moments
    )
    assert all(w.torque_nm is not None for w in with_moment)


def test_no_torsion_patch_gives_no_free_moment(spec_bytes: bytes) -> None:
    source = MujocoOverlaySource(spec_bytes)
    source._torsion_patch_m = 0.0
    q = dict.fromkeys(source.coordinate_order, 0.0)
    assert source.contact_free_moments(q, dict(q)) == {}


def test_frame_source_marks_closure_status_on_real_spec(case) -> None:
    source, q, v, e, accel, bodies = case
    frame = source.axial_load_frame(
        q,
        v,
        accel,
        0.0,
        _site_centre_wrenches(source, q, v, bodies),
        closure=source.closure_wrench(q, v, e),
    )
    assert "grip closure wrench applied" in frame.source
    blank = source.axial_load_frame(q, v, accel, 0.0, [], closure=None)
    assert "grip closure unavailable" in blank.source
    assert any(value is None for value in blank.values_n.values())


def test_overlay_feed_blanks_axial_loads_at_ball_impact(spec_bytes: bytes) -> None:
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import SwingInput
    from src.tools.native_viewer_export.overlay import build_overlay_feed

    names = tuple(json.loads(spec_bytes)["coordinate_order"])
    q = np.zeros((4, len(names)))
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((3, len(names))),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    swing = SwingInput(bundle, q, "t", "Driver", "mujoco")
    feed, _ = build_overlay_feed(swing, "mujoco", axial=True, impact_time_s=0.002)
    before, at = feed.frame_at(0), feed.frame_at(2)
    assert any(v is not None for v in before.axial_loads.values_n.values())
    assert before.metadata["axial_closure"] == "applied"
    assert all(v is None for v in at.axial_loads.values_n.values())
    assert at.metadata["axial_impact"] == "unavailable"
