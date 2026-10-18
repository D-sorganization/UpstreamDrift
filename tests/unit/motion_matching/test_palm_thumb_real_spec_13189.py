"""Palm-on-thumb allocation on the real full-body spec (HRG-H1, #13189).

Reporting only: the compiled physics and the physics spec hash are identical
before and after the allocation, and the allocation reproduces the weld
multiplier exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.overlay_source import (  # noqa: E402
    MujocoOverlaySource,
)
from src.shared.python.core.handedness import subject_handedness  # noqa: E402
from src.shared.python.grip_contact.palm_thumb import (  # noqa: E402
    allocate_from_closure,
)
from src.shared.python.model_appearance import club_assembly as ca  # noqa: E402
from src.shared.python.model_appearance.grip_contact_patch import (  # noqa: E402
    thumb_palm_patch,
)
from src.shared.python.model_appearance.grip_hands import (  # noqa: E402
    grip_hand_frame,
    grip_radius_at,
)
from src.shared.python.model_appearance.grip_pose import (  # noqa: E402
    DEFAULT_GRIP_POSE,
    Hand,
)
from src.shared.python.model_appearance.schema import (  # noqa: E402
    physics_spec_sha256,
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
LAMBDA_TOL = 1e-10
THIRD_LAW_TOL = 1e-9
STATES = 24


def _physics_digest(model) -> str:
    h = hashlib.sha256()
    for name in ("body_mass", "body_inertia", "eq_data"):
        h.update(np.ascontiguousarray(getattr(model, name)).tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def patch(spec):
    club = ca.assembly_from_spec(spec)
    pose = DEFAULT_GRIP_POSE
    radius = grip_radius_at(club, pose.distance_below_butt_m(Hand.LEAD))
    return thumb_palm_patch(
        grip_hand_frame(club, Hand.LEAD, pose),
        grip_radius_m=radius,
        pose=pose,
        handedness=subject_handedness(spec),
    )


def _state(source, rng):
    names = source.coordinate_order
    q = dict(zip(names, rng.normal(0.0, 0.05, len(names)), strict=True))
    v = dict(zip(names, rng.normal(0.0, 0.3, len(names)), strict=True))
    e = dict(zip(names, rng.normal(0.0, 20.0, len(names)), strict=True))
    return q, v, e


def test_allocation_reproduces_lambda_and_leaves_the_physics_untouched(
    spec, patch
) -> None:
    source = MujocoOverlaySource(json.dumps(spec).encode("utf-8"))
    digest = _physics_digest(source.model)
    spec_hash = physics_spec_sha256(spec)
    assert source.closing_hand_side == patch.trail_side
    rng = np.random.default_rng(13189)
    mu = float(spec["contact"]["parameters"]["static_friction"])
    shares = []
    for _ in range(STATES):
        q, v, e = _state(source, rng)
        closure = source.closure_wrench(q, v, e)
        rot, origin = source.club_frame_world(q)
        out = allocate_from_closure(
            closure,
            club_rotation=rot,
            club_origin=origin,
            patch=patch,
            static_friction=mu,
        )
        club = closure.wrenches[1]
        moved = np.array(out.couple_nm) + np.cross(
            np.array(out.patch_point_m) - np.array(club.point_m), out.force_on_lead_n
        )
        assert np.abs(np.array(out.force_on_lead_n) - club.force_n).max() <= LAMBDA_TOL
        assert np.abs(moved - club.torque_nm).max() <= LAMBDA_TOL
        assert (
            np.abs(np.add(out.force_on_lead_n, out.force_on_trail_n)).max()
            <= THIRD_LAW_TOL
        )
        assert out.method == "allocation"
        shares.append(out.share_of_closure_force)
    assert all(0.0 <= s <= 1.0 for s in shares)
    assert _physics_digest(source.model) == digest
    assert physics_spec_sha256(spec) == spec_hash


def test_missing_efforts_make_the_allocation_unavailable(spec, patch) -> None:
    source = MujocoOverlaySource(json.dumps(spec).encode("utf-8"))
    q, v, _ = _state(source, np.random.default_rng(1))
    closure = source.closure_wrench(q, v, None)
    rot, origin = source.club_frame_world(q)
    out = allocate_from_closure(
        closure,
        club_rotation=rot,
        club_origin=origin,
        patch=patch,
        static_friction=0.9,
    )
    assert not out.available and out.reason and out.normal_force_n is None


def test_club_frame_maps_the_closure_placement_onto_the_weld_site(spec) -> None:
    """The spec club frame is not the MJCF body frame: placement_b must land on
    the weld site, and the patch must sit within a hand's reach of the grip."""
    source = MujocoOverlaySource(json.dumps(spec).encode("utf-8"))
    q, _, _ = _state(source, np.random.default_rng(7))
    rot, origin = source.club_frame_world(q)
    placement = np.asarray(spec["closure"]["placement_b"], dtype=float)
    site = source.data.site_xpos[source._closure[1]]
    np.testing.assert_allclose(rot @ placement[:3, 3] + origin, site, atol=1e-9)
    np.testing.assert_allclose(
        rot @ placement[:3, :3],
        source.data.site_xmat[source._closure[1]].reshape(3, 3),
        atol=1e-9,
    )


def test_feed_patch_helper_matches_the_spec_patch_and_friction(spec, patch) -> None:
    from src.tools.native_viewer_export.overlay import palm_thumb_patch_for_spec

    got, friction = palm_thumb_patch_for_spec(json.dumps(spec).encode("utf-8"))
    assert got == patch
    assert friction == spec["contact"]["parameters"]["static_friction"]
    with pytest.raises(ValueError, match="static_friction"):
        broken = json.loads(json.dumps(spec))
        del broken["contact"]["parameters"]["static_friction"]
        palm_thumb_patch_for_spec(json.dumps(broken).encode("utf-8"))
