"""Engine-agnostic grip-style contact topology tests (HRG-GS-G1, #13299)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.contact_topology import (
    ALLOCATED_HAND_GRIP_MODES,
    ALLOCATED_HAND_HAND_MODES,
    DEFAULT_HAND_LENGTH_M,
    GRIP,
    GRIP_RADIUS_M,
    PHYSICAL_HAND_GRIP_MODES,
    PHYSICAL_HAND_HAND_MODES,
    ContactPair,
    ContactSite,
    ContactTopology,
    GripStyle,
    HandAnatomy,
    contact_topology,
    pair_surface_gap_m,
    site_grip_surface_gap_m,
)
from src.shared.python.model_appearance.grip_pose import GripStyle as PoseGripStyle
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.motion_matching.execution.grip_rotation import (
    apply_grip_config,
    grip_spec_of,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    GripStyle as ContractGripStyle,
    grip_spec_to_physical,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
DRIVER_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
ROUNDTRIP_FIXTURE = ROOT / "tests/fixtures/grip/grip_roundtrip.json"

BASE_PAIRS: tuple[tuple[str, str, str, tuple[str, str]], ...] = (
    ("lead_palm_proximal_grip", "lead_palm_proximal", GRIP, ("sphere", "cylinder")),
    ("lead_palm_distal_grip", "lead_palm_distal", GRIP, ("sphere", "cylinder")),
    ("lead_finger_proximal_grip", "lead_finger_proximal", GRIP, ("sphere", "cylinder")),
    ("lead_finger_distal_grip", "lead_finger_distal", GRIP, ("sphere", "cylinder")),
    ("trail_palm_proximal_grip", "trail_palm_proximal", GRIP, ("sphere", "cylinder")),
    ("trail_palm_distal_grip", "trail_palm_distal", GRIP, ("sphere", "cylinder")),
    (
        "trail_finger_proximal_grip",
        "trail_finger_proximal",
        GRIP,
        ("sphere", "cylinder"),
    ),
    ("trail_finger_distal_grip", "trail_finger_distal", GRIP, ("sphere", "cylinder")),
    ("lead_heel_pad_butt", "lead_heel_pad", GRIP, ("sphere", "cylinder")),
    ("trail_palm_lead_thumb", "trail_palm_pocket", "lead_thumb", ("sphere", "capsule")),
)

STYLE_SPECIFIC_PAIRS: dict[
    GripStyle, tuple[tuple[str, str, str, tuple[str, str]], ...]
] = {
    GripStyle.OVERLAP: (
        (
            "overlap_little_finger_lead_index",
            "trail_little_finger",
            "lead_index",
            ("capsule", "capsule"),
        ),
    ),
    GripStyle.INTERLOCK: (
        (
            "interlock_little_finger_lead_middle",
            "trail_little_finger",
            "lead_middle",
            ("capsule", "capsule"),
        ),
        (
            "interlock_lead_index_wrap",
            "lead_index",
            "trail_little_finger",
            ("capsule", "capsule"),
        ),
    ),
    GripStyle.TEN_FINGER: (
        (
            "ten_finger_little_finger_grip",
            "trail_little_finger",
            GRIP,
            ("capsule", "cylinder"),
        ),
    ),
}

TRIGGER_PAIR: tuple[str, str, str, tuple[str, str]] = (
    "trail_index_trigger_grip",
    "trail_index_trigger",
    GRIP,
    ("capsule", "cylinder"),
)


def _antagonist_pair(site: str) -> tuple[str, str, str, tuple[str, str]]:
    return (f"{site}_grip", site, GRIP, ("sphere", "cylinder"))


def test_grip_style_has_exactly_one_definition_in_src() -> None:
    """GripStyle enum has a single definition under src/ and is re-exported cleanly."""
    pattern = re.compile(r"^\s*class\s+GripStyle\b", re.MULTILINE)
    defining_files = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(SRC_DIR.rglob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert defining_files == ["src/shared/python/grip_contact/contact_topology.py"]
    assert PoseGripStyle is GripStyle
    assert ContractGripStyle is GripStyle
    assert tuple(s.value for s in GripStyle) == ("overlap", "interlock", "ten_finger")


@pytest.mark.parametrize("style", list(GripStyle))
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_pair_counts_and_exact_pair_sets_per_style(
    style: GripStyle, trigger_finger: bool
) -> None:
    """Overlap, interlock, and ten-finger produce their documented pair set exactly."""
    spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = contact_topology(spec)
    assert isinstance(topo, ContactTopology)
    assert topo.style is style
    assert topo.trigger_finger is trigger_finger

    expected = list(BASE_PAIRS) + list(STYLE_SPECIFIC_PAIRS[style])
    if trigger_finger:
        expected.append(TRIGGER_PAIR)
    # Antagonists opposing the unpaired heel, trigger and ten-finger pads (#13555).
    expected.append(_antagonist_pair("lead_butt_fingers"))
    if trigger_finger:
        expected.append(_antagonist_pair("trail_thumb_pad"))
    if style is GripStyle.TEN_FINGER:
        expected.append(_antagonist_pair("trail_hypothenar"))

    actual = [
        (pair.name, pair.site_a, pair.site_b, pair.primitive_pair)
        for pair in topo.pairs
    ]
    assert actual == expected

    for pair in topo.pairs:
        # Every referenced site_a (and non-GRIP site_b) exists in topo.sites
        site_a = topo.site(pair.site_a)
        assert site_a.primitive == pair.primitive_pair[0]
        if pair.site_b == GRIP:
            assert pair.primitive_pair[1] == "cylinder"
            assert pair.physical_modes == PHYSICAL_HAND_GRIP_MODES
            assert pair.allocated_modes == ALLOCATED_HAND_GRIP_MODES
        else:
            site_b = topo.site(pair.site_b)
            assert site_b.primitive == pair.primitive_pair[1]
            assert pair.physical_modes == PHYSICAL_HAND_HAND_MODES
            assert pair.allocated_modes == ALLOCATED_HAND_HAND_MODES


def test_ten_finger_moves_trail_little_finger_to_abut_lead_index() -> None:
    """Ten-finger places the trail little finger on the grip abutting lead index."""
    overlap_topo = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    ten_topo = contact_topology(GripSpec(style=GripStyle.TEN_FINGER))

    ov_little = overlap_topo.site("trail_little_finger")
    ten_little = ten_topo.site("trail_little_finger")
    ten_lead_idx = ten_topo.site("lead_index")

    # In ten-finger, trail little finger is further down the shaft (+x) than in overlap
    assert ten_little.centre_m[0] > ov_little.centre_m[0]
    assert ten_little.centre_m[0] > ten_lead_idx.centre_m[0]
    # And sits directly on the grip surface while abutting lead_index within 0.5 mm
    assert abs(site_grip_surface_gap_m(ten_little, ten_topo.grip_radius_m)) <= 5.0e-4
    dist_idx_little = float(
        np.linalg.norm(
            np.asarray(ten_little.centre_m) - np.asarray(ten_lead_idx.centre_m)
        )
    )
    assert (
        abs(dist_idx_little - (ten_little.radius_m + ten_lead_idx.radius_m)) <= 5.0e-4
    )


@pytest.mark.parametrize("style", list(GripStyle))
@pytest.mark.parametrize("trigger_finger", [False, True])
@pytest.mark.parametrize("hand_length_m", [0.14, DEFAULT_HAND_LENGTH_M, 0.24])
def test_every_site_lies_on_or_within_one_pad_radius_at_neutral_pose(
    style: GripStyle, trigger_finger: bool, hand_length_m: float
) -> None:
    """Every site lies on or within one pad radius of grip or partner site (<= 0.5 mm)."""
    spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = contact_topology(spec, anatomy=hand_length_m)

    # 1. Every documented contact pair has surface gap/penetration within 0.5 mm
    for pair in topo.pairs:
        gap = pair_surface_gap_m(topo, pair)
        assert -5.0e-4 <= gap <= 5.0e-4, f"{pair.name} gap={gap}"

    # 2. Every site in topology.sites lies on or within one pad radius of the grip
    # surface OR a partner site's surface to within 0.5 mm.
    for site in topo.sites:
        grip_radial_dist = (
            math.hypot(site.centre_m[1], site.centre_m[2]) - topo.grip_radius_m
        )
        grip_surface_gap = abs(grip_radial_dist - site.radius_m)
        on_grip = (0.0 < grip_radial_dist <= site.radius_m + 1e-12) and (
            grip_surface_gap <= 5.0e-4
        )

        on_partner = False
        for other in topo.sites:
            if other.name == site.name:
                continue
            center_dist = float(
                np.linalg.norm(np.asarray(site.centre_m) - np.asarray(other.centre_m))
            )
            dist_to_other_surface = center_dist - other.radius_m
            partner_gap = abs(dist_to_other_surface - site.radius_m)
            if (0.0 < dist_to_other_surface <= site.radius_m + 1e-12) and (
                partner_gap <= 5.0e-4
            ):
                on_partner = True
                break

        assert on_grip or on_partner, (
            f"site {site.name} ({style.value}) neither on grip "
            f"(gap={grip_surface_gap:.6f}) nor on partner"
        )


@pytest.mark.parametrize("hand_length_m", [0.14, 0.165, 0.19, 0.215, 0.24])
def test_anatomy_scaling_scales_radii_proportionally(hand_length_m: float) -> None:
    """Site radii scale linearly with L / 0.19 for little, index, thumb, and heel pad."""
    spec = GripSpec(style=GripStyle.OVERLAP, trigger_finger=True)
    topo = contact_topology(spec, anatomy=HandAnatomy(hand_length_m=hand_length_m))
    scale = hand_length_m / 0.19

    assert topo.site("trail_little_finger").radius_m == pytest.approx(
        0.0075 * scale, rel=1e-12
    )
    assert topo.site("lead_index").radius_m == pytest.approx(0.008 * scale, rel=1e-12)
    assert topo.site("lead_middle").radius_m == pytest.approx(0.008 * scale, rel=1e-12)
    assert topo.site("trail_index_trigger").radius_m == pytest.approx(
        0.008 * scale, rel=1e-12
    )
    assert topo.site("lead_thumb").radius_m == pytest.approx(0.0095 * scale, rel=1e-12)
    assert topo.site("lead_heel_pad").radius_m == pytest.approx(
        0.012 * scale, rel=1e-12
    )


@pytest.mark.parametrize(
    "bad_length", [0.1399, 0.2401, 0.0, -0.19, float("nan"), float("inf")]
)
def test_anatomy_rejects_hand_length_outside_bounds(bad_length: float) -> None:
    """Hand length L outside [0.14, 0.24] m raises ValueError."""
    with pytest.raises(ValueError, match="hand_length_m"):
        HandAnatomy(hand_length_m=bad_length)
    with pytest.raises(ValueError, match="hand_length_m"):
        contact_topology(GripSpec(), anatomy=bad_length)


def test_v1_default_spec_bytes_unchanged_and_non_default_style_creates_revision() -> (
    None
):
    """v1 default spec bytes are unchanged; non-default style/trigger mints a revision."""
    raw_bytes = DRIVER_SPEC.read_bytes()
    base_doc = json.loads(raw_bytes.decode("utf-8"))
    fixture = json.loads(ROUNDTRIP_FIXTURE.read_text(encoding="utf-8"))

    # 1. v1 file bytes on disk and canonical hash are untouched
    assert (
        physics_spec_sha256(base_doc) == fixture["parent"]["driver_physics_spec_sha256"]
    )
    assert DRIVER_SPEC.read_bytes() == raw_bytes

    # 2. Default GripSpec() (OVERLAP, trigger_finger=False) has identical physical JSON
    default_spec = GripSpec()
    assert default_spec.style is GripStyle.OVERLAP
    assert default_spec.trigger_finger is False
    assert grip_spec_to_physical(default_spec) == {
        "lead": {"alpha_deg": 0.0, "psi_deg": 40.0, "gamma_deg": 0.0},
        "trail": {"alpha_deg": 0.0, "psi_deg": 40.0, "gamma_deg": 0.0},
        "lead_below_butt_m": 0.032,
        "hand_spacing_m": 0.0762,
        "handedness": "right",
    }

    # Presets produce a valid revision recording the parent's physics hash
    weak_doc = apply_grip_config(base_doc, GripSpec.preset("weak"))
    assert (
        weak_doc["spec_revision"]["parent_physics_spec_sha256"]
        == (fixture["parent"]["driver_physics_spec_sha256"])
    )

    # 3. Non-default style produces a new spec revision through apply_grip_config
    overlap_doc = apply_grip_config(base_doc, default_spec)
    interlock_doc = apply_grip_config(
        overlap_doc, replace(default_spec, style=GripStyle.INTERLOCK)
    )
    assert interlock_doc["spec_revision"]["version"] == 2
    assert interlock_doc["spec_revision"]["parent_spec_canonical_sha256"] == (
        canonical_sha256(overlap_doc)
    )
    assert physics_spec_sha256(interlock_doc) != physics_spec_sha256(overlap_doc)
    assert grip_spec_of(interlock_doc).style is GripStyle.INTERLOCK

    ten_finger_doc = apply_grip_config(
        base_doc, replace(default_spec, style=GripStyle.TEN_FINGER)
    )
    assert ten_finger_doc["spec_revision"]["version"] == 1
    assert physics_spec_sha256(ten_finger_doc) != physics_spec_sha256(overlap_doc)
    assert grip_spec_of(ten_finger_doc).style is GripStyle.TEN_FINGER

    # 4. trigger_finger=True also produces a new spec revision and round-trips
    trigger_doc = apply_grip_config(
        overlap_doc, replace(default_spec, trigger_finger=True)
    )
    assert trigger_doc["spec_revision"]["version"] == 2
    assert physics_spec_sha256(trigger_doc) != physics_spec_sha256(overlap_doc)
    assert grip_spec_of(trigger_doc).trigger_finger is True


def test_dbc_preconditions_reject_invalid_inputs() -> None:
    """DbC preconditions reject bad types and out-of-range values."""
    with pytest.raises(TypeError):
        HandAnatomy(hand_length_m=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        contact_topology("not_a_spec")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        contact_topology(GripSpec(), anatomy="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GripSpec(style="overlap")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GripSpec(trigger_finger=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hand"):
        ContactSite(
            name="bad",
            hand="middle",
            segment="thumb",
            primitive="sphere",
            centre_m=(0.0, 0.0, 0.02),
            radius_m=0.008,
        )
    with pytest.raises(ValueError, match="primitive"):
        ContactSite(
            name="bad",
            hand="lead",
            segment="thumb",
            primitive="box",
            centre_m=(0.0, 0.0, 0.02),
            radius_m=0.008,
        )
    with pytest.raises(ValueError, match="radius_m"):
        ContactSite(
            name="bad",
            hand="lead",
            segment="thumb",
            primitive="sphere",
            centre_m=(0.0, 0.0, 0.02),
            radius_m=0.0,
        )
    with pytest.raises(ValueError, match="disjoint"):
        ContactPair(
            name="bad_pair",
            site_a="lead_thumb",
            site_b=GRIP,
            primitive_pair=("capsule", "cylinder"),
            physical_modes=("contact",),
            allocated_modes=("contact",),
        )
