"""Allocated hand-hand contact reporting for all style pairs (HRG-GS-G3, #13301)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.mujoco_axial_loads import GripClosure
from src.shared.python.force_overlay.bundle_provider import BundleOverlayProvider
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.palm_thumb import (
    METHOD,
    PALM_THUMB_LABEL,
    AllocatedHandContacts,
    AllocatedPairForce,
    PalmThumbForce,
    allocate_from_closure,
    allocate_hand_hand_from_closure,
    allocate_pair_force,
    allocate_palm_thumb,
    pair_contact_patch,
    unavailable_pair_force,
    unavailable_palm_thumb,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.grip_contract import GripSpec
from src.tools.native_viewer_export.core import OverlayFeed
from tests.unit.force_overlay.test_axial_closure_provider import _Kin, _bundle

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

LAMBDA_TOL = 1e-9
THIRD_LAW_TOL = 1e-9


class _SyntheticSource:
    closing_hand_side = "trail"

    def __init__(
        self,
        closure: GripClosure | None = None,
        rotation: np.ndarray | None = None,
        origin: np.ndarray | None = None,
    ) -> None:
        self._closure = closure
        self._rotation = np.eye(3) if rotation is None else rotation
        self._origin = np.zeros(3) if origin is None else origin

    def closure_wrench(self, coordinates, rates, primitive_efforts):
        if self._closure is None:
            return GripClosure.from_multiplier(
                body_a="trail_hand",
                body_b="club",
                point_a_m=(0.0, 0.0, 0.0),
                point_b_m=(0.02, -0.01, 0.05),
                multiplier=(0.0, 40.0, 20.0, 1.5, -0.8, 2.2),
            )
        return self._closure

    def club_frame_world(self, coordinates):
        return self._rotation, self._origin


def _synthetic_closure(
    f: tuple[float, float, float] = (10.0, -40.0, 25.0),
    m: tuple[float, float, float] = (1.5, -0.8, 2.2),
    site: tuple[float, float, float] = (0.02, -0.01, 0.05),
) -> GripClosure:
    return GripClosure.from_multiplier(
        body_a="trail_hand",
        body_b="club",
        point_a_m=(0.0, 0.0, 0.0),
        point_b_m=site,
        multiplier=(*f, *m),
    )


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_pair_contact_patch_computes_finite_patch_in_grip_frame(
    style: GripStyle,
) -> None:
    """pair_contact_patch returns finite patch point and unit normal from lead to trail."""
    topo = contact_topology(GripSpec(style=style))
    hand_hand_pairs = topo.hand_hand_pairs()
    assert len(hand_hand_pairs) >= 1

    for pair in hand_hand_pairs:
        pt, normal = pair_contact_patch(topo, pair)
        assert isinstance(pt, np.ndarray) and pt.shape == (3,)
        assert isinstance(normal, np.ndarray) and normal.shape == (3,)
        assert np.isfinite(pt).all()
        assert np.isfinite(normal).all()
        assert abs(float(np.linalg.norm(normal)) - 1.0) <= 1e-9


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_sum_of_allocated_forces_reproduces_closure_lambda_without_hand_club_forces(
    style: GripStyle,
) -> None:
    """The sum of allocated hand-hand forces reproduces closure lambda to <= 1e-9 N."""
    topo = contact_topology(GripSpec(style=style))
    rng = np.random.default_rng(13301)

    for _ in range(10):
        f = tuple(float(x) for x in rng.normal(0, 50, 3))
        m = tuple(float(x) for x in rng.normal(0, 5, 3))
        site = (0.03, -0.02, 0.08)
        closure = _synthetic_closure(f, m, site)

        out = allocate_hand_hand_from_closure(
            closure,
            club_rotation=np.eye(3),
            club_origin=np.zeros(3),
            topology=topo,
            static_friction=0.9,
        )

        assert out.available is True
        assert out.method == "allocation"
        assert len(out.pairs) == len(topo.hand_hand_pairs())

        # Sum of forces on lead hand
        sum_f_lead = np.zeros(3)
        for pair_force in out:
            assert pair_force.force_on_lead_n is not None
            sum_f_lead += np.asarray(pair_force.force_on_lead_n)

            # Newton's third law on each pair
            assert pair_force.force_on_trail_n is not None
            assert (
                np.linalg.norm(
                    np.asarray(pair_force.force_on_lead_n)
                    + np.asarray(pair_force.force_on_trail_n)
                )
                <= THIRD_LAW_TOL
            )

        # Closure force reproduction on lead hand: -lambda = wrenches[-1].force_n
        expected_f = np.asarray(closure.wrenches[-1].force_n)
        assert np.linalg.norm(sum_f_lead - expected_f) <= LAMBDA_TOL
        assert out.reproduction_error_n <= LAMBDA_TOL


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_sum_of_allocated_hand_hand_and_hand_club_forces_reproduces_lambda(
    style: GripStyle,
) -> None:
    """The sum of allocated hand-hand forces and hand-club forces reproduces lambda to <= 1e-9 N."""
    topo = contact_topology(GripSpec(style=style))
    rng = np.random.default_rng(13302)

    for _ in range(10):
        f_closure = tuple(float(x) for x in rng.normal(0, 50, 3))
        m_closure = tuple(float(x) for x in rng.normal(0, 5, 3))
        site = (0.03, -0.02, 0.08)
        closure = _synthetic_closure(f_closure, m_closure, site)

        # Non-zero hand-club force and torque
        f_hc = tuple(float(x) for x in rng.normal(0, 20, 3))
        t_hc = tuple(float(x) for x in rng.normal(0, 2, 3))

        out = allocate_hand_hand_from_closure(
            closure,
            club_rotation=np.eye(3),
            club_origin=np.zeros(3),
            topology=topo,
            static_friction=0.9,
            hand_club_force_n=f_hc,
            hand_club_torque_nm=t_hc,
        )

        assert out.available is True
        sum_f_hh = sum(np.asarray(p.force_on_lead_n) for p in out)
        total_f = sum_f_hh + np.asarray(f_hc)

        expected_f = np.asarray(closure.wrenches[-1].force_n)
        assert np.linalg.norm(total_f - expected_f) <= LAMBDA_TOL
        assert out.reproduction_error_n <= LAMBDA_TOL


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_every_allocated_output_carries_method_allocation_and_user_visible_label(
    style: GripStyle,
) -> None:
    """Every allocated output carries method='allocation' and label matching pair.allocated_label."""
    topo = contact_topology(GripSpec(style=style))
    closure = _synthetic_closure()

    out = allocate_hand_hand_from_closure(
        closure,
        club_rotation=np.eye(3),
        club_origin=np.zeros(3),
        topology=topo,
        static_friction=0.9,
    )

    for pair in topo.hand_hand_pairs():
        assert pair.name in out.pairs
        pair_force = out[pair.name]
        assert isinstance(pair_force, AllocatedPairForce)
        assert pair_force.method == "allocation"
        assert pair_force.label == pair.allocated_label
        assert pair_force.label == f"external:{pair.name}_allocated"
        assert pair_force.pair_name == pair.name

        # Dict conversion preserves label and method
        d = pair_force.to_dict()
        assert d["method"] == "allocation"
        assert d["label"] == pair.allocated_label
        assert d["pair_name"] == pair.name


def test_physical_mode_on_welded_topology_raises_schema_feature_unavailable_error() -> (
    None
):
    """Physical-mode output for a welded topology raises SchemaFeatureUnavailableError."""
    topo = contact_topology()
    closure = _synthetic_closure()

    with pytest.raises(SchemaFeatureUnavailableError) as exc_info:
        allocate_hand_hand_from_closure(
            closure,
            club_rotation=np.eye(3),
            club_origin=np.zeros(3),
            topology=topo,
            static_friction=0.9,
            mode="physical",
            schema_id="full-body-v1",
        )

    err = exc_info.value
    assert err.schema_id == "full-body-v1"
    assert err.issue == 13309
    assert err.child == "HRG-GS-G11"
    assert "physical hand-hand contact" in err.feature


@pytest.mark.parametrize(
    ("style", "expected_count"),
    [
        (GripStyle.OVERLAP, 2),
        (GripStyle.INTERLOCK, 3),
        (GripStyle.TEN_FINGER, 1),
    ],
)
def test_overlay_renders_one_arrow_per_allocated_pair_offscreen(
    style: GripStyle, expected_count: int
) -> None:
    """Overlay renders one arrow per allocated pair when toggle is on."""
    topo = contact_topology(GripSpec(style=style))
    mult = (
        (0.0, -150.0, 1500.0, 1.5, -0.8, 2.2)
        if style == GripStyle.INTERLOCK
        else (0.0, 150.0, 0.0, 1.5, -0.8, 2.2)
    )
    closure = GripClosure.from_multiplier(
        body_a="trail_hand",
        body_b="club",
        point_a_m=(0.0, 0.0, 0.0),
        point_b_m=(0.02, -0.01, 0.05),
        multiplier=mult,
    )
    source = _SyntheticSource(closure=closure)
    kin = _Kin()

    provider = BundleOverlayProvider(
        _bundle(), kin, kin, engine="mujoco"
    ).with_allocated_contacts(source, topo, static_friction=0.9)

    # When show_palm_thumb=True, feed shows all allocated pair arrows
    feed = OverlayFeed(provider.frame_at, show_palm_thumb=True)
    glyphs = feed.glyphs_at(1)

    expected_labels = {p.allocated_label for p in topo.hand_hand_pairs()}
    rendered_allocated_arrows = [a for a in glyphs.arrows if a.label in expected_labels]
    assert len(rendered_allocated_arrows) == expected_count
    assert {a.label for a in rendered_allocated_arrows} == expected_labels

    # When show_palm_thumb=False, allocated arrows are hidden
    feed_off = OverlayFeed(provider.frame_at, show_palm_thumb=False)
    glyphs_off = feed_off.glyphs_at(1)
    rendered_off = [a for a in glyphs_off.arrows if a.label in expected_labels]
    assert len(rendered_off) == 0


def test_palm_thumb_thin_wrapper_backward_compatibility() -> None:
    """Existing allocate_palm_thumb, allocate_from_closure and PalmThumbForce remain compatible."""
    f = (0.0, -100.0, 0.0)
    m = (1.0, 2.0, 3.0)
    pt = (0.3, -0.1, 0.2)
    site = (0.29, -0.12, 0.18)
    n = (0.0, 1.0, 0.0)

    out = allocate_palm_thumb(
        force_on_lead_n=f,
        torque_on_lead_nm=m,
        reference_point_m=site,
        patch_point_m=pt,
        patch_normal=n,
        static_friction=0.9,
    )
    assert isinstance(out, PalmThumbForce)
    assert out.method == "allocation"
    assert out.label == PALM_THUMB_LABEL
    assert out.normal_force_n == pytest.approx(100.0)

    unavail = unavailable_palm_thumb("no solve")
    assert unavail.available is False
    assert unavail.reason == "no solve"
