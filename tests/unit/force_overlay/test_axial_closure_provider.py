"""Provider plumbing for the grip closure wrench and ball impact blackout (#13179)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.mujoco_axial_loads import GripClosure
from src.shared.python.force_overlay.bundle_provider import (
    BundleOverlayProvider,
    ClosureAxialLoadSource,
    contact_sample_wrenches,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

DT = 0.01


class _Kin:
    total_mass_kg = 80.0

    def joint_frames(self, coordinates):
        return {}

    def center_of_mass_m(self, coordinates):
        return (0.0, 0.0, 1.0)

    def evaluate_contact_samples(self, coordinates, rates):
        return {}


def _bundle(steps: int = 6) -> InputBundle:
    t = np.arange(steps + 1) * DT
    q = np.stack([t, 0 * t], axis=1)
    return InputBundle(
        spec_bytes=json.dumps({"coordinate_order": ["A", "B"]}).encode(),
        coordinate_order=("A", "B"),
        dt_s=DT,
        q0=q[0],
        v0=q[0],
        efforts=np.stack([np.arange(steps) + 1.0, np.zeros(steps)], axis=1),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )


class _ClosureSource:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.efforts: list = []
        self.closures: list = []

    def closure_wrench(self, coordinates, rates, primitive_efforts):
        self.efforts.append(primitive_efforts)
        if not self.available:
            return GripClosure.unavailable("synthetic: no solve")
        return GripClosure.from_multiplier(
            body_a="hand",
            body_b="club",
            point_a_m=(0, 0, 0),
            point_b_m=(0, 0, 0),
            multiplier=(1, 2, 3, 4, 5, 6),
        )

    def axial_load_frame(
        self,
        coordinates,
        rates,
        accelerations,
        time_s,
        external_wrenches=None,
        closure=None,
    ):
        self.closures.append(closure)
        return AxialLoadFrame(time_s, {"arm": 7.0, "leg": 2.0}, "fake")


def _provider(source) -> BundleOverlayProvider:
    kin = _Kin()
    return BundleOverlayProvider(_bundle(), kin, kin, engine="mujoco").with_axial_loads(
        source
    )


def test_closure_source_receives_the_frames_efforts_and_reports_applied() -> None:
    source = _ClosureSource()
    assert isinstance(source, ClosureAxialLoadSource)
    frame = _provider(source).frame_at(2)
    assert source.efforts[-1] == {"A": 3.0, "B": 0.0}
    assert source.closures[-1].available
    assert frame.metadata["axial_closure"] == "applied"
    assert frame.metadata["axial_closure_multiplier"] == [1, 2, 3, 4, 5, 6]
    assert frame.axial_loads.values_n["arm"] == 7.0


def test_unavailable_closure_is_surfaced_with_its_reason() -> None:
    source = _ClosureSource(available=False)
    frame = _provider(source).frame_at(1)
    assert not source.closures[-1].available
    assert frame.metadata["axial_closure"] == "unavailable"
    assert frame.metadata["axial_closure_reason"] == "synthetic: no solve"
    assert "axial_closure_multiplier" not in frame.metadata


def test_plain_axial_source_is_called_as_before() -> None:
    class Plain:
        def axial_load_frame(
            self, coordinates, rates, accelerations, time_s, external_wrenches=None
        ):
            return AxialLoadFrame(time_s, {"arm": 1.0}, "plain")

    frame = _provider(Plain()).frame_at(1)
    assert frame.axial_loads.values_n["arm"] == 1.0
    assert "axial_closure" not in frame.metadata


def test_impact_window_blanks_axial_values_never_zero() -> None:
    provider = _provider(_ClosureSource()).with_axial_impact_window(3.2 * DT, DT)
    inside = [provider.frame_at(k) for k in (3, 4)]
    outside = [provider.frame_at(k) for k in (1, 2, 5)]
    for frame in inside:
        assert dict(frame.axial_loads.values_n) == {"arm": None, "leg": None}
        assert "ball impact" in frame.axial_loads.source
        assert frame.metadata["axial_impact"] == "unavailable"
    for frame in outside:
        assert frame.axial_loads.values_n["arm"] == 7.0
        assert "axial_impact" not in frame.metadata


def test_impact_window_preconditions() -> None:
    provider = _provider(_ClosureSource())
    with pytest.raises(ValueError, match="impact_time_s"):
        provider.with_axial_impact_window(float("nan"), DT)
    with pytest.raises(ValueError, match="half_width_s"):
        provider.with_axial_impact_window(0.03, -1.0)


def test_ground_free_moment_rides_on_the_contact_wrench() -> None:
    sample = SimpleNamespace(
        normal_force_n=(0.0, 0.0, 400.0),
        friction_force_n=(1.0, 0.0, 0.0),
        contact_point_m=(0.1, 0.0, 0.0),
    )
    quiet = SimpleNamespace(
        normal_force_n=(0.0, 0.0, 0.0),
        friction_force_n=(0.0, 0.0, 0.0),
        contact_point_m=(0.0, 0.0, 0.0),
    )
    spin = {"heel": np.array([0.0, 0.0, -2.5]), "toe": np.array([0.0, 0.0, 9.0])}
    wrenches = contact_sample_wrenches(
        {"heel": sample, "toe": quiet},
        {"heel": "calcn_r", "toe": "toes_r"},
        source="t",
        free_moments=spin,
    )
    (heel,) = wrenches  # the unloaded toe sphere carries no moment either
    assert heel.torque_nm == (0.0, 0.0, -2.5)
    plain = contact_sample_wrenches({"heel": sample}, {"heel": "calcn_r"}, source="t")
    assert plain[0].torque_nm is None


def test_contact_source_free_moments_reach_the_axial_source() -> None:
    sample = SimpleNamespace(
        normal_force_n=(0.0, 0.0, 400.0),
        friction_force_n=(0.0, 0.0, 0.0),
        contact_point_m=(0.1, 0.0, 0.0),
    )

    class Ground(_Kin):
        def evaluate_contact_samples(self, coordinates, rates):
            return {"sphere_foot": sample}

        def contact_free_moments(self, coordinates, rates):
            return {"sphere_foot": np.array([0.0, 0.0, 1.5])}

    seen = []

    class Axial:
        def axial_load_frame(
            self, coordinates, rates, accelerations, time_s, external_wrenches=None
        ):
            seen.extend(external_wrenches)
            return AxialLoadFrame(time_s, {"arm": 1.0}, "plain")

    kin = Ground()
    BundleOverlayProvider(_bundle(), kin, kin, engine="mujoco").with_axial_loads(
        Axial()
    ).frame_at(1)
    assert seen[0].torque_nm == (0.0, 0.0, 1.5)
