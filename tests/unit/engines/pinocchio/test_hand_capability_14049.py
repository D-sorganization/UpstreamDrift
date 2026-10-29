"""Per-hand operational-space force- and acceleration-capacity ellipsoids on the Pinocchio golfer (CAP-9, #14049)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.hand_mobility import (  # noqa: E402
    HandMobility,
)
from src.engines.physics_engines.pinocchio.python.force_adapter import (  # noqa: E402
    PinocchioCapabilitySource,
    PinocchioForceAdapter,
)
from src.shared.python.loop_nullspace.capacity import CapacitySubject  # noqa: E402
from src.shared.python.spatial_algebra.operational_space import (  # noqa: E402
    CapacityEllipsoid,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]
ROOT = Path(__file__).resolve().parents[4]
ANTHRO_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def _require_real_pinocchio() -> None:
    pin = pytest.importorskip("pinocchio")
    if type(pin).__module__ == "unittest.mock" or not all(
        hasattr(pin, n) for n in ("rnea", "crba", "SE3", "Model")
    ):
        pytest.skip("Real pinocchio installation required (found mock)")


@pytest.fixture(scope="module")
def anthro_spec_bytes() -> bytes:
    return ANTHRO_SPEC.read_bytes()


@pytest.fixture(scope="module")
def source_with_limits(anthro_spec_bytes: bytes) -> PinocchioCapabilitySource:
    _require_real_pinocchio()
    return PinocchioCapabilitySource(anthro_spec_bytes)


@pytest.fixture(scope="module")
def source_with_explicit_subject(
    anthro_spec_bytes: bytes,
) -> PinocchioCapabilitySource:
    _require_real_pinocchio()
    subj = CapacitySubject(mass_kg=75.0, stature_m=1.75)
    return PinocchioCapabilitySource(anthro_spec_bytes, torque_limits=subj)


@pytest.fixture(scope="module")
def states(source_with_limits: PinocchioCapabilitySource) -> list[np.ndarray]:
    rng = np.random.default_rng(14049)
    n = len(source_with_limits.coordinate_order)
    return [rng.normal(0.0, 0.15, n) for _ in range(3)]


def test_pinocchio_capability_source_importable() -> None:
    """PinocchioCapabilitySource is exported by pinocchio.python.force_adapter."""
    assert PinocchioCapabilitySource is not None


def test_hand_mobility_emits_acceleration_and_force_capacity(
    source_with_limits: PinocchioCapabilitySource, states: list[np.ndarray]
) -> None:
    for q in states:
        hm_dict = source_with_limits.hand_mobility(q)
        for hand in ("lead", "trail"):
            hm = hm_dict[hand]
            assert isinstance(hm, HandMobility)
            assert hm.acceleration_capacity is not None
            assert isinstance(hm.acceleration_capacity, CapacityEllipsoid)
            assert hm.acceleration_capacity.radii.shape == (3,)
            assert (hm.acceleration_capacity.radii > 0).all()

            assert hm.force_capacity is not None
            assert isinstance(hm.force_capacity, CapacityEllipsoid)
            assert hm.force_capacity.radii.shape == (3,)
            assert (hm.force_capacity.radii > 0).all()


def test_direct_capacity_methods_return_capacity_ellipsoids(
    source_with_limits: PinocchioCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    for hand in ("lead", "trail"):
        acc = source_with_limits.acceleration_capacity(q, hand, mode="constrained")
        assert isinstance(acc, CapacityEllipsoid)
        assert acc.radii.shape == (3,)
        assert (acc.radii > 0).all()

        fc = source_with_limits.force_capacity(q, hand, mode="constrained")
        assert isinstance(fc, CapacityEllipsoid)
        assert fc.radii.shape == (3,)
        assert (fc.radii > 0).all()


def test_unconstrained_mode_returns_capacity(
    source_with_limits: PinocchioCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    for hand in ("lead", "trail"):
        acc = source_with_limits.acceleration_capacity(q, hand, mode="unconstrained")
        assert isinstance(acc, CapacityEllipsoid)
        assert (acc.radii > 0).all()

        fc = source_with_limits.force_capacity(q, hand, mode="unconstrained")
        assert isinstance(fc, CapacityEllipsoid)
        assert (fc.radii > 0).all()


def test_factory_method_on_force_adapter(anthro_spec_bytes: bytes) -> None:
    _require_real_pinocchio()
    adapter = PinocchioForceAdapter(anthro_spec_bytes)
    src = adapter.hand_mobility_source()
    assert isinstance(src, PinocchioCapabilitySource)
    assert src.coordinate_order == adapter.coordinate_order


def test_explicit_subject_torque_limits_configuration(
    source_with_explicit_subject: PinocchioCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    hm = source_with_explicit_subject.hand_mobility(q)
    assert hm["lead"].acceleration_capacity is not None
    assert hm["trail"].force_capacity is not None


def test_missing_torque_limits_raises(
    anthro_spec_bytes: bytes, states: list[np.ndarray]
) -> None:
    _require_real_pinocchio()
    spec = json.loads(anthro_spec_bytes.decode("utf-8"))
    del spec["subject"]
    spec_no_subject = json.dumps(spec).encode("utf-8")
    src = PinocchioCapabilitySource(spec_no_subject)
    q = states[0]
    with pytest.raises(ValueError, match="torque limits"):
        src.acceleration_capacity(q, "lead")


def test_invalid_hand_and_mode_raise(
    source_with_limits: PinocchioCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    with pytest.raises(ValueError, match="hand"):
        source_with_limits.grip_point_world("middle", q)

    with pytest.raises(ValueError, match="mode"):
        source_with_limits.acceleration_capacity(q, "lead", mode="semi-constrained")
