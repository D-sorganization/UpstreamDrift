"""Per-hand operational-space force- and acceleration-capacity ellipsoids on the Drake golfer (CAP-8, #14048)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.force_adapter import (  # noqa: E402
    DrakeCapabilitySource,
    DrakeForceAdapter,
)
from src.engines.physics_engines.mujoco.python.hand_mobility import (  # noqa: E402
    HandMobility,
)
from src.shared.python.loop_nullspace.capacity import CapacitySubject  # noqa: E402
from src.shared.python.spatial_algebra.operational_space import (  # noqa: E402
    CapacityEllipsoid,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]
ROOT = Path(__file__).resolve().parents[4]
ANTHRO_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def _require_real_pydrake() -> None:
    pytest.importorskip("pydrake")
    try:
        import pydrake.all as api

        if type(api).__module__ == "unittest.mock" or not hasattr(
            api, "MultibodyPlant"
        ):
            pytest.skip("Real pydrake installation required (found mock)")
    except ImportError:
        pytest.skip("pydrake not installed")


@pytest.fixture(scope="module")
def anthro_spec_bytes() -> bytes:
    return ANTHRO_SPEC.read_bytes()


@pytest.fixture(scope="module")
def source_with_limits(anthro_spec_bytes: bytes) -> DrakeCapabilitySource:
    _require_real_pydrake()
    return DrakeCapabilitySource(anthro_spec_bytes)


@pytest.fixture(scope="module")
def source_with_explicit_subject(anthro_spec_bytes: bytes) -> DrakeCapabilitySource:
    _require_real_pydrake()
    subj = CapacitySubject(mass_kg=75.0, stature_m=1.75)
    return DrakeCapabilitySource(anthro_spec_bytes, torque_limits=subj)


@pytest.fixture(scope="module")
def states(source_with_limits: DrakeCapabilitySource) -> list[np.ndarray]:
    rng = np.random.default_rng(14048)
    n = len(source_with_limits.coordinate_order)
    return [rng.normal(0.0, 0.15, n) for _ in range(3)]


def test_drake_capability_source_importable() -> None:
    """DrakeCapabilitySource is exported by drake.python.force_adapter."""
    assert DrakeCapabilitySource is not None


def test_hand_mobility_emits_acceleration_and_force_capacity(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
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
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    for hand in ("lead", "trail"):
        acc = source_with_limits.acceleration_capacity(q, hand, mode="constrained")
        assert isinstance(acc, CapacityEllipsoid)
        assert acc.radii.shape == (3,)
        assert (acc.radii > 0).all()

        f = source_with_limits.force_capacity(q, hand, mode="constrained")
        assert isinstance(f, CapacityEllipsoid)
        assert f.radii.shape == (3,)
        assert (f.radii > 0).all()


def test_constrained_vs_unconstrained_capacity_differ(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    for hand in ("lead", "trail"):
        acc_con = source_with_limits.acceleration_capacity(q, hand, mode="constrained")
        acc_free = source_with_limits.acceleration_capacity(
            q, hand, mode="unconstrained"
        )
        assert not np.allclose(acc_con.radii, acc_free.radii, atol=1e-5)

        f_con = source_with_limits.force_capacity(q, hand, mode="constrained")
        f_free = source_with_limits.force_capacity(q, hand, mode="unconstrained")
        assert not np.allclose(f_con.radii, f_free.radii, atol=1e-5)


def test_welded_lead_hand_topology_caption_and_provenance(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    hm = source_with_limits.hand_mobility(q)
    for hand in ("lead", "trail"):
        prov = hm[hand].provenance
        assert prov["engine"] == "drake"
        assert prov["welded_lead_hand"] is True
        assert "caption" in prov
        assert "welded" in prov["caption"].lower()
        assert prov["torque_limit_source"] == "loop_nullspace_registry"
        assert prov["assumed_slots"] > 0
        assert prov["verified_slots"] > 0


def test_explicit_subject_torque_limits_configuration(
    source_with_explicit_subject: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    hm = source_with_explicit_subject.hand_mobility(q)
    for hand in ("lead", "trail"):
        assert hm[hand].acceleration_capacity is not None
        assert hm[hand].force_capacity is not None


def test_missing_torque_limits_raises(
    anthro_spec_bytes: bytes, states: list[np.ndarray]
) -> None:
    _require_real_pydrake()
    spec = json.loads(anthro_spec_bytes.decode("utf-8"))
    del spec["subject"]
    spec_no_subject = json.dumps(spec).encode("utf-8")
    source_no_limits = DrakeCapabilitySource(spec_no_subject)

    q = states[0]
    with pytest.raises(ValueError, match="torque limits"):
        source_no_limits.acceleration_capacity(q, "lead")
    with pytest.raises(ValueError, match="torque limits"):
        source_no_limits.force_capacity(q, "lead")


def test_bad_inputs_raise(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    q = states[0]
    with pytest.raises(ValueError, match="mode"):
        source_with_limits.acceleration_capacity(q, "lead", mode="invalid")
    with pytest.raises(ValueError, match="hand"):
        source_with_limits.acceleration_capacity(q, "both")
    with pytest.raises(ValueError, match="mode"):
        source_with_limits.force_capacity(q, "lead", mode="invalid")
    with pytest.raises(ValueError, match="hand"):
        source_with_limits.force_capacity(q, "invalid_hand")
