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
    return DrakeCapabilitySource(
        anthro_spec_bytes, torque_limits=CapacitySubject(75.0, 1.75)
    )


@pytest.fixture(scope="module")
def states(source_with_limits: DrakeCapabilitySource) -> list[np.ndarray]:
    rng = np.random.default_rng(14048)
    return [
        rng.normal(0.0, 0.15, len(source_with_limits.coordinate_order))
        for _ in range(3)
    ]


def test_drake_capability_source_importable() -> None:
    assert DrakeCapabilitySource is not None


def test_hand_mobility_emits_acceleration_and_force_capacity(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    for q in states:
        for hm in source_with_limits.hand_mobility(q).values():
            assert isinstance(hm, HandMobility)
            assert (
                isinstance(hm.acceleration_capacity, CapacityEllipsoid)
                and (hm.acceleration_capacity.radii > 0).all()
            )
            assert (
                isinstance(hm.force_capacity, CapacityEllipsoid)
                and (hm.force_capacity.radii > 0).all()
            )


def test_constrained_vs_unconstrained_capacity_differ(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    for hand in ("lead", "trail"):
        ac = source_with_limits.acceleration_capacity(
            states[0], hand, mode="constrained"
        )
        af = source_with_limits.acceleration_capacity(
            states[0], hand, mode="unconstrained"
        )
        fc = source_with_limits.force_capacity(states[0], hand, mode="constrained")
        ff = source_with_limits.force_capacity(states[0], hand, mode="unconstrained")
        assert (ac.radii > 0).all() and not np.allclose(ac.radii, af.radii, atol=1e-5)
        assert (fc.radii > 0).all() and not np.allclose(fc.radii, ff.radii, atol=1e-5)


def test_welded_lead_hand_topology_caption_and_provenance(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    for hm in source_with_limits.hand_mobility(states[0]).values():
        prov = hm.provenance
        assert prov["engine"] == "drake" and prov["welded_lead_hand"] is True
        assert "welded" in prov["caption"].lower() and prov["assumed_slots"] > 0


def test_explicit_subject_torque_limits_configuration(
    source_with_explicit_subject: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    hm = source_with_explicit_subject.hand_mobility(states[0])
    for hand in ("lead", "trail"):
        assert (
            hm[hand].acceleration_capacity is not None
            and hm[hand].force_capacity is not None
        )


def test_missing_torque_limits_raises(
    anthro_spec_bytes: bytes, states: list[np.ndarray]
) -> None:
    _require_real_pydrake()
    spec = json.loads(anthro_spec_bytes.decode("utf-8"))
    del spec["subject"]
    src_no_lim = DrakeCapabilitySource(json.dumps(spec).encode("utf-8"))
    for fn in (src_no_lim.acceleration_capacity, src_no_lim.force_capacity):
        with pytest.raises(ValueError, match="torque limits"):
            fn(states[0], "lead")


def test_bad_inputs_raise(
    source_with_limits: DrakeCapabilitySource, states: list[np.ndarray]
) -> None:
    for meth, bad_h in (
        (source_with_limits.acceleration_capacity, "both"),
        (source_with_limits.force_capacity, "invalid_hand"),
    ):
        with pytest.raises(ValueError, match="mode"):
            meth(states[0], "lead", mode="invalid")
        with pytest.raises(ValueError, match="hand"):
            meth(states[0], bad_h)
