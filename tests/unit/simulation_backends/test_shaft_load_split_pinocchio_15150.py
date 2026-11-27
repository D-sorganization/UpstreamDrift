"""Shaft loads by part on the live Pinocchio chain adapter (#15150 Z9).

The internal wrench at each shaft joint, rebuilt from the axial, shear,
torsion and bending split, must equal the spatial force Pinocchio's RNEA
passes through that joint (the distal links hold no other body).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

SEEDS = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
RTOL = 1e-10
FRAME = 400


@pytest.fixture(scope="module")
def pin():
    return pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def adapter(pin):
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.motion_matching.execution.spec_builder import (
        with_shaft_model,
    )

    doc = json.loads((SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes())
    return PinocchioForceAdapter(
        json.dumps(with_shaft_model(doc, "driver", "chain")).encode()
    )


def _state(adapter, seed: int):
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        q = np.zeros(adapter.nv)
        q[: rec["q"].shape[1]] = rec["q"][FRAME]
    rng = np.random.default_rng(seed)
    return q, rng.normal(size=adapter.nv), rng.normal(size=adapter.nv)


def _rnea_station_wrenches(adapter, pin, q, v, a, gravity: bool):
    """World force and moment about each station point from RNEA."""
    model = adapter._model.copy()
    if not gravity:
        model.gravity = pin.Motion.Zero()
    data = model.createData()
    pin.rnea(model, data, q, v, a)
    pin.forwardKinematics(model, data, q)
    forces, moments = [], []
    for joints in adapter._shaft_station_joints():
        world = data.oMi[joints[0]].act(data.f[joints[0]])
        point = data.oMi[joints[0]].translation
        forces.append(world.linear)
        moments.append(world.angular - np.cross(point, world.linear))
    return np.array(forces), np.array(moments)


@pytest.mark.parametrize("gravity", [True, False])
def test_section_loads_rebuild_the_rnea_joint_wrench(adapter, pin, gravity) -> None:
    from src.shared.python.simulation_backends.shaft_load_split import section_loads

    q, v, a = _state(adapter, 9)
    points, axes = adapter.shaft_stations(q)
    loads = section_loads(points, axes, *adapter.shaft_link_loads(q, v, a, gravity))
    force = loads.shear_n[0] - loads.axial_n[0][:, None] * axes
    moment = loads.bending_nm[0] + loads.torsion_nm[0][:, None] * axes
    ref_f, ref_m = _rnea_station_wrenches(adapter, pin, q, v, a, gravity)
    scale = max(1.0, float(np.max(np.abs(np.r_[ref_f, ref_m]))))
    assert np.max(np.abs(force - ref_f)) <= RTOL * scale
    assert np.max(np.abs(moment - ref_m)) <= RTOL * scale


def test_axes_point_from_the_grip_to_the_head(adapter) -> None:
    q, _, _ = _state(adapter, 1)
    points, axes = adapter.shaft_stations(q)
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-12)
    steps = np.diff(points, axis=0)
    assert np.all(np.einsum("ij,ij->i", steps, axes[:-1]) > 0.0)


def test_live_split_parts_sum_to_the_total(adapter) -> None:
    from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
    from src.shared.python.simulation_backends.shaft_load_split import (
        shaft_load_split,
    )

    q, v, _ = _state(adapter, 3)
    tau = np.random.default_rng(4).normal(scale=5.0, size=adapter.nv)
    split = shaft_load_split(
        AdapterConstrainedProvider(adapter),
        adapter.shaft_stations,
        adapter.shaft_link_loads,
        q,
        v,
        tau,
    )
    total = split.total()
    assert np.all(np.isfinite(total.bending_nm))
    assert total.bending_magnitude_nm[0, 0] > 0.0
    np.testing.assert_allclose(
        split.part("ztcf").axial_n + split.part("zvcf").axial_n, total.axial_n
    )
