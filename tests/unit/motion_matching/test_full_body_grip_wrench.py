"""Grip wrench extraction in the spec-built full-body engines (GCV-8, #11714).

Same-input bundle: the committed ground-support record, projected onto the
dual-grip closure manifold so MuJoCo (exact KKT), Drake and MyoSuite share one
state.  Sign convention: wrench exerted by the hand ON THE CLUB, world frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_wrench import to_overlay_wrenches
from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    project_to_closure,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.requires_mujoco,
    pytest.mark.requires_drake,
    pytest.mark.requires_pinocchio,
]

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
FRAMES = (0, 400, 420)  # 420 is a contact spike
# Same acceleration-parity level (L1, epic #11605) the engines already meet:
# |dX| <= ATOL + RTOL * max|X|.  The multiplier is M a - f, so it inherits it.
ATOL = 1e-6
RTOL = 1e-8


@pytest.fixture(scope="module")
def plants() -> dict[str, VectorPlant]:
    for module in ("mujoco", "pydrake", "pinocchio"):
        pytest.importorskip(module)
    spec = (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()
    return {e: VectorPlant(e, spec) for e in ("mujoco", "drake", "pinocchio")}


@pytest.fixture(scope="module")
def record() -> dict[str, np.ndarray]:
    with np.load(EVIDENCE / "dynamics_record.npz") as data:
        return {key: data[key] for key in ("q", "v", "tau")}


def _state(plants, record, frame):
    st = project_to_closure(plants["pinocchio"], record["q"][frame], record["v"][frame])
    return st.q, st.v, record["tau"][frame]


def _grip(plant: VectorPlant, q, v, tau):
    return plant._adapter.grip_analysis(
        plant._named(q), plant._named(v), plant._named(tau)
    )


@pytest.mark.parametrize("engine", ["mujoco", "drake"])
@pytest.mark.parametrize("frame", FRAMES)
def test_solve_with_multipliers_leaves_accelerations_unchanged(
    plants, record, engine, frame
) -> None:
    plant = plants[engine]
    adapter = plant._adapter
    q, v, tau = (plant._named(x) for x in _state(plants, record, frame))
    plain = adapter.accelerations(q, v, tau)
    with_lambda, lam = adapter.solve_with_multipliers(q, v, tau)
    assert plain == with_lambda  # bitwise: same solve, same arithmetic
    assert lam.shape == (6,) and np.isfinite(lam).all()


@pytest.mark.parametrize("frame", FRAMES)
def test_mujoco_and_drake_agree_on_per_hand_and_net_grip(plants, record, frame) -> None:
    q, v, tau = _state(plants, record, frame)
    mj, dr = (_grip(plants[e], q, v, tau) for e in ("mujoco", "drake"))
    for g in (mj, dr):
        assert g.split_method == "constraint_multiplier"
        assert g.left is not None and g.right is not None
    scale = np.abs(np.array(dr.net_force_n)).max()
    for field in ("net_force_n", "couple_at_midpoint_nm"):
        a, b = np.array(getattr(mj, field)), np.array(getattr(dr, field))
        bound = ATOL + RTOL * max(np.abs(b).max(), scale)
        np.testing.assert_allclose(a, b, rtol=0, atol=bound, err_msg=field)
    for side in ("left", "right"):
        a = np.array(getattr(mj, side).force_on_club_n)
        b = np.array(getattr(dr, side).force_on_club_n)
        np.testing.assert_allclose(a, b, rtol=0, atol=ATOL + RTOL * np.abs(b).max())


@pytest.mark.parametrize("frame", FRAMES)
def test_net_force_matches_mujoco_recursive_newton_euler(plants, record, frame) -> None:
    """Independent check: MuJoCo's own RNE gives the net hand force on the club."""
    import mujoco

    plant = plants["mujoco"]
    adapter = plant._adapter
    q, v, tau = _state(plants, record, frame)
    grip = _grip(plant, q, v, tau)
    accel = adapter.accelerations(plant._named(q), plant._named(v), plant._named(tau))
    model, data = adapter.model, adapter.data
    qacc = np.zeros(model.nv)
    for name, index in adapter._indices.items():
        qacc[index] = accel[name]
    data.qacc[:] = qacc
    data.efc_force[:] = 0.0  # the stock weld is not part of this solve
    data.xfrc_applied[:] = 0.0
    mujoco.mj_rnePostConstraint(model, data)
    body = int(model.site_bodyid[adapter._closure[1]])
    force_world = data.cfrc_int[body][3:]  # parent-on-subtree, world frame
    np.testing.assert_allclose(
        grip.net_force_n,
        force_world,
        rtol=0,
        atol=ATOL + RTOL * np.abs(force_world).max(),
    )


def test_closing_hand_is_the_spec_right_hand(plants) -> None:
    assert plants["mujoco"]._adapter.closing_hand_side == "R"
    assert plants["drake"]._adapter.closing_hand_side == "R"


def test_overlay_wrenches_are_grip_kind_with_per_hand_labels(plants, record) -> None:
    q, v, tau = _state(plants, record, 400)
    for engine in ("mujoco", "drake"):
        plant = plants[engine]
        wrenches = plant._adapter.grip_overlay_wrenches(
            plant._named(q), plant._named(v), plant._named(tau)
        )
        assert {w.kind for w in wrenches} == {WrenchKind.GRIP}
        labels = {w.label for w in wrenches}
        assert {"grip:hand_left", "grip:hand_right", "grip:net_midpoint"} <= labels


def test_grip_analysis_round_trips_to_overlay(plants, record) -> None:
    q, v, tau = _state(plants, record, 0)
    g = _grip(plants["mujoco"], q, v, tau)
    assert (
        len(to_overlay_wrenches(g, source="t")) == 6
    )  # 2 hands, net, couple, 2 moments of force
