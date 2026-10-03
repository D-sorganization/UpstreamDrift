"""Tests for OpenSim muscle lines of action (FTO-16, #11301).

Synthetic ``Thelen2003Muscle`` hanging a block on a vertical slider. Needs the
real ``opensim`` package and skips cleanly without it. No visualizer is used.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

osim = pytest.importorskip("opensim", reason="OpenSim not installed")
if not hasattr(osim, "Model"):
    pytest.skip("real opensim is unavailable", allow_module_level=True)

from scipy.optimize import brentq  # noqa: E402

from src.engines.physics_engines.opensim.python.opensim_force_torque import (  # noqa: E402
    OpenSimForceTorqueSource,
)
from src.shared.python.force_overlay import WrenchKind  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

G = 9.80665
BLOCK_MASS = 25.0
MAX_ISOMETRIC_N = 500.0
ORIGIN_HEIGHT_M = 0.5
ACTIVATION = 0.5
# Tendon force at q=-0.15 m is above the weight and at q=-0.10 m below it.
Q_BRACKET = (-0.15, -0.10)


def _muscle_block(*, with_muscle: bool = True, via_point: bool = False):
    """Block on a vertical slider (OpenSim +y) hung from a ground-fixed muscle."""
    model = osim.Model()
    model.setName("synthetic_muscle_block")
    block = osim.Body("block", BLOCK_MASS, osim.Vec3(0), osim.Inertia(0.1, 0.1, 0.1))
    model.addBody(block)
    half_turn = osim.Vec3(0, 0, math.pi / 2)  # slider x axis -> ground +y
    joint = osim.SliderJoint(
        "slide",
        model.getGround(),
        osim.Vec3(0),
        half_turn,
        block,
        osim.Vec3(0),
        half_turn,
    )
    model.addJoint(joint)
    if with_muscle:
        muscle = osim.Thelen2003Muscle("biceps", MAX_ISOMETRIC_N, 0.3, 0.2, 0.0)
        muscle.addNewPathPoint(
            "origin", model.getGround(), osim.Vec3(0, ORIGIN_HEIGHT_M, 0)
        )
        if via_point:
            muscle.addNewPathPoint(
                "via", model.getGround(), osim.Vec3(0.02, ORIGIN_HEIGHT_M - 0.2, 0)
            )
        muscle.addNewPathPoint("insertion", block, osim.Vec3(0))
        model.addForce(muscle)
    return model, joint


def _settle(model, joint, activation: float, q: float):
    """State with the slider at ``q`` and muscle fibre in equilibrium."""
    state = model.initSystem()
    joint.updCoordinate().setValue(state, q)
    model.realizePosition(state)
    muscle = osim.Muscle.safeDownCast(model.getForceSet().get(0))
    muscle.setActivation(state, activation)
    model.equilibrateMuscles(state)
    model.realizeAcceleration(state)
    return state, muscle


def _equilibrium():
    """Settled state where the tendon force equals the block weight."""
    model, joint = _muscle_block()

    def residual(q: float) -> float:
        state, muscle = _settle(model, joint, ACTIVATION, q)
        return muscle.getTendonForce(state) - BLOCK_MASS * G

    q_eq = brentq(residual, *Q_BRACKET, xtol=1e-12)
    state, muscle = _settle(model, joint, ACTIVATION, q_eq)
    return model, state, muscle


def _by_label(frame) -> dict:
    return {w.label: w for w in frame.by_kind(WrenchKind.MUSCLE)}


def test_insertion_wrench_balances_block_weight_at_equilibrium() -> None:
    model, state, muscle = _equilibrium()
    wrenches = _by_label(OpenSimForceTorqueSource(model).sample(state))
    insertion = np.asarray(wrenches["muscle:biceps:insertion"].force_n)
    tendon = muscle.getTendonForce(state)
    assert np.linalg.norm(insertion) == pytest.approx(tendon, rel=1e-9)
    # World is Z-up: the block hangs below the origin, so the force points up.
    assert insertion[2] > 0.0
    assert insertion[2] + (-BLOCK_MASS * G) == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(insertion[:2], 0.0, atol=1e-9)


def test_insertion_force_points_from_block_toward_origin() -> None:
    model, state, _ = _equilibrium()
    wrenches = _by_label(OpenSimForceTorqueSource(model).sample(state))
    origin_pt = np.asarray(wrenches["muscle:biceps:origin"].point_m)
    insertion_pt = np.asarray(wrenches["muscle:biceps:insertion"].point_m)
    toward_origin = origin_pt - insertion_pt
    force = np.asarray(wrenches["muscle:biceps:insertion"].force_n)
    cosine = (
        force @ toward_origin / (np.linalg.norm(force) * np.linalg.norm(toward_origin))
    )
    assert cosine == pytest.approx(1.0, abs=1e-9)
    # Attachment points are in the Z-up world: origin 0.5 m above ground.
    assert origin_pt == pytest.approx([0.0, 0.0, ORIGIN_HEIGHT_M], abs=1e-9)
    assert wrenches["muscle:biceps:insertion"].body == "block"
    assert wrenches["muscle:biceps:origin"].body == "ground"


def test_origin_and_insertion_forces_are_equal_and_opposite() -> None:
    model, state, _ = _equilibrium()
    wrenches = _by_label(OpenSimForceTorqueSource(model).sample(state))
    origin = np.asarray(wrenches["muscle:biceps:origin"].force_n)
    insertion = np.asarray(wrenches["muscle:biceps:insertion"].force_n)
    assert origin + insertion == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert origin[2] < 0.0  # the origin is pulled down toward the block


def test_muscle_wrenches_have_no_torque_and_the_muscle_source() -> None:
    model, state, _ = _equilibrium()
    for wrench in _by_label(OpenSimForceTorqueSource(model).sample(state)).values():
        assert wrench.kind is WrenchKind.MUSCLE
        assert wrench.torque_nm is None
        assert wrench.source == "opensim:Muscle.getTendonForce"


def test_zero_activation_still_reports_the_passive_tendon_force() -> None:
    model, joint = _muscle_block()
    state, muscle = _settle(model, joint, 0.0, Q_BRACKET[0])  # stretched 0.65 m
    tendon = muscle.getTendonForce(state)
    assert tendon > 0.0  # passive fibre force carried by the tendon
    wrenches = _by_label(OpenSimForceTorqueSource(model).sample(state))
    force = np.asarray(wrenches["muscle:biceps:insertion"].force_n)
    assert np.linalg.norm(force) == pytest.approx(tendon, rel=1e-9)


def test_model_without_muscles_has_no_muscle_wrenches() -> None:
    model, _ = _muscle_block(with_muscle=False)
    state = model.initSystem()
    frame = OpenSimForceTorqueSource(model).sample(state)
    assert [w.label for w in frame.wrenches if w.label.startswith("muscle:")] == []


def test_include_muscles_false_omits_muscle_wrenches() -> None:
    model, state, _ = _equilibrium()
    frame = OpenSimForceTorqueSource(model, include_muscles=False).sample(state)
    assert not frame.by_kind(WrenchKind.MUSCLE)


def test_disabled_muscle_is_omitted_not_zero() -> None:
    model, joint = _muscle_block()
    muscle = osim.Muscle.safeDownCast(model.getForceSet().get(0))
    muscle.set_appliesForce(False)
    state = model.initSystem()
    frame = OpenSimForceTorqueSource(model).sample(state)
    assert not frame.by_kind(WrenchKind.MUSCLE)


def test_via_point_ends_use_the_effective_end_directions() -> None:
    """Only the end attachments are drawn; via points still bend the path."""
    model, joint = _muscle_block(via_point=True)
    state, muscle = _settle(model, joint, ACTIVATION, Q_BRACKET[0])
    wrenches = _by_label(OpenSimForceTorqueSource(model).sample(state))
    assert sorted(wrenches) == ["muscle:biceps:insertion", "muscle:biceps:origin"]
    tendon = muscle.getTendonForce(state)
    for wrench in wrenches.values():
        assert np.linalg.norm(wrench.force_n) == pytest.approx(tendon, rel=1e-9)
    # The origin and the via point share the ground body, so the first force
    # carrying point is the via point; it is pulled along the via->block line.
    origin = wrenches["muscle:biceps:origin"]
    assert origin.body == "ground"
    assert origin.point_m == pytest.approx([0.02, 0.0, ORIGIN_HEIGHT_M - 0.2])
    assert origin.force_n[0] < 0.0  # toward the block, which sits on x = 0


def test_negative_tendon_force_violates_the_postcondition(monkeypatch) -> None:
    model, state, _ = _equilibrium()
    source = OpenSimForceTorqueSource(model)
    monkeypatch.setattr(osim.Muscle, "getTendonForce", lambda self, s: -1.0)
    with pytest.raises(AssertionError, match="tendon force"):
        source.sample(state)


def _function_based_muscle_model():
    """Block hung from a muscle whose path is a FunctionBasedPath (OpenSim 4.5+).

    Its path is not a ``GeometryPath`` (no points), so ``getGeometryPath()``
    throws ``std::bad_cast``.
    """
    model, _ = _muscle_block(with_muscle=False)
    muscle = osim.Thelen2003Muscle("fb_muscle", MAX_ISOMETRIC_N, 0.3, 0.2, 0.0)
    path = osim.FunctionBasedPath()
    path.appendCoordinatePath("/jointset/slide/slide_coord_0")
    path.setLengthFunction(osim.LinearFunction(-1.0, ORIGIN_HEIGHT_M))
    path.setLengtheningSpeedFunction(
        osim.MultivariatePolynomialFunction(osim.Vector(3, 0.0), 2, 1)
    )
    muscle.set_path(path)
    model.addForce(muscle)
    return model


def test_function_based_path_muscle_is_omitted_and_sample_still_works() -> None:
    model = _function_based_muscle_model()
    state = model.initSystem()
    frame = OpenSimForceTorqueSource(model).sample(state)  # must not raise
    assert not frame.by_kind(WrenchKind.MUSCLE)
    assert frame.by_kind(WrenchKind.JOINT_REACTION)  # other channels unaffected
    assert OpenSimForceTorqueSource(model).muscle_wrenches(state) == ()


def _current_rss_kib() -> int:
    """Resident set size now (not the peak, which earlier tests may have set)."""
    with open("/proc/self/statm") as statm:
        pages = int(statm.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") // 1024


@pytest.mark.skipif(
    not os.path.exists("/proc/self/statm"), reason="needs /proc for current RSS"
)
def test_repeated_sampling_does_not_leak_point_force_directions() -> None:
    """``getPointForceDirections`` hands ownership of each point to the caller."""
    model, state, _ = _equilibrium()
    source = OpenSimForceTorqueSource(model)
    for _ in range(2000):  # warm up allocator pools
        source.muscle_wrenches(state)
    before = _current_rss_kib()
    for _ in range(40000):
        source.muscle_wrenches(state)
    growth_kib = _current_rss_kib() - before
    # Unfixed, every call leaks ~3 native objects (>5 MiB over this loop).
    assert growth_kib < 2048
