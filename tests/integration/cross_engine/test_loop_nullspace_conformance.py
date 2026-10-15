"""NSX-56: two-track cross-engine conformance of the closed-loop snapshot (#12584).

Every state is first projected onto the closure with ``project_to_closure``
(so the Jacobian drift is exact on the closure tangent space), then each FULL
engine builds its snapshot through ``snapshot_from_adapter`` and is compared
with the Pinocchio reference.

* **Track N (numerical)** compares the canonical ``M, h, J_c, Jdot_v`` plus the
  grip frame, club inertia split and loop topology.  Each metric's tolerance is
  the entry registered for the engine pair in
  ``divergence_registry.yaml``; nothing is widened here.
* **Track S (scientific)** compares invariants and envelopes derived from the
  snapshot: constrained acceleration, grip-wrench multiplier, kinetic energy,
  power, closure singular values and the grip-wrench envelope.  The tolerance is
  propagated from the Track N differences measured in the same run
  (``kkt_bound``); it is never entered by hand.

Engines without a FULL ``CLOSED_LOOP_SNAPSHOT`` capability (Simscape, and the
synthetic fixtures) are asserted to refuse, not silently skipped.  Agreement
between engines is not physical validation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.engine_core.cross_engine_validator import DivergenceRegistry
from src.shared.python.loop_nullspace.contracts import InputProvenance
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
    create_engine_force_adapter,
)
from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    project_to_closure,
)
from tests.integration.cross_engine.loop_nullspace_conformance_support import (
    CHECK_NAME,
    EVIDENCE,
    SPEC_PATH,
    TRACK_N_METRICS,
    build_snapshot,
    effort_response,
    kkt_bound,
    kkt_solve,
    multiplier_envelope,
    seeded_effort,
    to_reference_order,
    track_n_errors,
)

pytestmark = pytest.mark.integration

REGISTRY = DivergenceRegistry.from_yaml(
    Path(__file__).with_name("divergence_registry.yaml")
)
REFERENCE = EngineType.PINOCCHIO
OTHERS = (
    pytest.param(EngineType.MUJOCO, marks=pytest.mark.requires_mujoco, id="mujoco"),
    pytest.param(EngineType.DRAKE, marks=pytest.mark.requires_drake, id="drake"),
    pytest.param(EngineType.OPENSIM, marks=pytest.mark.requires_opensim, id="opensim"),
)
SDK = {
    EngineType.MUJOCO: "mujoco",
    EngineType.DRAKE: "pydrake",
    EngineType.OPENSIM: "opensim",
}
FRAMES = (0, 400)
#: Guard that a propagated bound stays informative (a vacuous bound proves nothing).
INFORMATIVE = 1e-6


@pytest.fixture(scope="module")
def reference_adapter():
    pytest.importorskip("pinocchio")
    return create_engine_force_adapter(REFERENCE, SPEC_PATH, allow_synthetic=False)


@pytest.fixture(scope="module")
def plant():
    pytest.importorskip("pinocchio")
    return VectorPlant("pinocchio", SPEC_PATH.read_bytes())


@pytest.fixture(scope="module", params=FRAMES, ids=lambda f: f"frame{f}")
def projected(request, plant):
    """One closure-consistent named state (the projection is done exactly once)."""
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][request.param], rec["v"][request.param]
    proj = project_to_closure(plant, q0, v0)
    return plant._named(proj.q), plant._named(proj.v), request.param


@pytest.fixture(scope="module", params=OTHERS)
def pair(request, reference_adapter, projected):
    """``(engine, reference NamedSnapshot, other NamedSnapshot)`` at one state."""
    engine = request.param
    pytest.importorskip(SDK[engine])
    other_adapter = create_engine_force_adapter(
        engine, SPEC_PATH, allow_synthetic=False
    )
    named_q, named_v, frame = projected
    source = f"dynamics_record frame {frame}"
    order = tuple(reference_adapter.coordinate_order)
    ref = to_reference_order(
        build_snapshot(reference_adapter, named_q, named_v, source), order
    )
    other = to_reference_order(
        build_snapshot(other_adapter, named_q, named_v, source), order
    )
    return engine, ref, other


def _entry(engine: EngineType, metric: str):
    entry = REGISTRY.find(
        check_name=CHECK_NAME,
        metric_name=metric,
        engine1=REFERENCE.value,
        engine2=engine.value,
    )
    assert entry is not None, f"no registry entry: {engine.value} {metric}"
    return entry


# -- Track N ---------------------------------------------------------------


def test_every_pair_metric_is_registered_with_a_rationale() -> None:
    for param in OTHERS:
        for metric in TRACK_N_METRICS:
            entry = _entry(param.values[0], metric)
            assert entry.tolerance > 0 and entry.rationale


def test_track_n_snapshots_agree_within_the_registered_tolerance(pair) -> None:
    engine, ref, other = pair
    for metric, err in track_n_errors(ref, other).items():
        assert err <= _entry(engine, metric).tolerance, (
            f"{engine.value} {metric}: {err:.3g} exceeds the registered tolerance"
        )


def test_track_n_is_not_vacuous(pair) -> None:
    _, ref, _ = pair
    assert float(np.abs(ref.J_c).max()) > 1e-3
    assert float(np.abs(ref.Jdot_v).max()) > 1e-6
    assert float(np.abs(ref.h).max()) > 1.0


def test_track_n_grip_frame_inertias_and_topology_agree(pair) -> None:
    _, ref, other = pair
    r, o = ref.snapshot, other.snapshot
    np.testing.assert_allclose(
        o.grip_frame.rotation, r.grip_frame.rotation, rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        o.grip_frame.origin, r.grip_frame.origin, rtol=0, atol=1e-8
    )
    assert o.bare_club.mass == pytest.approx(r.bare_club.mass, abs=1e-12)
    np.testing.assert_allclose(o.bare_club.com, r.bare_club.com, atol=1e-9)
    np.testing.assert_allclose(o.bare_club.inertia, r.bare_club.inertia, atol=1e-9)
    assert len(o.welded_hand_inertias) == len(r.welded_hand_inertias)
    np.testing.assert_allclose(
        sorted(h.mass for h in o.welded_hand_inertias),
        sorted(h.mass for h in r.welded_hand_inertias),
        atol=1e-12,
    )
    assert o.topology.closure_kind is r.topology.closure_kind
    assert (o.topology.closing_side, o.topology.holding_side) == (
        r.topology.closing_side,
        r.topology.holding_side,
    )
    assert other.actuated == ref.actuated
    assert other.loop_support == ref.loop_support


def test_projected_state_satisfies_every_engines_closure(pair) -> None:
    for side in pair[1:]:
        scale = max(1.0, float(np.max(np.abs(side.v))))
        assert float(np.max(np.abs(side.J_c @ side.v))) / scale < 1e-10


# -- Track S ---------------------------------------------------------------


@pytest.fixture
def effort(pair, reference_adapter):
    _, ref, _ = pair
    order = tuple(reference_adapter.coordinate_order)
    tau = seeded_effort(order, ref.actuated)
    idx = [i for i, n in enumerate(order) if n in ref.actuated]
    return tau, idx


def test_track_s_constrained_acceleration_and_grip_wrench(pair, effort) -> None:
    _, ref, other = pair
    tau, _ = effort
    n = ref.M.shape[0]
    x_ref, x_other = kkt_solve(ref, tau), kkt_solve(other, tau)
    bound = kkt_bound(ref, other, float(np.linalg.norm(x_ref)), 1.0)
    scale = max(1.0, float(np.linalg.norm(x_ref)))
    assert bound / scale < INFORMATIVE
    assert float(np.linalg.norm(x_other - x_ref)) <= bound
    assert float(np.linalg.norm(x_other[n:] - x_ref[n:])) <= bound  # grip wrench
    assert float(np.linalg.norm(x_ref[n:])) > 1e-3  # a real wrench is carried


def test_track_s_energy_power_and_closure_spectrum(pair, effort) -> None:
    _, ref, other = pair
    tau, _ = effort
    v = ref.v
    d_m = float(np.linalg.norm(other.M - ref.M, 2))
    d_h = float(np.linalg.norm(other.h - ref.h))
    d_j = float(np.linalg.norm(other.J_c - ref.J_c, 2))
    kinetic = lambda s: 0.5 * float(v @ s.M @ v)  # noqa: E731
    assert abs(kinetic(other) - kinetic(ref)) <= 0.5 * float(v @ v) * d_m
    power = lambda s: float(v @ (tau - s.h))  # noqa: E731
    assert abs(power(other) - power(ref)) <= float(np.linalg.norm(v)) * d_h
    s_ref = np.linalg.svd(ref.J_c, compute_uv=False)
    s_other = np.linalg.svd(other.J_c, compute_uv=False)
    assert float(np.max(np.abs(s_other - s_ref))) <= d_j  # Weyl
    assert int(np.sum(s_ref > 1e-9)) == int(np.sum(s_other > 1e-9)) == 6
    # The grip wrench does no work on the closure tangent space.
    for side in (ref, other):
        lam = kkt_solve(side, tau)[side.M.shape[0] :]
        assert abs(float((side.J_c @ v) @ lam)) <= 1e-9 * max(1.0, float(lam @ lam))


def test_track_s_grip_wrench_envelope(pair, effort) -> None:
    _, ref, other = pair
    tau, idx = effort
    n = ref.M.shape[0]
    lo_r, hi_r = multiplier_envelope(ref, tau, idx)
    lo_o, hi_o = multiplier_envelope(other, tau, idx)
    x_ref = kkt_solve(ref, tau)
    resp_norm = float(np.linalg.norm(effort_response(ref, idx), 2))
    offset = kkt_bound(ref, other, float(np.linalg.norm(x_ref)), 1.0)
    response = np.sqrt(len(idx)) * kkt_bound(ref, other, resp_norm, 0.0)
    bound = offset + response
    assert np.all(hi_r > lo_r) and np.all(hi_o > lo_o)
    assert float(np.max(np.abs(hi_o - hi_r))) <= bound
    assert float(np.max(np.abs(lo_o - lo_r))) <= bound
    assert bound / max(1.0, float(np.max(hi_r - lo_r))) < INFORMATIVE


# -- Harness self-checks -----------------------------------------------------


def test_a_perturbed_engine_fails_track_n_and_stays_inside_the_track_s_bound(
    pair, effort
) -> None:
    """The harness discriminates: a 1e-6 relative error in ``M`` is caught by
    Track N, and the propagated Track S bound grows with it (still sound)."""
    engine, ref, other = pair
    tau, _ = effort
    bad = dataclasses.replace(other, M=other.M * (1.0 + 1e-6))
    assert track_n_errors(ref, bad)["M"] > _entry(engine, "M").tolerance
    x_ref, x_bad = kkt_solve(ref, tau), kkt_solve(bad, tau)
    bound = kkt_bound(ref, bad, float(np.linalg.norm(x_ref)), 1.0)
    assert float(np.linalg.norm(x_bad - x_ref)) <= bound
    assert bound > kkt_bound(ref, other, float(np.linalg.norm(x_ref)), 1.0)


def test_vocabulary_mismatch_is_refused(pair) -> None:
    _, ref, other = pair
    with pytest.raises(ValueError, match="coordinate vocabulary"):
        to_reference_order(other.snapshot, ("not_a_coordinate",))


# -- Capability matrix -----------------------------------------------------


def test_reference_engine_is_full(reference_adapter) -> None:
    level = reference_adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT)
    assert level is CapabilityLevel.FULL


@pytest.mark.parametrize("engine", list(EngineType))
def test_synthetic_fixtures_never_claim_conformance(engine) -> None:
    """Quarantined fixtures declare NONE and refuse a snapshot (no silent skip)."""
    if engine is REFERENCE:
        pytest.skip("Pinocchio synthetic fixture covered by its adapter tests")
    synthetic = create_engine_force_adapter(engine, allow_synthetic=True)
    assert synthetic.capability(Capability.CLOSED_LOOP_SNAPSHOT) is (
        CapabilityLevel.NONE
    )
    nv = synthetic.nv
    zeros = np.zeros(nv)
    with pytest.raises(CapabilityUnavailable):
        snapshot_from_adapter(
            synthetic,
            zeros,
            zeros,
            zeros,
            input_provenance=InputProvenance(kind="test", source="synthetic"),
        )


def test_simscape_native_adapter_fails_closed() -> None:
    """Simscape has no native adapter: production routes refuse (capability NONE)."""
    with pytest.raises(RuntimeError, match="Simscape force adapter is not yet"):
        create_engine_force_adapter(EngineType.SIMSCAPE, allow_synthetic=False)
