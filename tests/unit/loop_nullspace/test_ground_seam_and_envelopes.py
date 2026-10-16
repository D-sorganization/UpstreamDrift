"""NSX-60: the adapter seam for foot contacts and the ground envelope quantities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from src.shared.python.engine_core.capabilities import CapabilityUnavailable  # noqa: E402
from src.shared.python.loop_nullspace.contracts import InputProvenance  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import decompose  # noqa: E402
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.fixtures import (  # noqa: E402
    spatial_two_arm_club_snapshot,
)
from src.shared.python.loop_nullspace.ground_envelopes import (  # noqa: E402
    closure_wrench_quantities,
    ground_force_quantities,
)
from src.shared.python.loop_nullspace.snapshot import (  # noqa: E402
    snapshot_from_adapter,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (  # noqa: E402
    EngineType,
    create_engine_force_adapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
    / "full_body_spec_hipcal_scaled.json"
)
PROVENANCE = InputProvenance("test", "ground seam")


@pytest.fixture(scope="module")
def adapter():
    return create_engine_force_adapter(
        EngineType.PINOCCHIO, SPEC, allow_synthetic=False
    )


@pytest.fixture(scope="module")
def state(adapter):
    """Zero pose lifted so the lowest sphere sits 1 mm above the ground."""
    lift = adapter.coordinate_order.index("TranslationInputZ")

    def lowest_gap(height: float) -> float:
        q = np.zeros(adapter.nv)
        q[lift] = height
        return min(c.gap_m for c in adapter.foot_contact_candidates(q))

    slope = (lowest_gap(1.0) - lowest_gap(0.0)) / 1.0  # gaps are affine in the lift
    q = np.zeros(adapter.nv)
    q[lift] = (0.001 - lowest_gap(0.0)) / slope
    assert lowest_gap(q[lift]) == pytest.approx(0.001, abs=1e-9)
    return q, np.zeros(adapter.nv), np.zeros(adapter.nv)


def test_candidates_follow_contact_names_with_unit_ground_normal(
    adapter, state
) -> None:
    cands = adapter.foot_contact_candidates(state[0])
    assert tuple(c.contact.name for c in cands) == adapter.contact_names
    for c in cands:
        assert np.allclose(c.contact.normal, [0.0, 0.0, 1.0])
        assert c.contact.mu == pytest.approx(0.9)  # spec static_friction
    # point of each candidate is where compute_contact_jacobian evaluates its rows
    assert adapter.compute_contact_jacobian(state[0]).shape == (
        3 * len(cands),
        adapter.nv,
    )


def test_snapshot_keeps_ground_contact_off_by_default(adapter, state) -> None:
    snap = snapshot_from_adapter(adapter, *state, input_provenance=PROVENANCE)
    assert snap.J_g is None and snap.foot_contacts == ()


def test_snapshot_selects_contacts_within_the_tolerance(adapter, state) -> None:
    q = state[0]
    cands = adapter.foot_contact_candidates(q)
    gaps = np.array([c.gap_m for c in cands])
    tol = float(np.sort(gaps)[2])  # the three lowest spheres
    assert tol > 0.0
    snap = snapshot_from_adapter(
        adapter, *state, input_provenance=PROVENANCE, foot_contact_tol_m=tol
    )
    keep = [k for k, c in enumerate(cands) if c.gap_m <= tol]
    assert [c.name for c in snap.foot_contacts] == [cands[k].contact.name for k in keep]
    full = adapter.compute_contact_jacobian(q)
    expected = np.vstack([full[3 * k : 3 * k + 3] for k in keep])
    assert np.array_equal(snap.J_g, expected)


def test_snapshot_rejects_a_negative_tolerance(adapter, state) -> None:
    with pytest.raises(ValueError, match="gap_tol_m"):
        snapshot_from_adapter(
            adapter, *state, input_provenance=PROVENANCE, foot_contact_tol_m=-1.0
        )


def test_snapshot_needs_an_adapter_that_exposes_contacts(adapter, state) -> None:
    class NoFeet:
        def __getattr__(self, name):
            if name == "foot_contact_candidates":
                raise AttributeError(name)
            return getattr(adapter, name)

    with pytest.raises(CapabilityUnavailable, match="foot_contact_candidates"):
        snapshot_from_adapter(
            NoFeet(), *state, input_provenance=PROVENANCE, foot_contact_tol_m=0.0
        )


def _empty_hypotheses():
    from src.shared.python.loop_nullspace.registry import HypothesisSet

    return HypothesisSet("t", "None", (), ())


@pytest.fixture(scope="module")
def spatial_set():
    snap, _ = spatial_two_arm_club_snapshot(
        floating_root=True, feet=True, zero_root_pose=True, foot_points=2
    )
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    fs = FeasibleSet.from_hypotheses(dec, _empty_hypotheses())
    return snap, dec, fs


def test_closure_wrench_quantities_are_the_multiplier_block(spatial_set) -> None:
    _, dec, fs = spatial_set
    quantities = closure_wrench_quantities()(fs)
    assert [q.name for q in quantities] == [
        "lambda_fx",
        "lambda_fy",
        "lambda_fz",
        "lambda_tx",
        "lambda_ty",
        "lambda_tz",
    ]
    lead = dec.n_actuated
    for k, q in enumerate(quantities):
        assert np.array_equal(q.coeff, dec.N[lead + k])
        assert q.const == dec.x_p[lead + k]
    assert [q.units for q in quantities] == ["N"] * 3 + ["N m"] * 3


def test_ground_force_quantities_sum_the_declared_group(spatial_set) -> None:
    snap, dec, fs = spatial_set
    names = [c.name for c in snap.foot_contacts]
    quantities = ground_force_quantities(
        lambda _fs: names, {"first": lambda n: n == names[0], "none": lambda n: False}
    )(fs)
    assert [q.name for q in quantities] == [
        "ground_fx",
        "ground_fy",
        "ground_fz",
        "first",
        "none",
    ]
    lead = dec.n_actuated + dec.m
    z = np.random.default_rng(3).normal(size=dec.nullity)
    x = dec.x_p + dec.N @ z
    f = x[lead:].reshape(-1, 3)
    values = [q.const + q.coeff @ z for q in quantities]
    assert np.allclose(values[:3], f.sum(axis=0))
    assert values[3] == pytest.approx(f[0, 2])
    assert values[4] == 0.0


def test_ground_force_quantities_reject_mismatched_names(spatial_set) -> None:
    _, _, fs = spatial_set
    with pytest.raises(ValueError, match="contact names"):
        ground_force_quantities(lambda _fs: ["only"])(fs)
