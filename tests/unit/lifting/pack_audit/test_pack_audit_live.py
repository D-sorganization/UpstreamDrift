"""Live four-engine smoke and parity; skips when packs or engines are missing."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _baseline():
    for mod in ("mujoco", "pinocchio", "opensim", "pydrake.all"):
        pytest.importorskip(mod)
    from src.shared.python.lifting.pack_audit.baseline import run_baseline
    from src.shared.python.lifting.pack_audit.names import ENGINES
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    if any(locate_pack(e) is None for e in ENGINES):
        pytest.skip("lift model pack checkouts not found (set LIFT_PACK_ROOT)")
    return run_baseline(lifts=("deadlift",))


@pytest.fixture(scope="module")
def receipt():
    return _baseline()


def test_all_engines_load_and_step(receipt):
    for res in receipt["results"]["deadlift"].values():
        assert res["smoke"]["loaded"] and res["smoke"]["stepped"]


def test_same_q_feet_agree_across_engines(receipt):
    comp = receipt["comparisons"]["deadlift"]["poses"]
    for pairs in comp.values():
        for metrics in pairs.values():
            assert metrics["feet_max_m"] < 0.02


def test_total_mass_agrees(receipt):
    assert receipt["comparisons"]["deadlift"]["mass"]["spread_kg"] < 1e-3


def test_drake_inertia_is_about_com_not_body_origin():
    """Drake ``inertia_diag`` is COM-referenced (no parallel-axis term), #12073."""
    pytest.importorskip("pydrake.all")
    import numpy as np

    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack("drake")
    if pack is None:
        pytest.skip("Drake lift pack checkout not found (set LIFT_PACK_ROOT)")
    adapter = create_adapter(pack, "deadlift", Anthropometry(80.0, 1.78, 50.0))
    reported = adapter.structure()["inertia_diag"]
    for name in adapter._bodies():
        si = adapter.plant.GetBodyByName(name).default_spatial_inertia()
        g = si.get_unit_inertia().ShiftToCenterOfMass(si.get_com())
        expected = si.get_mass() * np.asarray(g.get_moments())
        np.testing.assert_allclose(reported[name], expected, rtol=1e-6, atol=1e-12)
