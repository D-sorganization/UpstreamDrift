"""The grip-weld pair of an injected multiplier (#15150 Track Z, Z5).

One part of the ZTCF/ZVCF split solves its own closure multiplier; the
overlay source turns it into the same +/- wrench pair that
``closure_wrench`` builds from its own solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.overlay_source import (  # noqa: E402
    MujocoOverlaySource,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    load_full_body_spec,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture(scope="module")
def case():
    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    source = MujocoOverlaySource(json.dumps(spec).encode("utf-8"))
    rng = np.random.default_rng(15150)
    names = source.coordinate_order
    q = dict(zip(names, rng.normal(0.0, 0.05, len(names)), strict=True))
    v = dict(zip(names, rng.normal(0.0, 0.3, len(names)), strict=True))
    e = dict(zip(names, rng.normal(0.0, 20.0, len(names)), strict=True))
    return source, q, v, e


def test_the_pair_of_the_solved_multiplier_is_the_closure_wrench(case) -> None:
    source, q, v, e = case
    _, multiplier = source.solve_with_multipliers(q, v, e)
    solved = source.closure_wrench(q, v, e)
    assert solved.available
    source.closure_wrench(
        {n: x + 0.1 for n, x in q.items()}, v, e
    )  # move the state away first
    injected = source.closure_from_multiplier(q, multiplier)
    assert injected.available
    assert injected.multiplier_world == pytest.approx(solved.multiplier_world)
    for got, want in zip(injected.wrenches, solved.wrenches, strict=True):
        assert got.body == want.body
        assert got.point_m == pytest.approx(want.point_m, abs=1e-12)
        assert got.force_n == pytest.approx(want.force_n)
        assert got.torque_nm == pytest.approx(want.torque_nm)


@pytest.mark.parametrize(
    "multiplier",
    [np.ones(5), np.array([1.0, 2.0, np.nan, 0.0, 0.0, 0.0])],
    ids=["five", "nan"],
)
def test_a_multiplier_that_is_not_a_finite_6_vector_is_refused(
    case, multiplier
) -> None:
    source, q, _, _ = case
    with pytest.raises(ValueError, match="multiplier"):
        source.closure_from_multiplier(q, multiplier)
