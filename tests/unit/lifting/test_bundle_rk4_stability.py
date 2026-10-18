"""Every committed lift bundle is integrated inside the RK4 stability region.

A same-input bundle records the stiffest open-loop rate of its reference run
(``provenance["rk4_stability"]``).  Before #13041 only bar-loaded bundles
were given more substeps, so the deadlift ran at the default eight with
``|z| = 4.24`` against the real stability limit 2.785 (amplification 6.5):
its foot coordinates drifted 1e-4 rad between engines.  The bundle builder
now picks the substep count from the recorded rate, so no committed bundle
may exceed the limit.  The limit itself is never relaxed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.lifting.lift_recipes import bundle_lifts, recipe
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.same_input.stiffness import (
    RK4_REAL_STABILITY_LIMIT,
)

pytestmark = pytest.mark.unit

_EVIDENCE = Path(__file__).resolve().parents[3] / "docs/development/lifting/evidence"
_BUNDLES = {
    **{lift: recipe(lift).bundle_path for lift in bundle_lifts()},
    "back_squat_barbell": _EVIDENCE / "back_squat/back_squat_barbell_bundle.npz",
}


@pytest.mark.parametrize("name", sorted(_BUNDLES))
def test_committed_bundle_is_inside_the_rk4_stability_region(name: str) -> None:
    bundle = InputBundle.load(_BUNDLES[name])
    record = bundle.provenance["rk4_stability"]
    assert record["real_stability_limit"] == RK4_REAL_STABILITY_LIMIT
    assert record["substep_s"] == pytest.approx(bundle.dt_s / bundle.substeps)
    assert record["max_abs_z"] <= RK4_REAL_STABILITY_LIMIT, (name, record)
    assert record["max_amplification"] <= 1.0, (name, record)
