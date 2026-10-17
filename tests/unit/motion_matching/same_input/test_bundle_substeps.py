"""Per-bundle RK4 substeps and the stiff-mode substep rule (OSL-16, #13104).

A bar-loaded lift stiffens the foot contact beyond what eight 0.125 ms RK4
substeps integrate stably, so a bundle may record its own substep count; every
replay reads it back through :meth:`InputBundle.step_policy`.  Bundles without
the field keep the original eight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    ROOT_COORDINATES,
    InputBundle,
)
from src.shared.python.motion_matching.same_input.integrator import (
    DEFAULT_SUBSTEPS,
    effort_holds_for,
)
from src.shared.python.motion_matching.same_input.stiffness import (
    RK4_REAL_STABILITY_LIMIT,
    rk4_amplification,
    stiffest_rate,
    substeps_for,
)

pytestmark = pytest.mark.unit

ORDER = (*ROOT_COORDINATES, "Joint")
SPEC = json.dumps({"coordinate_order": list(ORDER)}).encode()
STEPS = 4


def _bundle(substeps: int = DEFAULT_SUBSTEPS, holds: int = 1) -> InputBundle:
    nv = len(ORDER)
    zeros = np.zeros((STEPS + 1, nv))
    return InputBundle(
        spec_bytes=SPEC,
        coordinate_order=ORDER,
        dt_s=1e-3,
        q0=np.zeros(nv),
        v0=np.zeros(nv),
        efforts=np.zeros((STEPS, nv)),
        reference_q=zeros,
        reference_v=zeros.copy(),
        reference_engine="mujoco",
        effort_holds=holds,
        hold_efforts=None if holds == 1 else np.zeros((STEPS * holds, nv)),
        substeps=substeps,
    )


def test_default_bundle_keeps_eight_substeps() -> None:
    bundle = _bundle()
    assert bundle.substeps == DEFAULT_SUBSTEPS == 8
    assert bundle.step_policy().substeps == 8
    assert bundle.manifest()["policy"]["substeps"] == 8


def test_bundle_records_and_replays_its_substeps(tmp_path: Path) -> None:
    bundle = _bundle(substeps=32, holds=4)
    policy = bundle.manifest()["policy"]
    assert policy["substeps"] == 32
    assert policy["effort_substeps"] == 8
    path = tmp_path / "loaded.npz"
    bundle.save(path)
    loaded = InputBundle.load(path)
    assert loaded.substeps == 32
    assert loaded.step_policy().substeps == 32
    assert loaded.step_policy().substeps_per_hold == 8
    assert loaded.effort_substeps == 8
    assert loaded.manifest() == bundle.manifest()


def test_legacy_manifest_without_the_field_loads_eight(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    _bundle().save(path)
    assert InputBundle.load(path).substeps == 8


@pytest.mark.parametrize("substeps", [0, -8, 12])
def test_bundle_rejects_substeps_the_holds_do_not_divide(substeps: int) -> None:
    with pytest.raises(ValueError):
        _bundle(substeps=substeps, holds=8)


def test_bundle_rejects_a_non_integer_substep_count() -> None:
    with pytest.raises(TypeError):
        _bundle(substeps=8.0)  # type: ignore[arg-type]


def test_effort_holds_for_accepts_the_bundle_substeps() -> None:
    assert effort_holds_for(2.5e-4, 1e-3, 32) == 4
    with pytest.raises(ValueError):
        effort_holds_for(1e-3 / 16, 1e-3, 8)
    assert effort_holds_for(1e-3 / 16, 1e-3, 32) == 16


# -- the substep rule -------------------------------------------------------
class _Damper:
    """``v' = -rate * v`` on every coordinate: one real mode at ``-rate``."""

    coordinate_order = ORDER

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def acceleration(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return -self.rate * v + tau


def test_stiffest_rate_finds_the_damper_mode() -> None:
    nv = len(ORDER)
    rate = stiffest_rate(_Damper(2.9e4), np.zeros(nv), np.zeros(nv), np.zeros(nv))
    assert rate == pytest.approx(2.9e4, rel=1e-6)


def test_rk4_amplification_matches_the_stability_limit() -> None:
    assert rk4_amplification(-RK4_REAL_STABILITY_LIMIT) == pytest.approx(1.0, abs=1e-4)
    assert rk4_amplification(-1.0) < 1.0
    assert rk4_amplification(-3.6) > 1.0


def test_substeps_for_keeps_the_stiffest_mode_inside_the_target() -> None:
    # The loaded squat's stiffest foot modes over a 1 ms step: 8.95e4 1/s
    # gives |z| = 2.80 at 32 substeps (past the 2.785 limit), so 64; the
    # window peak 1.3e5 gives 2.03 at 64.
    assert substeps_for(8.95e4, 1e-3) == 64
    assert substeps_for(1.3e5, 1e-3) == 64
    assert substeps_for(1.0e4, 1e-3) == DEFAULT_SUBSTEPS
    assert substeps_for(1.0e4, 1e-3, target=1.0) == 16
    with pytest.raises(ValueError):
        substeps_for(-1.0, 1e-3)
    with pytest.raises(ValueError):
        substeps_for(1.0, 1e-3, target=0.0)


def test_stability_record_reports_the_stiffest_sample() -> None:
    from src.shared.python.motion_matching.same_input.stiffness import (
        rk4_stability_record,
    )

    nv = len(ORDER)
    rows = np.zeros((3, nv))
    record = rk4_stability_record(_Damper(2.9e4), rows, rows, rows, 1e-3 / 8)
    assert record["max_abs_z"] == pytest.approx(3.625, rel=1e-6)
    assert record["max_amplification"] > 1.0
    fine = rk4_stability_record(_Damper(2.9e4), rows, rows, rows, 1e-3 / 32)
    assert fine["max_abs_z"] < 1.0 and fine["max_amplification"] < 1.0


def test_hold_policy_carries_holds_and_substeps() -> None:
    from src.shared.python.motion_matching.same_input import hold_policy

    policy = hold_policy(2.5e-4, 32)
    assert (policy.substeps, policy.effort_holds, policy.substeps_per_hold) == (
        32,
        4,
        8,
    )
    assert hold_policy() == hold_policy(1e-3, 8)
    with pytest.raises(ValueError):
        hold_policy(1e-3 / 16)
