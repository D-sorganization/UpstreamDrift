"""Pinocchio grip RHS: memoised hand frames are exact (#13396).

Acceptance criteria:
- the weld forward kinematics runs once per distinct ``(hand_source, t)``
  however often the right-hand side is evaluated at that time (Radau stage
  re-evaluations and finite-difference Jacobian columns);
- derivatives from the cached path equal a fresh, uncached evaluation bit for
  bit;
- the cache is bounded and callers cannot corrupt it through the returned dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact import GripInterface, load_coordinate_swing
from src.shared.python.grip_contact.prescribed_motion import CoordinateSpline

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FIXTURES = ROOT / "tests/fixtures/club_face"


def _grip_bushing() -> Any:
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")
    from src.engines.physics_engines.pinocchio.python import grip_bushing

    return grip_bushing


class _CountingSource:
    """Pure hand source over a swing spline that counts its evaluations."""

    def __init__(self, spline: CoordinateSpline) -> None:
        self._spline = spline
        self.calls: list[float] = []

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(t)
        return self._spline.evaluate(t)


@pytest.fixture(scope="module")
def setup() -> tuple[Any, dict[str, Any], list[str], GripInterface, CoordinateSpline]:
    module = _grip_bushing()
    spec = json.loads(SPEC_PATH.read_text())
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    return module, spec, names, interface, CoordinateSpline(swing.time_s, swing.q)


def _state(sim: Any, spline: CoordinateSpline, t: float) -> np.ndarray:
    """A club state displaced from the weld so the bushings are loaded."""
    from pinocchio import Quaternion

    weld = sim.kinematics.state(*spline.evaluate(t))
    quat = Quaternion(weld.rotation)
    rng = np.random.default_rng(13384)
    return np.concatenate(
        [
            weld.position_m + rng.normal(scale=1e-3, size=3),
            [quat.x, quat.y, quat.z, quat.w],
            rng.normal(scale=0.5, size=6),
        ]
    )


def test_hand_kinematics_run_once_per_time_and_match_fresh(setup: Any) -> None:
    module, spec, names, interface, spline = setup
    cached = module.ClubOnBushings(spec, names, interface)
    source = _CountingSource(spline)
    times = [0.01, 0.0125, 0.015]
    y0 = _state(cached, spline, times[0])
    for t in times:
        for k in range(14):  # one RHS plus 13 finite-difference columns
            y = y0.copy()
            y[k % 13] += 1e-8 * k
            got = cached.derivatives(t, y, source)
            fresh = module.ClubOnBushings(spec, names, interface)
            want = fresh.derivatives(t, y, spline.evaluate)
            np.testing.assert_array_equal(got, want)
    assert source.calls == times


def test_cache_is_bounded_and_isolated(setup: Any) -> None:
    module, spec, names, interface, spline = setup
    sim = module.ClubOnBushings(spec, names, interface)
    source = _CountingSource(spline)
    for k in range(3 * module.HAND_CACHE_SIZE):
        sim.hand_frames(0.01 + 1e-4 * k, source)
    assert len(sim._hand_cache) == module.HAND_CACHE_SIZE  # noqa: SLF001
    hands = sim.hand_frames(0.5, source)
    hands.clear()
    assert set(sim.hand_frames(0.5, source)) == set(module.SIDES)
    assert source.calls.count(0.5) == 1


def test_derivatives_build_the_club_state_once(setup: Any) -> None:
    """One ``club_state`` per right-hand side; the result is unchanged."""
    module, spec, names, interface, spline = setup
    sim = module.ClubOnBushings(spec, names, interface)
    y = _state(sim, spline, 0.01)
    want = module.ClubOnBushings(spec, names, interface).derivatives(
        0.01, y, spline.evaluate
    )
    original = sim.club_state
    calls: list[int] = []

    def counting(state: np.ndarray) -> Any:
        calls.append(1)
        return original(state)

    sim.club_state = counting
    got = sim.derivatives(0.01, y, spline.evaluate)
    assert len(calls) == 1
    np.testing.assert_array_equal(got, want)
