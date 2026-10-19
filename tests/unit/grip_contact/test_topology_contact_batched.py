"""Batched topology contact law equals the per-pair reference (#13384, #13304 follow-up).

Acceptance criteria:
- ``topology_side_wrench`` (one batched pass per side) equals the
  per-pair reference ``topology_side_wrench_reference`` to <= 1e-12 N (and
  N m) at random poses and velocities, for every grip style with and without
  the trigger finger;
- inputs are validated once per side call, and the per-pair kernels are not
  called on the batched path;
- the float segment query and the stacked transform reproduce the scalar
  reference bit for bit, including the parallel and degenerate branches;
- the batched path is at least 6x faster than the reference in CPU time
  (measured ~10x; see the PR and the HRG calculation reference).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import BushingState, GripInterface
from src.shared.python.grip_contact import topology_contact as tc
from src.shared.python.grip_contact.batched_pair_contact import (
    closest_points_segments_floats,
    rigid_points_floats,
)
from src.shared.python.grip_contact.contact_topology import GripStyle, contact_topology
from src.shared.python.grip_contact.pair_contact import (
    PairWrench,
    closest_points_segments,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_side_wrench,
    topology_side_wrench_reference,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
ATOL_N = 1e-12
CASES = [(style, trig) for style in GripStyle for trig in (False, True)]


@pytest.fixture(scope="module")
def interface() -> GripInterface:
    return GripInterface.from_spec(json.loads(SPEC_PATH.read_text()))


def _model(
    interface: GripInterface, style: GripStyle, trigger: bool
) -> TopologyContactModel:
    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger))
    return build_topology_model(interface, topo, squeeze_per_hand_n=1100.0)


def _rotation(rng: np.random.Generator, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation about a random axis by a random angle up to ``angle_rad``."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(-angle_rad, angle_rad)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _states(rng: np.random.Generator, slip: bool) -> tuple[BushingState, BushingState]:
    """Club in a random world pose; hand perturbed about the weld pose."""
    r_club = _rotation(rng, np.pi)
    p_club = rng.normal(scale=0.5, size=3)
    r_hand = r_club @ _rotation(rng, 0.08)
    p_hand = p_club + r_club @ rng.normal(scale=1.5e-3, size=3)
    v_club = rng.normal(scale=2.0, size=3)
    w_club = rng.normal(scale=5.0, size=3)
    if slip:
        v_hand = v_club + rng.normal(scale=rng.choice([1e-4, 1e-3, 1e-1]), size=3)
        w_hand = w_club + rng.normal(scale=2.0, size=3)
    else:  # no relative motion: the zero-slip friction branch
        v_hand, w_hand = v_club.copy(), w_club.copy()
        p_hand = p_club.copy()
        r_hand = r_club.copy()
    if rng.random() < 0.3:  # Fortran-ordered rotations take the same BLAS path
        r_hand = np.asfortranarray(r_hand)
    hand = BushingState(r_hand, p_hand, v_hand, w_hand)
    club = BushingState(r_club, p_club, v_club, w_club)
    return hand, club


def _assert_pair_equal(batched: PairWrench, ref: PairWrench) -> None:
    for name in ("force_a_n", "force_b_n", "moment_a_nm", "moment_b_nm"):
        np.testing.assert_allclose(
            getattr(batched, name), getattr(ref, name), rtol=0, atol=ATOL_N
        )
    for name in ("contact_point_m", "normal", "point_a_m", "point_b_m"):
        np.testing.assert_allclose(
            getattr(batched, name), getattr(ref, name), rtol=0, atol=1e-15
        )
    assert abs(batched.penetration_m - ref.penetration_m) <= 1e-15
    assert abs(batched.normal_force_mag_n - ref.normal_force_mag_n) <= ATOL_N
    assert abs(batched.tangential_force_mag_n - ref.tangential_force_mag_n) <= ATOL_N


@pytest.mark.parametrize(("style", "trigger"), CASES)
def test_batched_side_wrench_equals_per_pair_reference(
    interface: GripInterface, style: GripStyle, trigger: bool
) -> None:
    """Batched resultant and every pair wrench equal the reference to <= 1e-12 N."""
    model = _model(interface, style, trigger)
    rng = np.random.default_rng(13304 + 2 * list(GripStyle).index(style) + trigger)
    active = separated = 0
    worst = 0.0
    for k in range(60):
        hand, club = _states(rng, slip=k % 6 != 0)
        for side in ("L", "R"):
            got = topology_side_wrench(model, side, hand, club)
            ref = topology_side_wrench_reference(model, side, hand, club)
            worst = max(worst, float(np.abs(got.force_n - ref.force_n).max()))
            np.testing.assert_allclose(got.force_n, ref.force_n, rtol=0, atol=ATOL_N)
            np.testing.assert_allclose(
                got.moment_nm, ref.moment_nm, rtol=0, atol=ATOL_N
            )
            np.testing.assert_allclose(
                got.normal_force_n, ref.normal_force_n, rtol=0, atol=ATOL_N
            )
            np.testing.assert_allclose(
                got.tangential_force_n, ref.tangential_force_n, rtol=0, atol=ATOL_N
            )
            assert got.total_normal_force_n == pytest.approx(
                ref.total_normal_force_n, abs=ATOL_N
            )
            assert list(got.pair_wrenches) == list(ref.pair_wrenches)
            for name, ref_pw in ref.pair_wrenches.items():
                _assert_pair_equal(got.pair_wrenches[name], ref_pw)
                if ref_pw.penetration_m > 0.0:
                    active += 1
                else:
                    separated += 1
    assert active > 0 and separated > 0, (active, separated)
    assert worst <= ATOL_N


def test_batched_side_wrench_validates_once_and_skips_pair_kernels(
    interface: GripInterface, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DbC runs once per side call; no per-pair kernel is called."""
    model = _model(interface, GripStyle.TEN_FINGER, True)
    hand, club = _states(np.random.default_rng(7), slip=True)
    calls: list[str] = []
    original = tc._validate_bushing_state  # noqa: SLF001

    def counting(state: BushingState, name: str) -> None:
        calls.append(name)
        original(state, name)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("per-pair kernel called on the batched path")

    monkeypatch.setattr(tc, "_validate_bushing_state", counting)
    monkeypatch.setattr(tc, "sphere_capsule", forbidden)
    monkeypatch.setattr(tc, "capsule_capsule", forbidden)
    topology_side_wrench(model, "R", hand, club)
    assert calls == ["hand_frame", "club_frame"]


def test_batched_side_wrench_rejects_improper_rotation(
    interface: GripInterface,
) -> None:
    model = _model(interface, GripStyle.OVERLAP, False)
    still = np.zeros(3)
    good = BushingState(np.eye(3), still, still, still)
    mirror = BushingState(np.diag([1.0, 1.0, -1.0]), still, still, still)
    with pytest.raises(ValueError, match="proper rotation"):
        topology_side_wrench(model, "L", mirror, good)
    with pytest.raises(ValueError, match="side must be one of"):
        topology_side_wrench(model, "X", good, good)
    with pytest.raises(TypeError, match="TopologyContactModel"):
        topology_side_wrench(object(), "L", good, good)  # type: ignore[arg-type]


def test_mode_filter_matches_reference(interface: GripInterface) -> None:
    """A mode with no physical pairs yields an empty, zero side wrench."""
    model = _model(interface, GripStyle.INTERLOCK, True)
    hand, club = _states(np.random.default_rng(3), slip=True)
    for mode in ("contact", "allocated"):
        got = topology_side_wrench(model, "R", hand, club, mode=mode)
        ref = topology_side_wrench_reference(model, "R", hand, club, mode=mode)
        assert list(got.pair_wrenches) == list(ref.pair_wrenches)
        np.testing.assert_allclose(got.force_n, ref.force_n, rtol=0, atol=ATOL_N)
        np.testing.assert_allclose(got.moment_nm, ref.moment_nm, rtol=0, atol=ATOL_N)


def _segment(rng: np.random.Generator, kind: str) -> tuple[np.ndarray, np.ndarray]:
    a0 = rng.normal(scale=0.05, size=3)
    if kind == "point":
        return a0, a0.copy()
    return a0, a0 + rng.normal(scale=0.05, size=3)


@pytest.mark.parametrize("kind", ["general", "parallel", "point_a", "point_b"])
def test_float_segment_query_matches_scalar_closest_points(kind: str) -> None:
    """The Python-float segment/segment query equals the scalar reference exactly."""
    rng = np.random.default_rng(len(kind))
    b0, b1 = _segment(rng, "point" if kind == "point_b" else "general")
    for _ in range(400):
        if kind == "parallel":
            a0 = rng.normal(scale=0.05, size=3)
            a1 = a0 + rng.uniform(-2.0, 2.0) * (b1 - b0)
        else:
            a0, a1 = _segment(rng, "point" if kind == "point_a" else "general")
        pa, pb = closest_points_segments_floats(
            a0.tolist(), a1.tolist(), b0.tolist(), b1.tolist()
        )
        ra, rb = closest_points_segments(a0, a1, b0, b1)
        np.testing.assert_array_equal(pa, ra)
        np.testing.assert_array_equal(pb, rb)


def test_stacked_transform_matches_scalar() -> None:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(300, 3))
    for rot in (_rotation(rng, np.pi), np.asfortranarray(_rotation(rng, np.pi))):
        pos = rng.normal(size=3)
        world = rigid_points_floats(rot, pos, points[:, None, :])
        for k, pt in enumerate(points):
            assert world[k] == tuple((pos + rot @ pt).tolist())


def _best_cpu_time(fn: object, repeats: int = 7, number: int = 40) -> float:
    """Best per-call thread CPU time (robust to a loaded host's scheduling)."""
    best = float("inf")
    for _ in range(repeats):
        start = time.thread_time()
        for _ in range(number):
            fn()  # type: ignore[operator]
        best = min(best, (time.thread_time() - start) / number)
    return best


@pytest.mark.benchmark
def test_batched_side_wrench_is_faster_than_reference(
    interface: GripInterface,
) -> None:
    """Regression guard on the speedup (>= 6x here; ~10x measured)."""
    model = _model(interface, GripStyle.TEN_FINGER, True)
    hand, club = _states(np.random.default_rng(5), slip=True)
    batched = _best_cpu_time(lambda: topology_side_wrench(model, "R", hand, club))
    reference = _best_cpu_time(
        lambda: topology_side_wrench_reference(model, "R", hand, club)
    )
    assert reference / batched >= 6.0, (reference, batched)
