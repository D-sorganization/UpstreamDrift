"""Records each engine's start-pose conformance per lift (LIFT-3, #11743).

Runs the real pack models.  Cases where a pack fails today are
``xfail(strict=True)`` and cite the pack-repo issue, so fixing a pack turns
its case into an XPASS failure that forces this table to be updated; limits
are never loosened.  Unavailable checks (an engine that cannot supply the
quantity) skip with the adapter's reason.  Skips cleanly without the engine
or pack checkout (set ``LIFT_PACK_ROOT``).

Recorded against pack ``main`` at the commits in ``_RECORDED_AT``.  A checkout
older than that is skipped (its results predate the recording); a newer one
runs, so a pack fix surfaces as an XPASS.
"""

from __future__ import annotations

import subprocess

import pytest

from src.shared.python.lifting.pack_audit.names import ENGINES, LIFTS
from src.shared.python.lifting.pack_audit.start_conformance import CHECK_NAMES

pytestmark = pytest.mark.unit

_ENGINE_MODULE = {
    "mujoco": "mujoco",
    "opensim": "opensim",
    "drake": "pydrake.all",
    "pinocchio": "pinocchio",
}
_PULLS = ("deadlift", "snatch", "clean_and_jerk")
_RECORDED_AT = {
    "mujoco": "284b9b2",
    "opensim": "16cb5cc",
    "drake": "18118ac",
    "pinocchio": "5939024",
}
_NONE = "no dedicated pack issue yet (tracked in UpstreamDrift#11743)"

_KNOWN_FAILURES: dict[tuple[str, str, str], str] = {}


def _fail(engines, lifts, check, reason):
    for engine in engines:
        for lift in lifts:
            _KNOWN_FAILURES[(engine, lift, check)] = reason


_fail(("mujoco",), _PULLS, "bar_start", "MuJoCo_Models#407 bar not on the floor")
_fail(("drake",), _PULLS, "bar_start", "Drake_Models#381 bar 0.7-0.8 m up")
_fail(("opensim",), _PULLS, "bar_start", "OpenSim_Models#416 bar 0.67-0.83 m up")
_fail(("pinocchio",), _PULLS, "bar_start", "Pinocchio_Models#450 not grounded")
_fail(("drake",), ("bench_press",), "bar_start", "Drake_Models#402 bar off-centre")
_fail(("mujoco",), ("bench_press",), "feet_flat_in_contact", "MuJoCo_Models#407")
_fail(("drake",), ("bench_press",), "feet_flat_in_contact", "Drake_Models#364")
_fail(("opensim",), ("bench_press",), "feet_flat_in_contact", "OpenSim_Models#384")
_fail(("mujoco",), ("bench_press",), "joint_limits", "MuJoCo_Models#409")
_fail(
    ("mujoco",),
    ("squat", "deadlift", "snatch", "clean_and_jerk"),
    "joint_limits",
    f"MuJoCo hip_flex phase reference above range; {_NONE}",
)
_fail(("drake",), ("bench_press",), "joint_limits", f"Drake elbow; {_NONE}")
_fail(
    ("pinocchio",),
    ("squat", "snatch", "clean_and_jerk"),
    "joint_limits",
    "Pinocchio_Models#433 ledgered joint limits",
)
_fail(("mujoco",), LIFTS, "static_grf", "MuJoCo_Models#406 no settled ground force")
_fail(("opensim",), LIFTS, "static_grf", "OpenSim_Models#391 / #384 contact force")


@pytest.fixture(scope="module")
def _results():
    return {}


def _at_least_recorded(pack) -> bool:
    """True when the checkout contains the commit this table was recorded at."""
    if not pack.commit:
        return False
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [  # noqa: S607
                "git",
                "-C",
                str(pack.root),
                "merge-base",
                "--is-ancestor",
                _RECORDED_AT[pack.engine],
                pack.commit,
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _conformance(results, engine, lift):
    key = (engine, lift)
    if key in results:
        return results[key]
    pytest.importorskip(_ENGINE_MODULE[engine])
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack
    from src.shared.python.lifting.pack_audit.start_conformance import (
        evaluate_start_conformance,
    )

    pack = locate_pack(engine)
    if pack is None:
        pytest.skip(f"{engine} lift pack checkout not found (set LIFT_PACK_ROOT)")
    if not _at_least_recorded(pack):
        pytest.skip(f"{engine} pack checkout predates {_RECORDED_AT[engine]}")
    try:
        adapter = create_adapter(pack, lift, Anthropometry())
        results[key] = evaluate_start_conformance(adapter, pack)
    except (ImportError, OSError) as exc:
        pytest.skip(f"{engine} {lift} not loadable: {exc}")
    return results[key]


def _params():
    for engine in ENGINES:
        for lift in LIFTS:
            for check in CHECK_NAMES:
                marks = []
                reason = _KNOWN_FAILURES.get((engine, lift, check))
                if reason:
                    marks.append(pytest.mark.xfail(strict=True, reason=reason))
                yield pytest.param(
                    engine, lift, check, marks=marks, id=f"{engine}-{lift}-{check}"
                )


@pytest.mark.parametrize(("engine", "lift", "check"), list(_params()))
def test_start_pose_conforms(_results, engine, lift, check):
    result = _conformance(_results, engine, lift).check(check)
    if result.status == "unavailable":
        pytest.skip(f"unavailable: {result.reason}")
    assert result.status == "pass", (
        f"{check} {result.value} vs limit {result.limit} {result.unit}; "
        f"{list(result.detail)[:3]}"
    )
