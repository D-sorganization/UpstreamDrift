"""Cross-engine physical parity for force/torque overlays (FTO-21, #11306).

Every engine's provider must show the same physics with the same sign conventions
for three ``synthetic_`` statics scenarios (see ``force_overlay_fixtures``). Rows
assert on the provider frame only. An engine row skips only when the engine is
absent (``importorskip``) or its provider has not merged yet (named issue).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.force_overlay import ForceTorqueFrame, WrenchKind

from .force_overlay_engine_builders import (
    PENDING_PROVIDERS,
    merged_pending_providers,
    opensim_hanging,
    opensim_inverted,
    opensim_resting_ball,
    simscape_hanging,
    simscape_inverted,
)
from .force_overlay_fixtures import (
    ANALYTIC_ATOL_N,
    BALL_MASS_KG,
    CONTACT_ATOL_N,
    CONTACT_POINT_ATOL_M,
    CONTACT_RTOL,
    GRAVITY_M_S2,
    PENDULUM,
    assert_wrench,
    hanging_expected,
    inverted_expected,
)

pytestmark = [pytest.mark.integration, pytest.mark.cross_engine]

Builder = Callable[[Path], ForceTorqueFrame]

#: scenario -> engine -> builder(tmp_path). Engines absent here are pending.
HANGING: dict[str, Builder] = {
    "opensim": lambda d: opensim_hanging(),
    "simscape": simscape_hanging,
}
INVERTED: dict[str, Builder] = {
    "opensim": lambda d: opensim_inverted(),
    "simscape": simscape_inverted,
}
RESTING: dict[str, Builder] = {"opensim": opensim_resting_ball}
#: Simscape is file based and logs no contact for this configuration.
NO_CONTACT = {"simscape": "file-based dataset has no contact channel for a free body"}
#: Simscape logs no axial loads (FTO-18 carries wrenches only).
NO_AXIAL = {"simscape"}
ENGINES = ["mujoco", "drake", "pinocchio", "opensim", "simscape"]


def _builder(table: dict[str, Builder], engine: str) -> Builder:
    if engine in PENDING_PROVIDERS:
        pytest.skip(f"{engine} provider not merged yet ({PENDING_PROVIDERS[engine]})")
    if engine in NO_CONTACT and table is RESTING:
        pytest.skip(NO_CONTACT[engine])
    return table[engine]


@pytest.mark.parametrize("engine", ENGINES)
def test_hanging_pendulum_reaction_is_weight_up_at_the_pivot(
    engine: str, tmp_path: Path
) -> None:
    frame = _builder(HANGING, engine)(tmp_path)
    want = hanging_expected()
    assert_wrench(
        frame,
        WrenchKind.JOINT_REACTION,
        force=want["force_n"],
        point=want["point_m"],
    )


@pytest.mark.parametrize("engine", [e for e in ENGINES if e not in NO_AXIAL])
def test_hanging_pendulum_axial_load_is_tension(engine: str, tmp_path: Path) -> None:
    frame = _builder(HANGING, engine)(tmp_path)
    assert frame.axial_loads is not None
    (value,) = frame.axial_loads.values_n.values()
    assert value == pytest.approx(hanging_expected()["axial_n"], abs=ANALYTIC_ATOL_N)


@pytest.mark.parametrize("engine", ENGINES)
def test_inverted_pendulum_actuator_balances_gravity(
    engine: str, tmp_path: Path
) -> None:
    frame = _builder(INVERTED, engine)(tmp_path)
    want = inverted_expected()
    assert_wrench(
        frame,
        WrenchKind.JOINT_ACTUATOR,
        torque=want["actuator_torque_nm"],
        point=want["point_m"],
    )
    (wrench,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert wrench.force_n is None, "an actuator has no force half; never zero-fill"
    assert np.linalg.norm(wrench.torque_nm) == pytest.approx(
        want["actuator_torque_magnitude_nm"], abs=ANALYTIC_ATOL_N
    )


@pytest.mark.parametrize("engine", [e for e in ENGINES if e not in NO_AXIAL])
def test_inverted_pendulum_axial_load_is_compression(
    engine: str, tmp_path: Path
) -> None:
    frame = _builder(INVERTED, engine)(tmp_path)
    assert frame.axial_loads is not None
    (value,) = frame.axial_loads.values_n.values()
    assert value < 0.0, "standing link is in compression (tension positive)"
    assert value == pytest.approx(inverted_expected()["axial_n"], abs=ANALYTIC_ATOL_N)


@pytest.mark.parametrize("engine", ENGINES)
def test_resting_body_contact_sums_to_weight_on_the_ground_plane(
    engine: str, tmp_path: Path
) -> None:
    frame = _builder(RESTING, engine)(tmp_path)
    contacts = frame.by_kind(WrenchKind.CONTACT)
    assert contacts, "no contact wrench emitted"
    total = np.sum([w.force_n for w in contacts], axis=0)
    np.testing.assert_allclose(
        total,
        (0.0, 0.0, BALL_MASS_KG * GRAVITY_M_S2),
        rtol=CONTACT_RTOL,
        atol=CONTACT_ATOL_N,
    )
    for w in contacts:
        assert abs(w.point_m[2]) < CONTACT_POINT_ATOL_M, w.label


def test_sign_convention_guard_flips_every_available_engine(tmp_path: Path) -> None:
    """Flipping the expected reaction sign must fail: proves the rows can bite."""
    available = [e for e in HANGING if e not in PENDING_PROVIDERS]
    assert available, "no engine row can run"
    flipped = tuple(-c for c in hanging_expected()["force_n"])
    ran = 0
    for engine in available:
        try:
            frame = HANGING[engine](tmp_path)
        except pytest.skip.Exception:
            continue
        ran += 1
        with pytest.raises(AssertionError):
            assert_wrench(frame, WrenchKind.JOINT_REACTION, force=flipped)
    if not ran:
        pytest.skip("no engine available on this host")


def test_no_provider_has_merged_without_its_parity_rows() -> None:
    """Fails once a pending engine's provider file exists, so its rows cannot be forgotten."""
    overdue = merged_pending_providers()
    assert not overdue, (
        f"provider merged for {overdue}: add its builders and remove it from "
        "PENDING_PROVIDERS (FTO-21 #11306)"
    )


def test_pending_engines_are_exactly_the_unbuilt_ones() -> None:
    """Rows and pending list stay in sync: a merged provider must gain a builder."""
    assert set(ENGINES) == set(HANGING) | set(PENDING_PROVIDERS)
    assert set(INVERTED) == set(HANGING)
    assert PENDULUM.weight_n == pytest.approx(PENDULUM.mass_kg * GRAVITY_M_S2)
