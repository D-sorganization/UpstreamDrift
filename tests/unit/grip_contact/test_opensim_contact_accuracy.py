"""Integrator accuracy is chosen by motion (#13560, D61/D63).

A swing's net-moment residual is the integrator's accuracy, so swings integrate at
5e-10; a hold already passes the gate at 5e-8 and costs 450-680 s at 5e-10, so it
keeps 5e-8. OpenSim is imported lazily by the module, so this runs without it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import grip_contact_osim_sim as sim

pytestmark = pytest.mark.unit


def test_swing_integrates_tighter_than_hold() -> None:
    assert sim.topology_contact_accuracy(hold=False) == 5e-10
    assert sim.topology_contact_accuracy(hold=True) == 5e-8
    assert sim.topology_contact_accuracy(hold=False) < sim.topology_contact_accuracy(
        hold=True
    )


def test_hold_accuracy_is_the_documented_default() -> None:
    assert sim.topology_contact_accuracy(hold=True) == sim.TOPOLOGY_CONTACT_ACCURACY


@pytest.mark.parametrize("hold", [False, True])
def test_simulate_grip_contact_selects_accuracy_by_motion(
    monkeypatch: pytest.MonkeyPatch, hold: bool
) -> None:
    seen: dict[str, object] = {}

    class FakeSimulator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, t_end: float, accuracy: float | None = None) -> object:
            seen["accuracy"] = accuracy
            return SimpleNamespace(to_contact_run=lambda: "run")

    monkeypatch.setattr(sim, "ContactGripSimulator", FakeSimulator)
    swing = SimpleNamespace(
        time_s=np.linspace(0.0, 0.02, 3), q=np.zeros((3, 2)), names=["a", "b"]
    )
    out = sim.simulate_grip_contact(
        b"",
        swing,
        topology_model=object(),
        hold=hold,  # type: ignore[arg-type]
    )
    assert out == "run"
    assert seen["accuracy"] == sim.topology_contact_accuracy(hold=hold)
