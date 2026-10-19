"""OpenSim contact grip integration tests with ContactTopology (#13305, HRG-GS-G7).

Acceptance criteria:
- For each style (overlap, interlock, ten_finger): a hold and a short swing replay,
  with the same fixture as the Pinocchio topology test.
- Parity of the per-side total normal force versus the Pinocchio topology reference,
  via grip_contact.parity, within EXISTING PEAK_TOLERANCE (5 %) and RMS_TOLERANCE (2 %).
- Every hand-hand pair is NaN with a reason.
- Existing OpenSim grip tests stay green, and the old pad path is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers.opensim_support import require_opensim

require_opensim()

from src.engines.physics_engines.opensim.python.grip_contact_osim_sim import (
    ContactGripSimulator,
    simulate_grip_contact,
)
from src.engines.physics_engines.pinocchio.python import (
    grip_contact_sim as pin_sim,
)
from src.shared.python.grip_contact import (
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    QuantityError,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.requires_opensim, pytest.mark.timeout(300)]

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
SQUEEZE = 1100.0


def _setup(
    style: GripStyle = GripStyle.OVERLAP, trigger_finger: bool = False
) -> tuple[bytes, dict[str, Any], list[str], Any, GripInterface, TopologyContactModel]:
    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    gspec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = contact_topology(gspec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE)
    return spec_bytes, spec, names, swing, interface, topo_model


def _check_pair_traces(run: Any, topo_model: TopologyContactModel) -> None:
    """Verify physical pairs are finite and hand-hand pairs are NaN with reasons."""
    assert run.pair_normal_force_n is not None
    assert run.pair_tangential_force_n is not None
    assert run.channel_reasons is not None

    for pair in topo_model.topology.hand_grip_pairs():
        fn_trace = run.pair_normal_force_n[pair.name]
        ft_trace = run.pair_tangential_force_n[pair.name]
        assert len(fn_trace) == len(run.series.time_s)
        assert len(ft_trace) == len(run.series.time_s)
        assert np.all(np.isfinite(fn_trace))
        assert np.all(np.isfinite(ft_trace))
        assert np.all(fn_trace >= 0.0)

    for pair in topo_model.topology.hand_hand_pairs():
        fn_trace = run.pair_normal_force_n[pair.name]
        ft_trace = run.pair_tangential_force_n[pair.name]
        assert np.all(np.isnan(fn_trace))
        assert np.all(np.isnan(ft_trace))
        assert pair.name in run.channel_reasons


def _check_parity_versus_pinocchio(
    run_osim: Any, run_pin: Any, label: str
) -> dict[str, QuantityError]:
    """Check per-side total normal force parity versus Pinocchio."""
    errors = {}
    for side in ("L", "R"):
        x_ref = np.asarray(run_pin.squeeze_n[side], dtype=float)
        x_cand = np.asarray(run_osim.squeeze_n[side], dtype=float)
        peak_ref = float(np.max(np.abs(x_ref)))
        peak_cand = float(np.max(np.abs(x_cand)))
        peak_err = abs(peak_cand - peak_ref) / peak_ref
        rms_err = float(np.sqrt(np.mean((x_cand - x_ref) ** 2))) / peak_ref
        qe = QuantityError(
            peak_reference=peak_ref,
            peak_candidate=peak_cand,
            peak_error=peak_err,
            rms_error=rms_err,
        )
        errors[side] = qe
        assert qe.passes(), (
            f"[{label}] side {side} parity failed: "
            f"peak_error={peak_err * 100:.3f}% (max {PEAK_TOLERANCE * 100}%), "
            f"rms_error={rms_err * 100:.3f}% (max {RMS_TOLERANCE * 100}%). "
            f"peak_osim={peak_cand:.3f}N, peak_pin={peak_ref:.3f}N"
        )
    return errors


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_static_hold_fast(style: GripStyle) -> None:
    """Fast check: 0.04 s static hold runs, pair traces valid, parity vs Pinocchio."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.04

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"hold_fast_{style.value}")


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_swing_replay_fast(style: GripStyle) -> None:
    """Fast check: 0.04 s swing replay runs, pair traces valid, parity vs Pinocchio."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.04

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"swing_fast_{style.value}")


@pytest.mark.slow
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_static_hold_slow(style: GripStyle) -> None:
    """Slow check: 0.1 s static hold over all styles."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.1

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"hold_slow_{style.value}")


@pytest.mark.slow
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_swing_replay_slow(style: GripStyle) -> None:
    """Slow check: 0.1 s swing replay over all styles."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.1

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"swing_slow_{style.value}")
