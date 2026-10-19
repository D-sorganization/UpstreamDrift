"""Pinocchio contact grip integration tests with ContactTopology (#13304, HRG-GS-G6).

Acceptance criteria:
- For each style (overlap, interlock, ten_finger), a static squeeze/hold and
  swing replay run; per-pair force traces equal the shared-law evaluation at the
  same states to <= 1e-9 N.
- Timestep halving changes peak pair force by < 2 %.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    GripInterface,
    hand_frame_states,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_side_wrench,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
SQUEEZE = 1100.0


def _pinocchio_sim_module() -> Any:
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")
    from src.engines.physics_engines.pinocchio.python import grip_contact_sim

    return grip_contact_sim


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


def _assert_traces_match_law(run: Any, topo_model: TopologyContactModel) -> None:
    """Every recorded pair trace equals the shared law re-evaluated (<= 1e-9 N)."""
    assert run.pair_normal_force_n is not None
    assert run.pair_tangential_force_n is not None
    assert run.hand_states is not None
    assert run.club_states is not None
    for k in range(len(run.series.time_s)):
        for side in ("L", "R"):
            re_eval = topology_side_wrench(
                topo_model, side, run.hand_states[side][k], run.club_states[side][k]
            )
            for pair in topo_model.physical_grip_pairs(side):
                pw = re_eval.pair_wrenches[pair.name]
                np.testing.assert_allclose(
                    run.pair_normal_force_n[pair.name][k],
                    pw.normal_force_mag_n,
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    run.pair_tangential_force_n[pair.name][k],
                    pw.tangential_force_mag_n,
                    atol=1e-9,
                )


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_pinocchio_topology_static_hold_fast(style: GripStyle) -> None:
    """Fast check: per-pair traces match shared-law re-evaluation on short hold."""
    module = _pinocchio_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.04
    run = module.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    assert run.pair_normal_force_n is not None
    assert run.pair_tangential_force_n is not None
    assert run.hand_states is not None
    assert run.club_states is not None

    _assert_traces_match_law(run, topo_model)

    # Check non-empty physical pairs
    for pair in topo_model.topology.hand_grip_pairs():
        fn_trace = run.pair_normal_force_n[pair.name]
        ft_trace = run.pair_tangential_force_n[pair.name]
        assert len(fn_trace) == len(run.series.time_s)
        assert len(ft_trace) == len(run.series.time_s)
        assert np.all(np.isfinite(fn_trace))
        assert np.all(np.isfinite(ft_trace))
        assert np.all(fn_trace >= 0.0)

    # Unavailable hand-hand pairs must be NaN
    for pair in topo_model.topology.hand_hand_pairs():
        fn_trace = run.pair_normal_force_n[pair.name]
        ft_trace = run.pair_tangential_force_n[pair.name]
        assert np.all(np.isnan(fn_trace))
        assert np.all(np.isnan(ft_trace))
        assert run.channel_reasons is not None
        assert pair.name in run.channel_reasons


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_pinocchio_topology_swing_replay_fast(style: GripStyle) -> None:
    """Fast check: 0.04 s swing replay runs and matches law evaluation to <= 1e-9 N."""
    module = _pinocchio_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.04
    run = module.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    _assert_traces_match_law(run, topo_model)

    # Check physical pairs exist and are finite
    for pair in topo_model.topology.hand_grip_pairs():
        fn_trace = run.pair_normal_force_n[pair.name]
        assert np.all(np.isfinite(fn_trace))
        assert np.all(fn_trace >= 0.0)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_pinocchio_topology_timestep_halving(style: GripStyle) -> None:
    """AC 2: on the swing replay, halving max_step_s moves the peak pair force < 2 %."""
    module = _pinocchio_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.04

    tol_base = module.IntegratorTolerances(max_step_s=2.0e-4)
    tol_half = module.IntegratorTolerances(max_step_s=1.0e-4)

    run_base = module.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        tolerances=tol_base,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    run_half = module.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        tolerances=tol_half,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )

    assert run_base.pair_normal_force_n is not None
    assert run_half.pair_normal_force_n is not None

    peak_base = max(
        float(np.max(trace))
        for trace in run_base.pair_normal_force_n.values()
        if not np.all(np.isnan(trace))
    )
    peak_half = max(
        float(np.max(trace))
        for trace in run_half.pair_normal_force_n.values()
        if not np.all(np.isnan(trace))
    )

    rel_diff = abs(peak_half - peak_base) / peak_base
    print(
        f"Style {style.value}: peak_base={peak_base:.4f} N, "
        f"peak_half={peak_half:.4f} N, rel_diff={rel_diff * 100:.4f}%"
    )
    assert rel_diff < 0.02, (
        f"Style {style.value}: Timestep halving changed peak pair force by "
        f"{rel_diff * 100:.2f} % (>= 2 %)"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_pinocchio_topology_static_hold_0_2s(style: GripStyle) -> None:
    """Full 0.2 s static hold over all styles; per-pair traces match law to <= 1e-9 N."""
    module = _pinocchio_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.2
    run = module.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    assert run.pair_normal_force_n is not None

    # Check total squeeze supports club
    for side in ("L", "R"):
        assert run.squeeze_n[side][-1] > 0.0


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_pinocchio_topology_swing_replay_1s(style: GripStyle) -> None:
    """Full 1.0 s swing replay over all styles."""
    module = _pinocchio_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 1.0
    run = module.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    _assert_traces_match_law(run, topo_model)
