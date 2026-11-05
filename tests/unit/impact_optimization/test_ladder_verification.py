"""SHOW-C3c slice 3b (#14559): per-speed verification and report of the ladder.

Acceptance tests on a synthetic plant: the real run is held until A6, but the
harness that verifies each rung at full fidelity and words the verdict is
testable now (supervisor D24, D29).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.full_body_ladder import (
    BaselineBand,
    run_full_body_ladder,
)
from src.shared.python.impact_optimization.full_body_rollout import RolloutWindow
from src.shared.python.impact_optimization.ladder_verification import (
    summarise_ladder,
    verify_at_full_fidelity,
    write_summary,
)
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.speed_continuation import LadderSpec
from src.shared.python.impact_optimization.trajectory_optimiser import OptimiserConfig
from src.shared.python.motion_matching.same_input.integrator import StepPolicy

pytestmark = pytest.mark.unit

ROOT = (
    "TranslationInputX", "TranslationInputY", "TranslationInputZ",
    "HipInputX", "HipInputY", "HipInputZ",
)  # fmt: skip
ORDER = (*ROOT, "J0", "J1", "J2")
DT, STEPS = 0.01, 8
CFG = OptimiserConfig(n_knots=2, max_iterations=3)


@dataclass
class _Impact:
    face_deg: float | None
    path_deg: float | None
    aoa_deg: float
    speed_mph: float
    strike_offset_x_m: float | None = 0.0
    strike_offset_y_m: float | None = 0.0


class _Plant:
    coordinate_order = ORDER

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def acceleration(self, q, v, tau):
        a = self.gain * np.array(tau, dtype=float)
        a[:6] = 0.0
        return a

    def closure_pose_residual(self, q):
        return np.zeros(1)


def _window(
    base: float = 100.0, scale: float = 1.0, gain: float = 1.0
) -> RolloutWindow:
    """``scale`` < 1: the full-fidelity policy reads less speed.

    ``gain`` > 1 moves the joints further for the same speed reading, so the
    replay trajectory differs from the search one.
    """

    def impact_fn(times, q, v):
        return _Impact(0.0, 0.0, 0.0, scale * (base + 40.0 * float(v[-1, 6]) / gain))

    eff = np.zeros((STEPS, len(ORDER)))
    eff[:, 6] = 1.0
    return RolloutWindow(
        _Plant(gain), np.zeros(len(ORDER)), np.zeros(len(ORDER)), eff, DT, impact_fn,
        StepPolicy(substeps=1, project=False),
    )  # fmt: skip


def _env(
    j0_rom: float = 90.0, j0_vel: float = 1.0e6, tau: float = 50.0
) -> PhysiologicalEnvelope:
    return PhysiologicalEnvelope(
        names=("J0", "J1", "J2"),
        tau_max_negative_nm=np.full(3, tau),
        tau_max_positive_nm=np.full(3, tau),
        rom_deg=np.array([(-j0_rom, j0_rom), (-90.0, 90.0), (-90.0, 90.0)]),
        vel_max_deg_s=np.array([j0_vel, 1.0e6, 1.0e6]),
        assumed_torque_slots=(),
        assumed_velocity_slots=(),
    )


def _run(env=None, ladder=LadderSpec(105.0, 110.0, 5.0)):
    return run_full_body_ladder(
        _window(), ("J0", "J1"), env or _env(), band=BaselineBand(),
        trust_radius=5.0, config=CFG, ladder=ladder,
    )  # fmt: skip


def test_per_speed_rows_carry_search_and_full_fidelity_speeds() -> None:
    run = _run()
    checks = verify_at_full_fidelity(run, _window(), _env())
    assert [c.speed_target_mph for c in checks] == [105.0, 110.0]
    out = summarise_ladder(run, checks, {})
    assert [r["target_mph"] for r in out["rungs"]] == [105.0, 110.0]
    for row, chk in zip(out["rungs"], checks, strict=True):
        assert row["search_mph"] == pytest.approx(chk.search_speed_mph)
        assert row["full_mph"] == pytest.approx(chk.full_speed_mph)
        assert "kinematic" in row and "torque_binding" in row


def test_verdict_uses_the_full_fidelity_run_not_the_search() -> None:
    run = _run()
    checks = verify_at_full_fidelity(
        run, _window(scale=0.9), _env()
    )  # full reads 10% less
    out = summarise_ladder(run, checks, {})
    assert all(not r["accepted"] for r in out["rungs"])
    assert out["reached_full_mph"] is None
    claim = out["claim_top_rung"]
    assert "NOT reached" in claim and "replay" in claim
    top, chk = out["rungs"][-1], checks[-1]
    assert f"{chk.full_speed_mph:.1f} mph" in claim
    assert f"{chk.search_speed_mph:.1f} mph" in claim
    assert top["search_feasible"] and not top["full_tolerances_met"]


def test_accepted_rung_needs_agreement_and_no_rom_or_velocity_breach() -> None:
    run = _run()
    out = summarise_ladder(run, verify_at_full_fidelity(run, _window(), _env()), {})
    assert all(r["accepted"] for r in out["rungs"])
    assert out["reached_full_mph"] == 110.0
    assert "reached within the controlled subset" in out["claim_top_rung"]


def test_rom_breach_is_reported_as_probable_miscalibration_never_dropped() -> None:
    run = _run()
    tight = _env(j0_rom=0.5 * _peak_j0_deg(run))  # limit at half the peak
    notes = {"J0": "prior-derived split; probable miscalibration"}
    out = summarise_ladder(run, verify_at_full_fidelity(run, _window(), tight), notes)
    row = out["rungs"][-1]
    assert not row["accepted"]  # D29: ROM is a hard constraint on accepted rungs
    assert row["probable_miscalibration"] == ["J0 ROM"]
    assert len(out["rungs"]) == 2  # reported, not silently dropped
    claim = out["claim_top_rung"]
    assert "J0" in claim and "probable miscalibration" in claim
    assert "within the controlled subset" in claim
    assert out["limit_notes"]["J0"] == notes["J0"]


def _peak_j0_deg(run) -> float:
    return float(np.max(np.abs(np.degrees(run.result.points[-1].trajectory[:, 0]))))


def _peak_j0_rate_deg_s(run) -> float:
    traj = np.degrees(run.result.points[-1].trajectory[:, 0])
    return float(np.max(np.abs(np.diff(traj))) / DT)


def test_replay_rom_breach_rejects_a_rung_the_search_found_clean() -> None:
    run = _run()
    limit = 1.5 * _peak_j0_deg(run)  # the search sits at 0.67 of the limit
    clean = verify_at_full_fidelity(run, _window(), _env(j0_rom=limit))
    assert not any(h.utilisation > 1.0 for c in clean for h in c.kinematic)
    out = summarise_ladder(
        run,
        verify_at_full_fidelity(run, _window(gain=2.0), _env(j0_rom=limit)),
        {},
    )  # replay moves J0 twice as far for the same speed reading: 1.33 of it
    row = out["rungs"][-1]
    assert row["full_tolerances_met"] and not row["accepted"]
    assert row["probable_miscalibration"] == ["J0 ROM"]
    assert out["reached_full_mph"] is None or out["reached_full_mph"] < 110.0
    assert "probable miscalibration" in out["claim_top_rung"]


def test_replay_velocity_breach_rejects_a_rung_the_search_found_clean() -> None:
    run = _run()
    limit = 1.5 * _peak_j0_rate_deg_s(run)  # search at 0.67 of the limit
    out = summarise_ladder(
        run,
        verify_at_full_fidelity(run, _window(gain=2.0), _env(j0_vel=limit)),
        {},
    )  # replay rate doubles: ~1.33 of the limit
    row = out["rungs"][-1]
    assert row["full_tolerances_met"] and not row["accepted"]
    assert row["probable_miscalibration"] == ["J0 velocity"]
    assert "J0 velocity" in out["claim_top_rung"]


def test_infeasible_search_names_the_binding_torque_joint() -> None:
    run = _run(env=_env(tau=1.0))  # baseline already uses the full 1 Nm
    out = summarise_ladder(
        run, verify_at_full_fidelity(run, _window(), _env(tau=1.0)), {}
    )
    top = out["rungs"][-1]
    assert not top["search_feasible"] and not top["accepted"]
    assert "NOT reached" in out["claim_top_rung"]
    assert "binding J0" in out["claim_top_rung"]


def test_summary_names_controlled_and_held_joints() -> None:
    run = _run()
    out = summarise_ladder(run, verify_at_full_fidelity(run, _window(), _env()), {})
    assert out["controlled"] == ["J0", "J1"] and out["held"] == ["J2"]
    assert out["baseline"]["speed_mph"] == pytest.approx(103.2)


def test_harness_still_refuses_a_non_physical_baseline() -> None:
    with pytest.raises(ContractViolationError, match="baseline"):
        run_full_body_ladder(
            _window(base=40.0), ("J0",), _env(), band=BaselineBand(), trust_radius=1.0
        )


def test_summary_is_json_and_write_refuses_npz(tmp_path: Path) -> None:
    run = _run()
    out = summarise_ladder(run, verify_at_full_fidelity(run, _window(), _env()), {})
    path = write_summary(out, tmp_path / "ladder.json")
    assert json.loads(path.read_text())["reached_full_mph"] == 110.0
    with pytest.raises(ContractViolationError):
        write_summary(out, tmp_path / "ladder.npz")


def test_search_breach_with_clean_replay_is_not_blamed_on_torque_binding() -> None:
    peak = _peak_j0_deg(_run())
    run = _run(env=_env(j0_rom=0.5 * peak))  # search breaches J0 ROM: demoted
    out = summarise_ladder(run, verify_at_full_fidelity(run, _window(), _env()), {})
    top = out["rungs"][-1]
    assert not top["search_feasible"] and not top["accepted"]  # stays unaccepted
    assert top["probable_miscalibration"] == []  # the replay is clean
    assert top["search_breaches"] == ["J0 ROM"]
    claim = out["claim_top_rung"]
    assert "NOT reached" in claim and "within the controlled subset" in claim
    assert "search" in claim and "J0 ROM" in claim
    assert "binding" not in claim
