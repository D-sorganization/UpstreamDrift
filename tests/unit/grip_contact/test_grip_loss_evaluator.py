"""Offline grip-loss evaluator (#15128, HRG-GB5b)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.grip_loss_config import (
    GripBreakawayThresholds,
    GripLossPhysicsConfig,
)
from src.shared.python.grip_contact.grip_loss_evaluator import (
    PREDICTION_DISCLAIMER,
    GripLossEvaluator,
    serialise_grip_loss_report,
)
from src.shared.python.grip_contact.grip_loss_loads import (
    GripLoadSeries,
    loads_from_contact_run,
    register_grip_load_provider,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/grip_kinetics"
CONTACT = EVIDENCE / "contact"


def _thresholds(
    *,
    slip_enter: float = 0.0,
    slip_exit: float = 0.06,
    release_enter: float = -0.15,
    min_slip_dwell_s: float = 0.004,
    min_release_dwell_s: float = 0.004,
) -> GripBreakawayThresholds:
    return GripBreakawayThresholds(
        slip_enter_margin=slip_enter,
        slip_exit_margin=slip_exit,
        release_enter_margin=release_enter,
        min_slip_dwell_s=min_slip_dwell_s,
        min_release_dwell_s=min_release_dwell_s,
    )


def _physics(**kwargs: object) -> GripLossPhysicsConfig:
    defaults = {
        "friction": 0.9,
        "normal_floor_n": 1.0,
        "thresholds": _thresholds(),
    }
    defaults.update(kwargs)
    return GripLossPhysicsConfig(**defaults)


def _synthetic_loads(
    margins: list[float],
    *,
    dt_s: float = 0.002,
    normal_n: float = 200.0,
) -> GripLoadSeries:
    """Build pad loads whose Coulomb margin tracks ``margins`` (one pad)."""
    n = len(margins)
    time_s = np.arange(n, dtype=float) * dt_s
    mu = 0.9
    fn = np.full((n, 1), normal_n)
    margin_arr = np.asarray(margins, dtype=float)
    tang = fn * mu * (1.0 - margin_arr[:, None])
    return GripLoadSeries(
        club="synthetic",
        source="synthetic",
        patch_labels=("lead:0",),
        time_s=time_s,
        normal_force_n=fn,
        tangential_force_n=tang,
    )


def test_state_machine_hysteresis_no_chatter_at_threshold() -> None:
    cfg = _physics(
        thresholds=_thresholds(
            slip_enter=0.05,
            slip_exit=0.12,
            release_enter=-0.5,
            min_slip_dwell_s=0.01,
        )
    )
    evalr = GripLossEvaluator(cfg)
    margin = [0.2] * 10 + [0.04] * 3 + [0.2] * 6 + [0.04] * 8 + [0.2] * 20
    loads = _synthetic_loads(margin)
    report = evalr.evaluate(loads)
    trail = report.state_trace["lead:0"]
    slip_entries = sum(
        1
        for idx in range(1, len(trail))
        if trail[idx] == "SLIPPING" and trail[idx - 1] != "SLIPPING"
    )
    assert slip_entries == 1
    assert "RELEASED" not in trail


def test_margin_above_exit_never_releases() -> None:
    cfg = _physics()
    report = GripLossEvaluator(cfg).evaluate(_synthetic_loads([0.5] * 40))
    assert report.first_slip_s is None
    assert report.first_release_s is None
    assert all(s == "HELD" for s in report.state_trace["lead:0"])


def test_synthetic_overload_releases_at_expected_step() -> None:
    cfg = _physics(
        thresholds=_thresholds(
            slip_enter=0.05,
            slip_exit=0.12,
            release_enter=-0.05,
            min_slip_dwell_s=0.002,
            min_release_dwell_s=0.002,
        )
    )
    margins = [0.5] * 30 + [0.0] * 20 + [-0.2] * 20
    loads = _synthetic_loads(margins, dt_s=0.002)
    report = GripLossEvaluator(cfg).evaluate(loads)
    assert report.first_slip_s == pytest.approx(30 * 0.002, rel=0, abs=0.003)
    assert report.first_release_s is not None
    assert report.first_release_s >= report.first_slip_s


def test_deterministic_report_on_committed_series() -> None:
    from src.shared.python.grip_contact.contact_run import ContactRun

    cfg = _physics()
    evalr = GripLossEvaluator(cfg)
    digests: dict[str, str] = {}
    for club in ("driver", "iron7"):
        path = CONTACT / f"mujoco_{club}_series.npz"
        loads = loads_from_contact_run(ContactRun.load_npz(path), club=club)
        report = evalr.evaluate(loads)
        payload = serialise_grip_loss_report(report)
        assert payload["disclaimer"] == PREDICTION_DISCLAIMER
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digests[club] = hashlib.sha256(blob.encode()).hexdigest()
    assert digests == {
        "driver": "7c6e7183db55ae64351425dc08bec0c80fa6085528ffc51d49b81cca4c0147e6",
        "iron7": "05b0d1583744a9e99b890fb40a640dc4cd54b703f68a4f822648326ab9d59dad",
    }


def test_swappable_load_provider_registry() -> None:
    seen: list[str] = []

    class _Stub:
        def load(self, club: str, *, evidence_root: Path) -> GripLoadSeries:
            seen.append(club)
            return _synthetic_loads([0.5] * 8)

    register_grip_load_provider("stub", _Stub())
    cfg = _physics()
    report = GripLossEvaluator(cfg, load_provider_name="stub").evaluate_club(
        "driver", evidence_root=EVIDENCE
    )
    assert seen == ["driver"]
    assert report.club == "synthetic"


def test_config_rejects_non_hysteretic_thresholds() -> None:
    with pytest.raises(ValueError, match="slip_exit_margin"):
        _thresholds(slip_enter=0.1, slip_exit=0.05)
