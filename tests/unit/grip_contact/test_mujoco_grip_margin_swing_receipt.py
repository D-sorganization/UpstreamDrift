"""MuJoCo scripted swing grip-margin receipt (#13885 PR 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.grip_contact.grip_margin_swing import (
    RECEIPT_DURATION_S,
    RECEIPT_LEAD_GRIP_FORCE_LIMIT_N,
    RECEIPT_TRAIL_GRIP_FORCE_LIMIT_N,
    SwingReceiptConfig,
    load_mu_bias_bound,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/grip_kinetics"
CALIBRATION = EVIDENCE / "mujoco_mu_crit_calibration.json"


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"mujoco not installed: {exc}")


@pytest.mark.timeout(120)
def test_scripted_swing_receipt_is_deterministic_under_force_limits() -> None:
    _require_mujoco()
    from scripts.grip.mujoco_grip_margin_swing_receipt import run_receipt

    cfg = SwingReceiptConfig(duration_s=RECEIPT_DURATION_S)
    first = run_receipt(cfg)
    second = run_receipt(cfg)
    assert first["min_margin"] == pytest.approx(second["min_margin"], rel=0, abs=0)
    assert first["slip_event_count"] == second["slip_event_count"]
    assert first["lead_grip_force_limit_n"] == RECEIPT_LEAD_GRIP_FORCE_LIMIT_N
    assert first["trail_grip_force_limit_n"] == RECEIPT_TRAIL_GRIP_FORCE_LIMIT_N
    assert first["duration_s"] == RECEIPT_DURATION_S
    assert json.loads(json.dumps(first)) == json.loads(json.dumps(second))
    report = first["report"]
    assert report["min_margin"] == pytest.approx(first["min_margin"])
    assert 0.0 <= report["min_margin"] <= 1.0 + 1e-6


def test_load_mu_bias_bound_reads_calibration() -> None:
    bound = load_mu_bias_bound(CALIBRATION)
    assert bound is None


def test_cli_writes_receipt(tmp_path: Path) -> None:
    _require_mujoco()
    from scripts.grip import mujoco_grip_margin_swing_receipt as cli

    out = tmp_path / "receipt.json"
    assert cli.main(["--out", str(out), "--duration-s", str(RECEIPT_DURATION_S)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["issue"] == 13885
    assert "min_margin" in payload and "slip_event_count" in payload
