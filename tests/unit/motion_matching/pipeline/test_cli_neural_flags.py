"""Neural CLI flags on the ground-support pipeline runner (#12970)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import cli
from src.shared.python.motion_matching.pipeline.receipt import (
    GroundSupportReceiptInputs,
    build_ground_support_receipt,
)
from src.tools.motion_matching import pipeline as mm_pipeline
from src.tools.motion_matching.controller import RunDisposition
from src.tools.motion_matching.neural_cli import (
    NeuralRunRejectedError,
    build_neural_receipt,
    enforce_neural_gate,
    match_request_from_ground_support_args,
    prepare_ground_support_neural,
)

pytestmark = pytest.mark.unit


def _parser_args(argv: list[str]) -> Namespace:
    return cli.build_parser().parse_args(argv)


def test_parser_accepts_neural_flags(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    ckpt.write_text("x", encoding="utf-8")
    args = _parser_args(
        [
            "--neural-mode",
            "verified",
            "--neural-model",
            "driven_double_pendulum",
            "--no-neural-fallback",
            "--neural-checkpoint",
            str(ckpt),
        ]
    )
    assert args.neural_mode == "verified"
    assert args.neural_model == "driven_double_pendulum"
    assert args.no_neural_fallback is True
    assert args.neural_checkpoint == ckpt


def test_match_request_maps_cli_flags(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"club": "driver"}', encoding="utf-8")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("w", encoding="utf-8")
    args = _parser_args(
        [
            "--spec",
            str(spec),
            "--neural-mode",
            "verified",
            "--neural-model",
            "driven_double_pendulum",
            "--no-neural-fallback",
            "--neural-checkpoint",
            str(ckpt),
        ]
    )
    req = match_request_from_ground_support_args(args)
    assert req.neural_mode == "verified"
    assert req.neural_model == "driven_double_pendulum"
    assert req.allow_fallback is False
    assert req.checkpoint_path == ckpt


def test_prepare_ground_support_neural_uses_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[mm_pipeline.MatchRequest] = []

    class _StubController:
        def prepare_run(
            self, request: mm_pipeline.MatchRequest, **_: object
        ) -> RunDisposition:
            seen.append(request)
            return RunDisposition(
                status="classical_executed",
                mode="classical",
                model_id=request.neural_model,
                allow_fallback=request.allow_fallback,
                is_preview=False,
                reason="",
            )

    monkeypatch.setattr(
        "src.tools.motion_matching.neural_cli.MotionMatchingController",
        _StubController,
    )
    args = _parser_args(
        [
            "--neural-mode",
            "preview",
            "--neural-model",
            "driven_double_pendulum",
        ]
    )
    req, disp = prepare_ground_support_neural(args)
    assert len(seen) == 1
    assert seen[0].neural_mode == "preview"
    assert disp.status == "classical_executed"
    assert req.neural_model == "driven_double_pendulum"


def test_unsupported_model_without_fallback_fails_closed() -> None:
    args = _parser_args(
        [
            "--neural-mode",
            "preview",
            "--neural-model",
            "reconstruction_triple_pendulum",
            "--no-neural-fallback",
        ]
    )
    with pytest.raises(NeuralRunRejectedError):
        cli._neural_receipt_from_args(args)


def test_build_neural_receipt_records_requested_and_executed() -> None:
    req = mm_pipeline.MatchRequest(
        capture="driver",
        club="driver",
        neural_mode="preview",
        neural_model="driven_double_pendulum",
        allow_fallback=True,
    )
    disp = RunDisposition(
        status="classical_fallback",
        mode="preview",
        model_id="driven_double_pendulum",
        allow_fallback=True,
        is_preview=True,
        reason="Missing checkpoint / proposal function",
        fallback_reason="Missing checkpoint / proposal function",
    )
    block = build_neural_receipt(req, disp)
    assert block["requested_mode"] == "preview"
    assert block["executed_mode"] == "classical_fallback"
    assert block["model_id"] == "driven_double_pendulum"
    assert block["allow_fallback"] is True
    assert block["disposition_status"] == "classical_fallback"


def test_enforce_neural_gate_raises_on_rejected() -> None:
    disp = RunDisposition(
        status="rejected",
        mode="verified",
        model_id="driven_double_pendulum",
        allow_fallback=False,
        is_preview=False,
        reason="no checkpoint",
    )
    with pytest.raises(NeuralRunRejectedError, match="no checkpoint"):
        enforce_neural_gate(disp)


def test_build_ground_support_receipt_includes_neural_block(tmp_path: Path) -> None:
    c3d_file = tmp_path / "dummy.c3d"
    c3d_file.write_bytes(b"dummy c3d content")
    base_spec = {"name": "test_spec", "club": "driver"}
    lane = MagicMock()
    lane.bounds = {}
    lane.labels = ("m1",)
    lane.ground_cal.height_m = 0.0
    lane.ground_cal.lowest_marker_height_m = 0.03
    lane.ground_cal.standoff_m = 0.03
    lane.ground_cal.policy = "lowest_toe"
    lane.stance = [("heel_r", "forefoot_r")]
    kin = MagicMock()

    neural = {
        "requested_mode": "verified",
        "executed_mode": "rejected",
        "model_id": "driven_double_pendulum",
        "allow_fallback": False,
        "disposition_status": "rejected",
    }
    with patch(
        "src.shared.python.motion_matching.pipeline.receipt.posture_summary",
        return_value={"spine_bend_deg": {}},
    ):
        receipt = build_ground_support_receipt(
            GroundSupportReceiptInputs(
                base_spec=base_spec,
                spec_path=Path("spec.json"),
                scaled_path=Path("scaled.json"),
                hipcal_path=Path("hipcal.json"),
                recalibrate_upper=False,
                anthropometric=None,
                qualification_note="test note",
                spec_bytes=b"{}",
                hip_report={"centre_r": [0, 0, 0]},
                candidate_bytes=b"{}",
                c3d_path=c3d_file,
                capture_name="driver",
                lane=lane,
                address_report={"rms": 0.005},
                ik_report={"rms": 0.027},
                dynamics_report={"rms": 0.074},
                kin=kin,
                q_ref=np.zeros((10, 3)),
                elapsed_s=1.0,
                validate=False,
                neural=neural,
            )
        )
    assert receipt["neural"] == neural
