"""Ground-support CLI wiring for neural-assisted matching (#11147, #12970)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.shared.python.contracts import precondition
from src.tools.motion_matching import pipeline as mm_pipeline
from src.tools.motion_matching.controller import (
    MotionMatchingController,
    RunDisposition,
)

__all__ = [
    "NeuralRunRejectedError",
    "build_neural_receipt",
    "enforce_neural_gate",
    "match_request_from_ground_support_args",
    "prepare_ground_support_neural",
]


class NeuralRunRejectedError(RuntimeError):
    """Neural disposition was rejected and classical fallback is disabled."""


def _club_from_spec(spec_path: Path) -> str:
    try:
        document = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "driver"
    club = document.get("club")
    if club is None:
        subject = document.get("subject")
        if isinstance(subject, dict):
            club = subject.get("club")
    if not isinstance(club, str) or club not in mm_pipeline.CLUBS:
        return "driver"
    return club


@precondition(lambda args: args is not None, "args must not be None")
def match_request_from_ground_support_args(
    args: argparse.Namespace,
) -> mm_pipeline.MatchRequest:
    """Build a :class:`MatchRequest` carrying neural flags from the pipeline CLI."""
    model_id = getattr(args, "neural_model", None) or "driven_double_pendulum"
    return mm_pipeline.MatchRequest(
        capture=str(args.capture),
        club=_club_from_spec(Path(args.spec)),
        neural_mode=str(getattr(args, "neural_mode", "classical")),
        neural_model=str(model_id),
        allow_fallback=not bool(getattr(args, "no_neural_fallback", False)),
        checkpoint_path=getattr(args, "neural_checkpoint", None),
    )


def prepare_ground_support_neural(
    args: argparse.Namespace,
    *,
    controller: MotionMatchingController | None = None,
) -> tuple[mm_pipeline.MatchRequest, RunDisposition]:
    """Validate neural CLI flags through the same controller path as the GUI."""
    request = match_request_from_ground_support_args(args)
    ctrl = controller or MotionMatchingController()
    disposition = ctrl.prepare_run(request)
    return request, disposition


def enforce_neural_gate(disposition: RunDisposition) -> None:
    """Fail closed when neural mode was rejected and fallback is disabled."""
    if disposition.status == "rejected":
        reason = disposition.reason or "neural run rejected with no classical fallback"
        raise NeuralRunRejectedError(reason)


def _executed_mode(disposition: RunDisposition) -> str:
    if disposition.status == "classical_executed":
        return "classical"
    if disposition.status == "classical_fallback":
        return "classical_fallback"
    if disposition.status == "neural_accepted":
        return disposition.mode
    if disposition.status == "rejected":
        return "rejected"
    return disposition.status


def build_neural_receipt(
    request: mm_pipeline.MatchRequest,
    disposition: RunDisposition,
) -> dict[str, Any]:
    """Receipt block recording requested versus executed neural disposition."""
    block: dict[str, Any] = {
        "requested_mode": request.neural_mode,
        "executed_mode": _executed_mode(disposition),
        "model_id": request.neural_model,
        "allow_fallback": request.allow_fallback,
        "disposition_status": disposition.status,
    }
    if request.checkpoint_path is not None:
        block["checkpoint_path"] = str(request.checkpoint_path)
    if disposition.reason:
        block["reason"] = disposition.reason
    if disposition.fallback_reason:
        block["fallback_reason"] = disposition.fallback_reason
    if disposition.neural_report is not None:
        block["inference_report"] = disposition.neural_report
    return block
