"""The single entry point for the full-body-v2 showpiece model (SHOW-A4, D12).

Consumers: the C3/C5 optimiser and render paths and the showpiece renders build
their MuJoCo model here so the cited neck range of motion
(:mod:`~src.shared.python.motion_matching.neck_rom`) always applies.  The
neck limits are fixed on in this module, not a caller choice: the default
``export_full_body_mjcf`` stays limit-free because necromatcher requires
``limited="false"`` on every joint and hashes that XML, so the showpiece must
opt in explicitly.

Both paths are covered: the compiled XML (``mj_step`` holds the bound) and
``NativeMujocoFullBodyModel`` (explicit forward dynamics that never runs
MuJoCo's limit solver, so it applies a soft limit torque instead).
"""

from __future__ import annotations

import json

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.contracts import require
from src.shared.python.motion_matching.full_body_v2_showpiece import SHOWPIECE_V2

__all__ = ["build_showpiece_model", "showpiece_mjcf"]


def _v2_bytes(spec_bytes: bytes | None) -> bytes:
    data = SHOWPIECE_V2.read_bytes() if spec_bytes is None else spec_bytes
    require(
        json.loads(data).get("schema_version") == "full-body-v2",
        "the showpiece model is built from a full-body-v2 spec",
    )
    return data


def showpiece_mjcf(spec_bytes: bytes | None = None) -> str:
    """MJCF of the showpiece with the cited neck limits compiled in."""
    xml, _ = export_full_body_mjcf(_v2_bytes(spec_bytes), neck_limits=True)
    return xml


def build_showpiece_model(spec_bytes: bytes | None = None) -> NativeMujocoFullBodyModel:
    """Native forward-dynamics showpiece model with the neck limits enforced."""
    return NativeMujocoFullBodyModel(_v2_bytes(spec_bytes), neck_limits=True)
