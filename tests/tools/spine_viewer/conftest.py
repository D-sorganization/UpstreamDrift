"""Shared synthetic-study fixture for the Spine Viewer tests (SPN-15b, #12772).

Every number is INVENTED. Receipts and ``_loads.npz`` artifacts are built in
``tmp_path`` with the SPN-13a record types.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    FileRecord,
    SpineStudyReceipt,
    StageRecord,
    write_receipt,
)
from src.shared.python.spine.levels import SpinalLevel

LEVELS = (SpinalLevel.L5_S1, SpinalLevel.L4_L5)
TIME = np.array([0.0, 1.0, 2.0, 3.0])
SHA = "0123456789abcdef" * 4
SOURCE = "invented-law-a"


def write_study(
    folder: Path,
    *,
    status: str = "pass",
    scientific_status: str = "unqualified",
    source: str = SOURCE,
) -> Path:
    """Write a synthetic receipt plus loads artifact; return the receipt path."""
    folder.mkdir(parents=True, exist_ok=True)
    force = np.zeros((4, 2, 3))
    force[:, 0, 1] = [0.0, 100.0, 300.0, 200.0]
    force[:, 1, 1] = [0.0, 10.0, 20.0, 10.0]
    force[:, 0, 0] = [0.0, -50.0, 40.0, 0.0]
    force[:, 0, 2] = [0.0, 5.0, 0.0, 0.0]
    npz = folder / "s1_loads.npz"
    np.savez(
        npz,
        time_s=TIME,
        frame_index=np.arange(4),
        levels=np.array([lv.name for lv in LEVELS]),
        id_time_s=TIME,
        id_moment_nm=np.zeros((4, 2, 3)),
        joint_reaction_force_n=force,
        joint_reaction_moment_nm=np.zeros((4, 2, 3)),
    )
    receipt = SpineStudyReceipt(
        study_id="s1",
        status=status,  # type: ignore[arg-type]
        status_reasons=() if status == "pass" else ("invented reason",),
        request={},
        request_sha256=SHA,
        inputs=(),
        code_version="x",
        opensim_version="4.6",
        scipy_version="1",
        numpy_version="1",
        law="law-id",
        distribution_source=source,
        asset=None,
        torso_alignment={
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "provenance": "calibrated:test fixture: identity by construction",
        },
        stages=(StageRecord("drive", "ok"),),
        artifacts=(
            FileRecord(
                "loads_npz", npz.name, hashlib.sha256(npz.read_bytes()).hexdigest()
            ),
        ),
        residual={},
        reserve={},
        context_comparisons=(),
        scientific_status=scientific_status,
    )
    return write_receipt(receipt, folder / "s1_receipt.json")


@pytest.fixture
def study_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return ``write_study`` bound to a fresh sub-folder per call."""
    counter = iter(range(1000))

    def make(**kwargs: object) -> Path:
        return write_study(tmp_path / f"study{next(counter)}", **kwargs)  # type: ignore[arg-type]

    return make
