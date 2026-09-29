"""Simscape ``starting_pose_offsets`` JSON format conformance.

The output produced by :func:`save_initial_state` for ``engine ==
"simscape"`` must match the exact key set that ``solve_starting_pose.m``
expects (top-level ``Tx, Ty, Tz, Rx, Ry, Rz, Scale, jointAngles``; nested
``jointAngles`` keyed by Simulink.Parameter names from
``REFERENCE_GOLFER_FIELDS``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.canonical import (
    canonical_from_reference_setup,
)
from src.shared.python.pose_interchange.pose_io import save_initial_state

pytestmark = pytest.mark.unit

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "pose_io"
    / "simscape_offsets_golden.json"
)

_TOPLEVEL_KEYS = {"Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Scale", "jointAngles"}


def test_simscape_topkey_set(tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    out = tmp_path / "offsets.json"
    save_initial_state(pose, "simscape", out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload.keys()) == _TOPLEVEL_KEYS


def test_simscape_jointangles_key_set(tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    out = tmp_path / "offsets.json"
    save_initial_state(pose, "simscape", out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["jointAngles"].keys()) == set(REFERENCE_GOLFER_FIELDS)


def test_simscape_matches_golden_fixture(tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    out = tmp_path / "offsets.json"
    save_initial_state(pose, "simscape", out)
    produced = json.loads(out.read_text(encoding="utf-8"))
    golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert produced == golden
