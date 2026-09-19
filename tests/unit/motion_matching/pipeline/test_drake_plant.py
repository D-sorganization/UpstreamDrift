"""Unit tests for Drake MatchingPlant implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.plant import (
    MatchingPlant,
    get_plant,
)

pytestmark = pytest.mark.unit


def _require_real_drake() -> None:
    try:
        import pydrake.all as drake_all
    except ImportError as exc:
        pytest.skip(f"pydrake not importable: {exc}")
    if type(drake_all).__module__ == "unittest.mock" or not hasattr(
        drake_all, "MultibodyPlant"
    ):
        pytest.skip("pydrake is mocked, not a real Drake installation")


def _load_spec() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    spec_path = (
        root / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    )
    return json.loads(spec_path.read_text(encoding="utf-8"))


def test_drake_plant_satisfies_matching_plant_protocol() -> None:
    _require_real_drake()
    spec = _load_spec()
    plant = get_plant("drake", spec)
    assert isinstance(plant, MatchingPlant)
    assert plant.engine_name == "drake"
    assert len(plant.coordinate_order) in (41, 44)
    assert plant.ground_plane is not None
    assert plant.plant_sha is not None


def test_drake_plant_frame_poses_and_markers() -> None:
    _require_real_drake()
    spec = _load_spec()
    plant = get_plant("drake", spec)
    q0 = np.zeros(len(plant.coordinate_order), dtype=float)
    attachments = {
        "m1": ("Torso", (0.0, 0.0, 0.1)),
        "m2": ("calcn_r", (0.05, 0.0, -0.02)),
    }
    poses = plant.frame_poses(attachments, q0)
    assert "Torso" in poses
    assert "calcn_r" in poses

    pos = plant.marker_positions(q0, attachments)
    assert pos.shape == (2, 3)
    assert np.all(np.isfinite(pos))


def test_drake_plant_step_and_accelerations() -> None:
    _require_real_drake()
    spec = _load_spec()
    plant = get_plant("drake", spec)
    coords = plant.coordinate_order
    n = len(coords)
    q = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    tau = np.zeros(n, dtype=float)

    coords_dict = dict(zip(coords, q, strict=True))
    rates_dict = dict(zip(coords, v, strict=True))
    efforts_dict = dict(zip(coords, tau, strict=True))

    acc = plant.accelerations(coords_dict, rates_dict, efforts_dict)
    assert len(acc) == n
    assert all(np.isfinite(val) for val in acc.values())

    next_q, next_v = plant.step(q, v, tau, dt=0.001)
    assert next_q.shape == (n,)
    assert next_v.shape == (n,)
    assert np.all(np.isfinite(next_q))
    assert np.all(np.isfinite(next_v))


def test_drake_ik_frame0_matches_setup_parity() -> None:
    _require_real_drake()
    root = Path(__file__).resolve().parents[4]
    parity_receipt_path = (
        root
        / "docs/development/full_body_models/evidence/setup_parity/receipt_anthro_driver.json"
    )
    if not parity_receipt_path.exists():
        pytest.skip("setup parity receipt not found")
    data = json.loads(parity_receipt_path.read_text(encoding="utf-8"))
    drake_result = data["engines"]["drake"]
    assert drake_result["passed"] is True
    assert drake_result["max_position_error_m"] <= 1e-5


def test_drake_matching_plant_alias_module() -> None:
    from src.engines.physics_engines.drake.python.matching_plant import (
        DrakeMatchingPlant as DrakeAlias,
    )
    from src.shared.python.motion_matching.pipeline.plants.drake_plant import (
        DrakeMatchingPlant,
    )

    assert DrakeAlias is DrakeMatchingPlant
