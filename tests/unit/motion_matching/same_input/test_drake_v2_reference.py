"""#14727 step 2: the Drake IK reference, migrated to v2, keeps the observed face.

Owner ruling (HANDOFF #49, 2026-11-06): D51 uses the ``anthro_driver_drake``
IK ``q_track`` directly as the reference, not a rollout, and reproduces the
observed face (about +12 deg open at impact); a square face is a SHOW-C
optimiser output.  The capture's club triad makes the face observable on
618/654 frames, so the migrated reference must carry it through unchanged,
with the hand closure inside the unchanged 5 mm acceptance gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import AcceptanceGates

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence"
TRUNK = EVIDENCE / "trunk_visuals"
RUN_DIR = EVIDENCE / "ground_support/anthro_driver_drake"
RUN_SPEC = RUN_DIR / "full_body_spec_hipcal_scaled.json"
V1_SOURCE = TRUNK / "full_body_v1_source_spec.json"
V2_SPEC = TRUNK / "full_body_v2_anthro_driver.json"

#: Club-face fixture face at impact (CANONICAL_RUN.md, #12221) and the
#: OSV-10 face tolerance it was qualified against.
FIXTURE_FACE_DEG = 14.96
OSV10_FACE_DEG = 5.0
#: Migration moves the trunk, not the club: the impact reading on v2 must
#: equal the run spec's within these bounds (observed 0.0 / 0.4 / 0.3 deg,
#: 0.8 % speed).
SAME_FACE_DEG = 0.1
SAME_ANGLE_DEG = 1.0
SAME_SPEED_REL = 0.03  # GCV-20


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.same_input.drake_reference import (
        drake_v2_reference,
    )

    return drake_v2_reference(RUN_DIR, V1_SOURCE.read_bytes(), V2_SPEC.read_bytes())


def _impact(spec_bytes: bytes, time_s, q, index: int):
    from src.shared.python.motion_matching.same_input.drake_reference import (
        impact_conditions,
    )

    return impact_conditions(spec_bytes, time_s, q, index)


@pytest.mark.unit
@pytest.mark.requires_mujoco
def test_reference_is_the_ik_track_in_v2_coordinates(reference) -> None:
    v2_order = tuple(json.loads(V2_SPEC.read_bytes())["coordinate_order"])
    with np.load(RUN_DIR / "dynamics_record.npz") as record:
        frames = record["q_track"].shape[0]
    assert reference.coordinate_order == v2_order
    assert reference.q.shape == (frames, len(v2_order))
    assert np.isfinite(reference.q).all()
    receipt = json.loads((RUN_DIR / "receipt.json").read_text(encoding="utf-8"))
    impact_s = receipt["dynamics"]["ball_impact"]["t_start_s"]
    assert abs(reference.time_s[reference.impact_index] - impact_s) <= 0.5 / 360


@pytest.mark.unit
@pytest.mark.requires_mujoco
def test_reference_keeps_the_observed_face_at_impact(reference) -> None:
    with np.load(RUN_DIR / "dynamics_record.npz") as record:
        run_q = record["q_track"]
    i = reference.impact_index
    v2 = _impact(V2_SPEC.read_bytes(), reference.time_s, reference.q, i)
    run = _impact(RUN_SPEC.read_bytes(), reference.time_s, run_q, i)
    assert v2.face_deg > OSV10_FACE_DEG, "face must stay open, not squared"
    assert abs(v2.face_deg - FIXTURE_FACE_DEG) <= OSV10_FACE_DEG
    assert abs(v2.face_deg - run.face_deg) <= SAME_FACE_DEG
    assert abs(v2.path_deg - run.path_deg) <= SAME_ANGLE_DEG
    assert abs(v2.aoa_deg - run.aoa_deg) <= SAME_ANGLE_DEG
    assert abs(v2.speed_mph / run.speed_mph - 1.0) <= SAME_SPEED_REL


@pytest.mark.unit
@pytest.mark.requires_mujoco
def test_reference_hand_closure_meets_the_acceptance_gate(reference) -> None:
    from src.shared.python.motion_matching.same_input.plant import VectorPlant

    plant = VectorPlant("mujoco", V2_SPEC.read_bytes())
    # MuJoCo's pose residual is the grip-site displacement (m), then a rotvec.
    worst = max(
        float(np.linalg.norm(plant.closure_pose_residual(q)[:3])) for q in reference.q
    )
    assert worst <= AcceptanceGates().max_closure_residual_m, f"{worst * 1e3:.3f} mm"
