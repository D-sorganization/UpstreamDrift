"""Pinocchio full-body coordinate-order contract (#12042 follow-up).

``FullBodyPinocchioModel`` adds one scalar Pinocchio joint per spec primitive
in depth-first tree order, so its ``q`` layout is *not* the spec
``coordinate_order`` for the branched full-body tree (the spine, torso, arm
and neck coordinates are permuted; there is no free-flyer, so ``nq == nv``).
The name-keyed model APIs map through ``_coordinates``; the array-valued IK
adapter and the matching plant must permute the same way.

Before the fix ``PinocchioFullBodyIK.pose_fn`` passed spec-ordered vectors
straight to ``forwardKinematics``: a pure ``TorsoInput`` twist landed on the
``SpineInputY`` slot (the same defect #12077 fixed for Drake). The adapter
also rejected the 44-coordinate anthropometric specs that Drake accepts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.identifiability import (
    compute_spec_marker_positions,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_drake/full_body_spec_hipcal_scaled.json"
)
V2_SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"
TWIST_RAD = float(np.radians(-60.0))


def _require_real_pinocchio() -> None:
    try:
        import pinocchio
    except ImportError as exc:
        pytest.skip(f"pinocchio not importable: {exc}")
    if type(pinocchio).__module__ == "unittest.mock" or not hasattr(
        pinocchio, "JointModelRZ"
    ):
        pytest.skip("real pinocchio runtime required (found mock/stub)")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def attachments(spec: dict[str, Any]) -> dict[str, tuple[str, tuple[float, ...]]]:
    """Body markers only; the club markers are excluded because the shared spec
    FK places the club frames differently from the engines (a separate
    follow-up, not a coordinate-order effect)."""
    club = spec["closure"]["body_b"]
    club_frames = {f["name"] for f in spec["frames"] if f["body"] == club} | {club}
    return {
        label: (att["body"], tuple(float(v) for v in att["offset_m"]))
        for label, att in spec["marker_attachments"].items()
        if att.get("offset_m") is not None and att["body"] not in club_frames
    }


@pytest.fixture(scope="module")
def plant(spec: dict[str, Any]) -> Any:
    _require_real_pinocchio()
    from src.shared.python.motion_matching.pipeline.plants.pinocchio_plant import (
        PinocchioMatchingPlant,
    )

    return PinocchioMatchingPlant(spec)


def _named(spec: dict[str, Any], **values: float) -> np.ndarray:
    names = spec["coordinate_order"]
    q = np.zeros(len(names))
    for name, value in values.items():
        q[names.index(name)] = value
    return q


def _random_q(spec: dict[str, Any], seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).uniform(-0.4, 0.4, len(spec["coordinate_order"]))
    q[:3] = (0.0, 0.0, 1.0)
    return q


def _shared_fk(
    spec: dict[str, Any], attachments: dict[str, Any], q: np.ndarray
) -> np.ndarray:
    positions = compute_spec_marker_positions(
        spec,
        q,
        marker_offsets=attachments,
        coordinate_names=spec["coordinate_order"],
    )
    return np.array([positions[label] for label in attachments], dtype=float)


def _assert_pose_fn_matches_name_keyed_fk(
    ik: Any, names: list[str], q: np.ndarray
) -> None:
    """Compare ``pose_fn`` with FK driven through the model's name map.

    ``FullBodyPinocchioModel.configuration`` places each named coordinate in
    its Pinocchio ``idx_q`` slot, so it is an independent oracle for the order.
    """
    model = ik.model
    poses = ik.pose_fn(q)
    q_pin = model.configuration(dict(zip(names, q, strict=True)))
    pin = model._pin
    data = model.model.createData()
    pin.forwardKinematics(model.model, data, q_pin)
    pin.updateFramePlacements(model.model, data)
    assert len(poses) >= 5
    for body, (rotation, translation) in poses.items():
        if body in model._frames:
            expected = data.oMf[model._frames[body]]
        else:
            joint, placement = model.body_placement(body)
            expected = data.oMi[joint] * placement
        np.testing.assert_allclose(rotation, expected.rotation, atol=1e-9)
        np.testing.assert_allclose(translation, expected.translation, atol=1e-9)


def test_pinocchio_q_order_differs_from_spec_order(
    plant: Any, spec: dict[str, Any]
) -> None:
    """Precondition for the other tests: the permutation is real."""
    model = plant.model
    order = [model._coordinates[name] for name in spec["coordinate_order"]]
    assert model.model.nq == model.model.nv == len(order)
    assert sorted(order) == list(range(len(order)))
    assert order != list(range(len(order)))


def test_torso_twist_matches_shared_spec_fk(
    plant: Any, spec: dict[str, Any], attachments: dict[str, Any]
) -> None:
    """Axis parity: a pure TorsoInput twist moves the markers as the spec says."""
    assert len(attachments) >= 20
    q = _named(spec, TranslationInputZ=1.0, TorsoInput=TWIST_RAD)
    np.testing.assert_allclose(
        plant.marker_positions(q, attachments),
        _shared_fk(spec, attachments, q),
        atol=1e-9,
    )


def test_array_pose_fn_matches_name_keyed_fk_and_shared_fk(
    plant: Any, spec: dict[str, Any], attachments: dict[str, Any]
) -> None:
    """The array IK path agrees with the model's name-keyed FK and shared FK."""
    names = spec["coordinate_order"]
    ik = plant.create_ik(attachments)
    for seed in range(3):
        q = _random_q(spec, seed)
        _assert_pose_fn_matches_name_keyed_fk(ik, names, q)
        np.testing.assert_allclose(
            plant.marker_positions(q, attachments),
            _shared_fk(spec, attachments, q),
            atol=1e-9,
        )


def test_v2_spec_pose_fn_matches_name_keyed_fk() -> None:
    """The 41-coordinate v2 inventory is permuted too and is mapped the same way.

    v2 marker offsets do not follow the shared-FK body convention, so this
    check compares against Pinocchio's own name-keyed FK only.
    """
    _require_real_pinocchio()
    from src.engines.physics_engines.pinocchio.python.full_body_ik import (
        PinocchioFullBodyIK,
    )

    v2 = json.loads(V2_SPEC_PATH.read_text(encoding="utf-8"))
    ik = PinocchioFullBodyIK(v2)
    q = _named(v2, TranslationInputZ=1.0, TorsoInput=TWIST_RAD)
    _assert_pose_fn_matches_name_keyed_fk(ik, v2["coordinate_order"], q)
