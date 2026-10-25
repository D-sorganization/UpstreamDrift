"""Pipeline support for a chain shaft: IK at rest, face frame, replay step (R6, #12686).

The chain coordinates of ``club.shaft.model = "chain"`` are unactuated
springs. Left free in the marker IK they wound to several hundred degrees on
the canonical driver (2026-10-24 run: ``shaft_bend_x_0`` -758 to -499 deg),
because the club markers can be fit by folding the shaft. The kinematic
reference therefore holds every chain coordinate at its undeflected rest
(0 rad); shaft deflection comes only from forward dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    apply_shaft_chain,
    build_shaft_chain,
    passive_coordinates,
)
from src.shared.python.motion_matching.contact_law import GroundPlane

pytest.importorskip("mujoco")

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)
GROUND = GroundPlane((0.0, 0.0, 1.0), 0.0)
BENT_RAD = 0.3


def _kinematics(spec: dict[str, Any]) -> Any:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    attachments = {
        label: (a["body"], tuple(float(x) for x in a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    adapter = NativeMujocoFullBodyModel(json.dumps(spec).encode())
    return FullBodyMarkerKinematics(adapter, attachments)


@pytest.fixture(scope="module")
def chain_spec() -> dict[str, Any]:
    rigid = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return apply_shaft_chain(rigid, build_shaft_chain(DRIVER))


def _bent_problem(kin: Any, spec: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    shaft = [kin.coordinate_order.index(c) for c in passive_coordinates(spec)]
    q = np.zeros(len(kin.coordinate_order))
    q[shaft] = BENT_RAD
    return q, shaft


def test_pose_ik_holds_chain_coordinates_at_rest(chain_spec: dict[str, Any]) -> None:
    kin = _kinematics(chain_spec)
    q_bent, shaft = _bent_problem(kin, chain_spec)
    targets = kin.marker_positions(q_bent)
    valid = np.ones(len(kin.labels), dtype=bool)
    fit = kin.solve_pose(targets, valid, q_bent, ground=GROUND, iterations=5)
    assert np.all(fit.q[shaft] == 0.0)


def test_explicit_lock_of_a_chain_coordinate_wins(chain_spec: dict[str, Any]) -> None:
    kin = _kinematics(chain_spec)
    q_bent, shaft = _bent_problem(kin, chain_spec)
    targets = kin.marker_positions(q_bent)
    valid = np.ones(len(kin.labels), dtype=bool)
    name = kin.coordinate_order[shaft[0]]
    fit = kin.solve_pose(
        targets, valid, q_bent, ground=GROUND, iterations=5, locked={name: 0.1}
    )
    assert fit.q[shaft[0]] == pytest.approx(0.1, abs=0.0)
    assert np.all(fit.q[shaft[1:]] == 0.0)


def test_rigid_spec_has_no_shaft_locks() -> None:
    rigid = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    kin = _kinematics(rigid)
    assert kin.shaft_rest_coordinates() == ()


def test_face_targets_accept_the_chain_tip_link(chain_spec: dict[str, Any]) -> None:
    """The chain moves ``Clubhead`` to the tip link, whose frame is the club
    frame at rest, so the face axis and centre equal the rigid club's."""
    from src.shared.python.motion_matching.club_face_target import (
        face_axis_in_frame,
        face_centre_in_frame,
    )

    rigid = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        face_axis_in_frame(chain_spec), face_axis_in_frame(rigid), atol=1e-12
    )
    np.testing.assert_allclose(
        face_centre_in_frame(chain_spec), face_centre_in_frame(rigid), atol=1e-12
    )


def test_face_targets_still_refuse_a_non_club_frame(chain_spec: dict[str, Any]) -> None:
    from src.shared.python.motion_matching.club_face_target import (
        face_axis_in_frame,
    )

    spec = json.loads(json.dumps(chain_spec))
    frame = next(f for f in spec["frames"] if f["name"] == "Clubhead")
    frame["body"] = spec["bodies"][0]["name"]
    with pytest.raises(ValueError, match="not on the club body"):
        face_axis_in_frame(spec)


def test_chain_replay_uses_the_qualified_substep(chain_spec: dict[str, Any]) -> None:
    """A rigid club keeps the 1 ms replay; a chain replays at 0.125 ms."""
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching.pipeline.constants import (
        CHAIN_DT_S,
        DT_S,
    )
    from src.shared.python.motion_matching.pipeline.dynamics import replay_dt_s
    from src.shared.python.motion_matching.same_input.bundle import (
        FINE_EFFORT_HOLD_DT_S,
    )

    rigid = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert replay_dt_s(NativeMujocoFullBodyModel(json.dumps(rigid).encode())) == DT_S
    chain = NativeMujocoFullBodyModel(json.dumps(chain_spec).encode())
    assert replay_dt_s(chain) == CHAIN_DT_S == FINE_EFFORT_HOLD_DT_S
