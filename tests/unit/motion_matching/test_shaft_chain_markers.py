"""Shaft markers ride chain link 0, not the tip link (#15150 C23).

``apply_shaft_chain`` moves the ``Clubhead`` frame to the tip link. Every club
marker was attached to that frame, so the shaft markers rode with the head and
a chain bend moved them by the whole tip deflection; the kinematic bend fit
(C13), whose shaft markers sit near the grip, then counted the bend twice.
The shaft markers (the club cluster outside the head triad) are re-attached
to link 0 through the ``SHAFT_MARKER_FRAME`` frame, which carries the
``Clubhead`` placement, so their offsets and their rest positions are kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import CLUB_BODY_SUFFIX, DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    apply_shaft_chain,
    build_shaft_chain,
)
from src.shared.python.motion_matching.execution.club_markers import (
    CLUB_MARKER_BODY,
    HEAD_TRIAD,
    SHAFT_MARKER_FRAME,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)
SHAFT_LABELS = ("Marker_3:3:1", "Marker_3:3:2", "Marker_3:3:3")
BEND_RAD = 0.2
#: Rest positions are compared to round-off; a bend moves a marker by
#: millimetres to decimetres, far above this.
ATOL_M = 1e-9


def _synthetic_document() -> dict[str, Any]:
    """Minimal rigid club with a rotated ``Clubhead`` frame and six markers."""
    from src.shared.python.motion_matching.club_models import club_solids

    prefix = f"Synthetic{CLUB_BODY_SUFFIX}"
    placement = np.eye(4)
    placement[:3, :3] = [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    placement[:3, 3] = [0.0, 0.0, 0.06]
    offsets = {
        HEAD_TRIAD[0]: [0.0, -0.07, -0.01],
        HEAD_TRIAD[1]: [0.05, -0.01, -0.05],
        HEAD_TRIAD[2]: [0.06, 0.0, -0.01],
        **{
            label: [-0.7 - 0.05 * i, -0.07, 0.0] for i, label in enumerate(SHAFT_LABELS)
        },
    }
    return {
        "coordinate_order": ["root_x"],
        "bodies": [
            {"name": "world", "solids": []},
            {"name": prefix, "solids": club_solids(prefix, DRIVER)},
        ],
        "joints": [
            {
                "name": "root",
                "parent": "world",
                "child": prefix,
                "parent_to_base": np.eye(4).tolist(),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [{"primitive": "Px", "coordinate": "root_x"}],
            }
        ],
        "frames": [
            {"name": "Clubhead", "body": prefix, "placement": placement.tolist()},
            {"name": "Grip", "body": prefix, "placement": np.eye(4).tolist()},
        ],
        "marker_attachments": {
            label: {"body": CLUB_MARKER_BODY, "offset_m": value}
            for label, value in offsets.items()
        },
        "club": {"name": DRIVER.name},
    }


def test_shaft_markers_move_to_link_zero_with_their_offsets() -> None:
    rigid = _synthetic_document()
    doc = apply_shaft_chain(rigid, build_shaft_chain(DRIVER))
    shaft = doc["club"]["shaft"]
    frames = {f["name"]: f for f in doc["frames"]}
    link0 = f"{shaft['club_body']}/Shaft Link 0"
    assert frames[SHAFT_MARKER_FRAME]["body"] == link0
    assert frames[SHAFT_MARKER_FRAME]["placement"] == frames["Clubhead"]["placement"]
    assert [f["name"] for f in doc["frames"]][:2] == ["Clubhead", "Grip"]
    attached = doc["marker_attachments"]
    for label, before in rigid["marker_attachments"].items():
        expected = CLUB_MARKER_BODY if label in HEAD_TRIAD else SHAFT_MARKER_FRAME
        assert attached[label] == {"body": expected, "offset_m": before["offset_m"]}
    assert shaft["shaft_marker_link"] == link0


def test_document_without_markers_gets_no_shaft_marker_frame() -> None:
    rigid = _synthetic_document()
    del rigid["marker_attachments"]
    doc = apply_shaft_chain(rigid, build_shaft_chain(DRIVER))
    assert SHAFT_MARKER_FRAME not in {f["name"] for f in doc["frames"]}
    assert "shaft_marker_link" not in doc["club"]["shaft"]


# --- engine world positions (MuJoCo) -------------------------------------------


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
        if a["offset_m"] is not None and label in (*HEAD_TRIAD, *SHAFT_LABELS)
    }
    adapter = NativeMujocoFullBodyModel(json.dumps(spec).encode())
    return FullBodyMarkerKinematics(adapter, attachments)


@pytest.fixture(scope="module")
def engines() -> tuple[Any, Any]:
    pytest.importorskip("mujoco")
    rigid = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return _kinematics(rigid), _kinematics(
        apply_shaft_chain(rigid, build_shaft_chain(DRIVER))
    )


def _split(kin: Any, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = list(kin.labels)
    head = positions[[labels.index(m) for m in HEAD_TRIAD]]
    return head, positions[[labels.index(m) for m in SHAFT_LABELS]]


def _bent(kin: Any, coordinate: str) -> np.ndarray:
    q = np.zeros(kin.nq)
    q[list(kin.coordinate_order).index(coordinate)] = BEND_RAD
    return np.asarray(kin.marker_positions(q))


def test_rest_positions_equal_the_rigid_club(engines: tuple[Any, Any]) -> None:
    rigid, chain = engines
    rest_rigid = rigid.marker_positions(np.zeros(rigid.nq))
    rest_chain = chain.marker_positions(np.zeros(chain.nq))
    order = [list(chain.labels).index(m) for m in rigid.labels]
    np.testing.assert_allclose(rest_chain[order], rest_rigid, rtol=0, atol=ATOL_M)


@pytest.mark.parametrize("joint", [1, 2, 3])
def test_bend_past_link_zero_moves_only_the_head(
    engines: tuple[Any, Any], joint: int
) -> None:
    _, chain = engines
    head0, shaft0 = _split(chain, chain.marker_positions(np.zeros(chain.nq)))
    head, shaft = _split(chain, _bent(chain, f"shaft_bend_x_{joint}"))
    np.testing.assert_allclose(shaft, shaft0, rtol=0, atol=ATOL_M)
    assert np.min(np.linalg.norm(head - head0, axis=1)) > 1e3 * ATOL_M


def test_bend_at_joint_zero_moves_the_shaft_markers(engines: tuple[Any, Any]) -> None:
    _, chain = engines
    _, shaft0 = _split(chain, chain.marker_positions(np.zeros(chain.nq)))
    _, shaft = _split(chain, _bent(chain, "shaft_bend_x_0"))
    assert np.min(np.linalg.norm(shaft - shaft0, axis=1)) > 1e3 * ATOL_M
