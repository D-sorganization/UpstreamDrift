"""Club roll, ball placement, impact card and flight read the bent head (#15150, C4).

With a chain shaft the face is on the last link, so every consumer of the
club head pose must follow the bent tip link, not the rigid club root. Each
test bends the shaft until the tip sits centimetres off the rigid prediction
(the rigid spec at the same joint angles) and asserts that the consumer reads
the tip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.shared.python.model_appearance import ball_strike  # noqa: E402
from src.shared.python.model_appearance.face_roll_calibration import (  # noqa: E402
    DESIGN_LOFT_AT_ADDRESS,
)
from src.shared.python.motion_matching.club_models import DRIVER  # noqa: E402
from src.shared.python.motion_matching.club_shaft_chain import (  # noqa: E402
    apply_shaft_chain,
    build_shaft_chain,
    passive_coordinates,
)
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402
from src.tools.native_viewer_export import (  # noqa: E402
    ball,
    flight_tracer,
    impact_card,
)
from src.tools.native_viewer_export.ball import BallPlacement  # noqa: E402
from src.tools.native_viewer_export.club_pose import (  # noqa: E402
    ClubPoseModel,
    face_roll_calibrated_spec_bytes,
)
from src.tools.native_viewer_export.core import SwingInput  # noqa: E402

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)
#: final bend (rad) about the first shaft axis of every link; the head moves
#: by decimetres, so the rigid and tip readings cannot agree by rounding
BEND_RAD = 0.1
#: smallest tip offset from the rigid prediction the case must show
MIN_OFFSET_M = 0.05
N_STATES, DT_S = 9, 0.005
IMPACT_INDEX = 4


@pytest.fixture(scope="module")
def rigid_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chain_spec(rigid_spec: dict[str, Any]) -> dict[str, Any]:
    return apply_shaft_chain(rigid_spec, build_shaft_chain(DRIVER))


@pytest.fixture(scope="module")
def chain_model(chain_spec: dict[str, Any]) -> ClubPoseModel:
    return ClubPoseModel(json.dumps(chain_spec).encode())


@pytest.fixture(scope="module")
def rigid_model(rigid_spec: dict[str, Any]) -> ClubPoseModel:
    return ClubPoseModel(json.dumps(rigid_spec).encode())


def _bend_names(spec: dict[str, Any]) -> list[str]:
    return [n for n in passive_coordinates(spec) if "bend_x" in n]


def _q(spec: dict[str, Any]) -> np.ndarray:
    """States with every joint at zero and the shaft bend ramping to BEND_RAD."""
    order = list(spec["coordinate_order"])
    q = np.zeros((N_STATES, len(order)))
    ramp = np.linspace(0.0, BEND_RAD, N_STATES)
    for name in _bend_names(spec):
        q[:, order.index(name)] = ramp
    return q


def _coords(spec: dict[str, Any], row: np.ndarray) -> dict[str, float]:
    return dict(zip(spec["coordinate_order"], map(float, row), strict=True))


def _rigid_coords(rigid: dict[str, Any], coords: dict[str, float]) -> dict[str, float]:
    return {name: coords[name] for name in rigid["coordinate_order"]}


def _swing(spec: dict[str, Any]) -> SwingInput:
    q = _q(spec)
    order = tuple(spec["coordinate_order"])
    bundle = InputBundle(
        spec_bytes=json.dumps(spec).encode(),
        coordinate_order=order,
        dt_s=DT_S,
        q0=q[0],
        v0=np.zeros(len(order)),
        efforts=np.zeros((N_STATES - 1, len(order))),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
        provenance={"impact_time_s": IMPACT_INDEX * DT_S},
    )
    return SwingInput(bundle, q, "t", "driver", "mujoco")


@pytest.fixture(scope="module")
def bent(chain_spec: dict[str, Any]) -> dict[str, float]:
    return _coords(chain_spec, _q(chain_spec)[-1])


def test_case_moves_the_tip_off_the_rigid_prediction(
    chain_spec: dict[str, Any],
    rigid_spec: dict[str, Any],
    chain_model: ClubPoseModel,
    rigid_model: ClubPoseModel,
    bent: dict[str, float],
) -> None:
    straight = _coords(chain_spec, _q(chain_spec)[0])
    rigid_straight = rigid_model.face_world(_rigid_coords(rigid_spec, straight))
    np.testing.assert_allclose(
        chain_model.face_world(straight).centre_m, rigid_straight.centre_m, atol=1e-9
    )
    rigid = rigid_model.face_world(_rigid_coords(rigid_spec, bent)).centre_m
    tip = chain_model.face_world(bent).centre_m
    assert np.linalg.norm(tip - rigid) > MIN_OFFSET_M


def test_club_roll_calibrates_from_the_bent_head(
    chain_spec: dict[str, Any],
    rigid_spec: dict[str, Any],
    bent: dict[str, float],
) -> None:
    def roll(spec: dict[str, Any], coords: dict[str, float]) -> float:
        out = face_roll_calibrated_spec_bytes(
            json.dumps(spec).encode(), coords, DESIGN_LOFT_AT_ADDRESS
        )
        return float(json.loads(out)["club"]["face_roll_deg"])

    straight = _coords(chain_spec, _q(chain_spec)[0])
    assert roll(chain_spec, straight) == pytest.approx(
        roll(rigid_spec, _rigid_coords(rigid_spec, straight)), abs=1e-9
    )
    assert roll(chain_spec, bent) != pytest.approx(
        roll(rigid_spec, _rigid_coords(rigid_spec, bent)), abs=1e-3
    )


def test_address_ball_sits_at_the_bent_face(
    chain_spec: dict[str, Any],
    rigid_spec: dict[str, Any],
    chain_model: ClubPoseModel,
    rigid_model: ClubPoseModel,
) -> None:
    swing = _swing(chain_spec)
    bent_first = _q(chain_spec)
    bent_first[0] = bent_first[-1]
    swing = SwingInput(
        swing.bundle, bent_first, swing.swing, swing.club, swing.rollout_engine
    )
    face = ball._address_face(swing)
    assert not isinstance(face, str)
    tip = chain_model.face_world(_coords(chain_spec, bent_first[0]))
    np.testing.assert_allclose(face[0], tip.centre_m, atol=1e-9)
    np.testing.assert_allclose(face[1], tip.normal, atol=1e-9)
    coords = _rigid_coords(rigid_spec, _coords(chain_spec, bent_first[0]))
    rigid = rigid_model.face_world(coords).centre_m
    assert np.linalg.norm(face[0] - rigid) > MIN_OFFSET_M


def test_strike_ball_searches_the_bent_face_path(
    chain_spec: dict[str, Any],
    rigid_spec: dict[str, Any],
    chain_model: ClubPoseModel,
    rigid_model: ClubPoseModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, np.ndarray] = {}

    def capture(times: Any, centres: Any, normals: Any, *_a: Any, **_k: Any) -> Any:
        seen["centres"], seen["normals"] = np.asarray(centres), np.asarray(normals)
        raise ValueError("captured")

    settings = ball_strike.load_ball_strike_settings()
    monkeypatch.setattr(
        ball,
        "load_ball_strike_settings",
        lambda: ball_strike.BallStrikeSettings(
            "strike",
            settings.search_window_s,
            settings.max_shift_m,
            settings.max_tee_height_m,
        ),
    )
    monkeypatch.setattr(ball, "strike_ball_centre", capture)
    swing = _swing(chain_spec)
    nominal = BallPlacement((0.0, 0.0, 0.0), 0.02, ball.SOURCE_ADDRESS, None)
    impact = IMPACT_INDEX * DT_S
    _, record = ball.strike_ball(swing, nominal, impact)
    assert record is not None and "captured" in record["unavailable"]
    times = np.array([swing.time_s(i) for i in range(N_STATES)])
    rows = np.nonzero(np.abs(times - impact) <= settings.search_window_s)[0]
    faces = [chain_model.face_world(_coords(chain_spec, swing.q[i])) for i in rows]
    np.testing.assert_allclose(seen["centres"], [f.centre_m for f in faces], atol=1e-9)
    np.testing.assert_allclose(seen["normals"], [f.normal for f in faces], atol=1e-9)
    last = _rigid_coords(rigid_spec, _coords(chain_spec, swing.q[rows[-1]]))
    rigid = rigid_model.face_world(last).centre_m
    assert np.linalg.norm(seen["centres"][-1] - rigid) > MIN_OFFSET_M


def test_impact_card_and_flight_series_read_the_tip(
    chain_spec: dict[str, Any],
    rigid_spec: dict[str, Any],
    chain_model: ClubPoseModel,
    rigid_model: ClubPoseModel,
) -> None:
    assert flight_tracer.club_series is impact_card._club_series
    swing = _swing(chain_spec)
    series = impact_card._club_series(swing)
    rows = [_coords(chain_spec, r) for r in swing.q]
    tip = np.array([chain_model.face_world(c).centre_m for c in rows])
    rigid = np.array(
        [rigid_model.face_world(_rigid_coords(rigid_spec, c)).centre_m for c in rows]
    )
    np.testing.assert_allclose(series.face_center_m, tip, atol=1e-9)
    times = np.array([swing.time_s(i) for i in range(N_STATES)])
    np.testing.assert_allclose(
        series.velocity_mps, np.gradient(tip, times, axis=0), atol=1e-9
    )
    rigid_velocity = np.gradient(rigid, times, axis=0)
    gap = np.linalg.norm(series.velocity_mps - rigid_velocity, axis=1)
    assert gap.max() * DT_S * (N_STATES - 1) > MIN_OFFSET_M
    normals = [chain_model.face_world(c).normal for c in rows]
    np.testing.assert_allclose(series.face_normal, normals, atol=1e-9)
