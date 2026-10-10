"""Hand-to-hand relative-pose drift of the prescribed grip input (#11986)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    GripInterface,
    hand_frame_states,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.hand_drift import hand_relative_drift

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros((a.size, 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = c, -s, s, c
    out[:, 2, 2] = 1.0
    return out


def _frames(n: int = 5):
    rng = np.random.default_rng(0)
    return _rot_z(rng.uniform(-1, 1, n)), rng.normal(size=(n, 3))


def test_rigid_pair_has_zero_drift() -> None:
    r_l, p_l = _frames()
    off_r, off_p = _rot_z(np.array([0.3]))[0], np.array([0.08, 0.025, 0.0])
    r_t = np.einsum("nij,jk->nik", r_l, off_r)
    p_t = p_l + np.einsum("nij,j->ni", r_l, off_p)
    drift = hand_relative_drift({"L": (r_l, p_l), "R": (r_t, p_t)})
    assert all(v < 1e-6 for v in drift.peak.values())


def test_known_drift_is_recovered_in_the_lead_frame() -> None:
    r_l, p_l = _frames()
    n = r_l.shape[0]
    shift = np.zeros((n, 3))
    shift[:, 0] = np.linspace(0.0, 2e-3, n)  # 2 mm along the lead x axis
    shift[:, 1] = np.linspace(0.0, 1e-3, n)  # 1 mm across
    twist = _rot_z(np.radians(np.linspace(0.0, 0.5, n)))
    r_t = np.einsum("nij,njk->nik", r_l, twist)
    p_t = p_l + np.einsum("nij,nj->ni", r_l, shift)
    peak = hand_relative_drift({"L": (r_l, p_l), "R": (r_t, p_t)}).peak
    assert peak["along_mm"] == pytest.approx(2.0, abs=1e-9)
    assert peak["across_mm"] == pytest.approx(1.0, abs=1e-9)
    assert peak["rotation_deg"] == pytest.approx(0.5, abs=1e-6)


def test_malformed_series_are_rejected() -> None:
    r_l, p_l = _frames()
    with pytest.raises(ValueError):
        hand_relative_drift({"L": (r_l, p_l), "R": (r_l[:3], p_l[:3])})
    with pytest.raises(ValueError):
        hand_relative_drift({"L": (r_l, p_l[:, :2]), "R": (r_l, p_l)})


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_prescribed_swing_input_is_rigid(club: str) -> None:
    """Both hand frames ride one weld club pose, so the input cannot drift."""
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.grip_bushing import (
        WeldClubKinematics,
    )

    spec_bytes = (MODELS / f"full_body_spec_anthro_{club}.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz", FIXTURES / "address_poses.json", club, names
    )
    interface = GripInterface.from_spec(spec)
    kin = WeldClubKinematics(spec_bytes, names)
    idx = np.linspace(0, swing.q.shape[0] - 1, 60).astype(int)
    series: dict[str, tuple[list, list]] = {"L": ([], []), "R": ([], [])}
    for i in idx:
        states = hand_frame_states(
            kin.state(swing.q[i], np.zeros(swing.q.shape[1])), interface
        )
        for s in "LR":
            series[s][0].append(states[s].rotation)
            series[s][1].append(states[s].position_m)
    pose = {s: (np.array(series[s][0]), np.array(series[s][1])) for s in "LR"}
    peak = hand_relative_drift(pose).peak
    assert max(peak.values()) < 1e-6
