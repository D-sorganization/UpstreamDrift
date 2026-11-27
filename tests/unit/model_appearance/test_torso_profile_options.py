"""Optional trunk-profile settings: slant, squareness, forward, rings (#15292)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.torso_profile import (
    TorsoFrame,
    TorsoProfile,
    TorsoStation,
    torso_mesh,
)

pytestmark = pytest.mark.unit
UP, FRONT, SIDE = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]
FRAME = TorsoFrame(np.zeros(3), UP, np.array(SIDE), 0.2, 0.1, FRONT)


def _mesh_stations() -> tuple[TorsoStation, ...]:
    return TorsoStation(0.0, 1.0, 1.0), TorsoStation(0.5, 1.0, 1.0)


def _mesh(top: dict | None = None, **profile: float):
    rows = _mesh_stations()
    top_row = TorsoStation(1.0, 0.5, 0.5, **(top or {}))
    prof = TorsoProfile((*rows, top_row), rings=6, sides=16, **profile)
    return torso_mesh(FRAME, prof, **({"cap_inset_m": 0.01} if profile else {}))


def _tip(mesh) -> np.ndarray:
    return mesh.vertices[mesh.vertices[:, 2] > 0.95]


def test_slant_lowers_the_front_and_keeps_the_back() -> None:
    top = _mesh({"slant": 0.2}).vertices
    assert top[top[:, 0] > 0.04, 2].max() < top[top[:, 0] < -0.04, 2].max() - 0.15


def test_forward_shifts_the_whole_section_to_the_front() -> None:
    shift = _tip(_mesh({"forward": 0.05}))[:, 0].mean() - _tip(_mesh())[:, 0].mean()
    assert shift == pytest.approx(0.05, abs=1e-6)


def test_station_squareness_rounds_the_section() -> None:
    def fill(top: dict) -> float:
        x, y = _tip(_mesh(top))[:, :2].T
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        return area / (np.ptp(x) * np.ptp(y))

    assert fill({"squareness": 2.0}) < fill({"squareness": 2.6})


def test_fine_rings_and_cap_inset_shape_the_mesh() -> None:
    fine = _mesh(fine_from=0.8, fine_rings=5)
    assert len(fine.vertices) > len(_mesh().vertices)
    assert fine.vertices[:, 2].max() == pytest.approx(0.99)  # inset 1 cm
    plain = TorsoProfile((*_mesh_stations(), TorsoStation(1.0, 0.5, 0.5)), rings=6)
    with pytest.raises(ValueError, match="positive finite"):
        torso_mesh(FRAME, plain, cap_inset_m=-1.0)


def test_invalid_station_options_raise() -> None:
    with pytest.raises(ValueError, match="squareness"):
        _mesh({"squareness": 1.5})
    with pytest.raises(ValueError, match="slants"):
        _mesh({"slant": -0.1})
