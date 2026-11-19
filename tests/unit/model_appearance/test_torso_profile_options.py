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

FRAME = TorsoFrame(
    np.zeros(3),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0]),
    0.2,
    0.1,
    np.array([1.0, 0.0, 0.0]),
)


def _profile(**top: float) -> TorsoProfile:
    rows = [TorsoStation(0.0, 1.0, 1.0), TorsoStation(0.5, 1.0, 1.0)]
    return TorsoProfile((*rows, TorsoStation(1.0, 0.5, 0.5, **top)), rings=6, sides=16)


def _top(mesh) -> np.ndarray:
    return mesh.vertices[mesh.vertices[:, 2] > 0.5]


def test_default_options_change_nothing() -> None:
    plain = torso_mesh(FRAME, _profile())
    assert np.array_equal(
        plain.vertices, torso_mesh(FRAME, _profile(slant=0.0)).vertices
    )
    assert plain.volume() > 0.0


def test_slant_lowers_the_front_and_keeps_the_back() -> None:
    top = _top(torso_mesh(FRAME, _profile(slant=0.2)))
    front, back = top[top[:, 0] > 0.04], top[top[:, 0] < -0.04]
    assert front[:, 2].max() < back[:, 2].max() - 0.15


def test_forward_shifts_the_whole_section_to_the_front() -> None:
    def tip(**top: float) -> float:
        mesh = torso_mesh(FRAME, _profile(**top))
        return float(mesh.vertices[mesh.vertices[:, 2] > 0.99, 0].mean())

    assert tip(forward=0.05) - tip() == pytest.approx(0.05, abs=1e-6)


def test_station_squareness_rounds_the_section() -> None:
    boxy = torso_mesh(FRAME, _profile(squareness=2.6))
    round_ = torso_mesh(FRAME, _profile(squareness=2.0))

    def fill(mesh) -> float:
        ring = mesh.vertices[mesh.vertices[:, 2] > 0.99][:, :2]
        x, y = ring[:, 0], ring[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        return area / (np.ptp(x) * np.ptp(y))

    assert fill(round_) < fill(boxy)


def test_fine_rings_add_vertices_above_fine_from() -> None:
    fine = TorsoProfile(
        _profile().stations, rings=6, sides=16, fine_from=0.8, fine_rings=5
    )
    assert len(torso_mesh(FRAME, fine).vertices) > len(
        torso_mesh(FRAME, _profile()).vertices
    )


def test_cap_inset_lowers_the_top_and_rejects_a_negative_one() -> None:
    low = torso_mesh(FRAME, _profile(), cap_inset_m=0.01)
    assert _top(low)[:, 2].max() == pytest.approx(1.0 - 0.01)
    with pytest.raises(ValueError, match="positive finite"):
        torso_mesh(FRAME, _profile(), cap_inset_m=-0.1)


def test_invalid_station_options_raise() -> None:
    with pytest.raises(ValueError, match="squareness"):
        _profile(squareness=1.5)
    with pytest.raises(ValueError, match="slants"):
        _profile(slant=-0.1)
