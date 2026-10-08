"""Frame and name conventions of the lift pack audit."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.pack_audit import frames, names

pytestmark = pytest.mark.unit


def test_y_up_vector_maps_to_canonical_z_up() -> None:
    # OpenSim up (+Y) must become canonical up (+Z); OpenSim -Z is canonical +Y (left).
    assert np.allclose(frames.to_canonical("y_up", [0, 1, 0]), [0, 0, 1])
    assert np.allclose(frames.to_canonical("y_up", [0, 0, -1]), [0, 1, 0])
    assert np.allclose(frames.to_canonical("y_up", [1, 0, 0]), [1, 0, 0])


def test_round_trip_is_identity() -> None:
    v = [0.3, -1.2, 2.5]
    back = frames.from_canonical("y_up", frames.to_canonical("y_up", v))
    assert np.allclose(back, v)


@pytest.mark.parametrize("bad", [[1, 2], [float("nan"), 0, 0], [[1, 2, 3]]])
def test_to_canonical_rejects_malformed(bad: list) -> None:
    with pytest.raises(ValueError):
        frames.to_canonical("z_up", bad)


def test_unknown_frame_rejected() -> None:
    with pytest.raises(ValueError, match="unknown engine frame"):
        frames.to_canonical("x_up", [0, 0, 0])


def test_canonical_set_matches_the_standard_count() -> None:
    # 4 midline + 12 sided patterns x 2 sides, per biomech_parity_standard.json.
    assert len(names.CANONICAL_COORDINATES) == 28
    assert len(set(names.CANONICAL_COORDINATES)) == 28


@pytest.mark.parametrize(
    ("engine", "canonical", "native"),
    [
        ("mujoco", "knee_l_flex", "knee_l_flex"),
        ("opensim", "wrist_r_deviate", "wrist_r_deviation"),
        ("opensim", "ankle_l_invert", "ankle_l_inversion"),
        ("drake", "knee_r_flex", "knee_r"),
        ("pinocchio", "neck_flex", "neck"),
        ("pinocchio", "elbow_l_flex", "elbow_l"),
    ],
)
def test_engine_names_round_trip(engine: str, canonical: str, native: str) -> None:
    assert names.engine_coordinate_name(engine, canonical) == native
    assert names.canonical_coordinate_name(engine, native) == canonical


def test_every_engine_mapping_is_injective() -> None:
    for engine in names.ENGINES:
        mapped = [
            names.engine_coordinate_name(engine, c) for c in names.CANONICAL_COORDINATES
        ]
        assert len(set(mapped)) == len(mapped)


def test_unknown_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        names.engine_coordinate_name("jaxsim", "knee_l_flex")
    with pytest.raises(ValueError):
        names.engine_coordinate_name("mujoco", "knee_flex")


def test_phase_key_expansion_exact_and_interpreted() -> None:
    assert names.expand_phase_key("mujoco", "hip_l_flex") == (["hip_l_flex"], True)
    keys, exact = names.expand_phase_key("opensim", "hip_flexion")
    assert keys == ["hip_l_flex", "hip_r_flex"]
    assert exact is False
    assert names.expand_phase_key("opensim", "mystery_joint") == ([], False)


def test_canonical_spelling_foreign_to_the_pack_is_flagged() -> None:
    # Pinocchio's URDF joint is ``knee_l``; an objective written as ``knee_l_flex``
    # still resolves but is reported as interpreted (name mismatch).
    assert names.expand_phase_key("pinocchio", "knee_l_flex") == (
        ["knee_l_flex"],
        False,
    )
    assert names.expand_phase_key("pinocchio", "knee_l") == (["knee_l_flex"], True)
