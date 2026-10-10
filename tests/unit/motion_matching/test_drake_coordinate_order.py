"""Drake full-body coordinate-order contract (#12042 slice 5).

Drake numbers the positions of a parsed URDF in its own tree order, which for
the branched full-body spec is *not* the spec ``coordinate_order`` (the spine,
torso, arm and neck coordinates are permuted). Every array-valued public API of
``FullBodyDrakeModel`` and ``DrakeFullBodyIK`` takes and returns spec-ordered
coordinates, so the adapters must permute on the way into the plant and back.

Before the fix the IK wrote spec-ordered vectors straight into the plant: the
torso twist landed on the ``SpineInputX`` slot and its ±35° document bound,
which is why Drake IK under-twisted the thorax on capture-A.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane
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
TWIST_RAD = float(np.radians(-60.0))


def _require_real_drake() -> None:
    try:
        import pydrake.all as drake_all
    except ImportError as exc:
        pytest.skip(f"pydrake not importable: {exc}")
    if type(drake_all).__module__ == "unittest.mock" or not hasattr(
        drake_all, "MultibodyPlant"
    ):
        pytest.skip("real pydrake runtime required (found mock/stub)")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def attachments(spec: dict[str, Any]) -> dict[str, tuple[str, tuple[float, ...]]]:
    return {
        label: (att["body"], tuple(float(v) for v in att["offset_m"]))
        for label, att in spec["marker_attachments"].items()
        if att.get("offset_m") is not None
    }


@pytest.fixture(scope="module")
def ik(spec: dict[str, Any], attachments: dict[str, Any]) -> Any:
    _require_real_drake()
    from src.engines.physics_engines.drake.python.full_body_ik import (
        DrakeFullBodyIK,
    )

    return DrakeFullBodyIK(spec, attachments=attachments)


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


def _body_mask(spec: dict[str, Any], ik: Any) -> np.ndarray:
    """Body markers; the club markers are excluded because the shared spec FK
    places the club frames differently from both Drake and MuJoCo (a separate
    follow-up, not a coordinate-order effect)."""
    club = spec["closure"]["body_b"]
    club_frames = {f["name"] for f in spec["frames"] if f["body"] == club}
    offsets = ik.marker_bodies_and_offsets
    return np.array(
        [offsets[label][0] not in club_frames | {club} for label in ik.labels]
    )


def _shared_fk(spec: dict[str, Any], ik: Any, q: np.ndarray) -> np.ndarray:
    """Shared spec FK of every marker (NaN for frame-attached markers)."""
    positions = compute_spec_marker_positions(
        spec,
        q,
        marker_offsets=ik.marker_bodies_and_offsets,
        coordinate_names=spec["coordinate_order"],
    )
    out = np.array([positions[label] for label in ik.labels], dtype=float)
    out[~_body_mask(spec, ik)] = np.nan
    return out


def _assert_matches_shared_fk(
    spec: dict[str, Any], ik: Any, q: np.ndarray, atol: float = 1e-9
) -> None:
    mask = _body_mask(spec, ik)
    assert mask.sum() >= 20
    np.testing.assert_allclose(
        ik.marker_positions(q)[mask], _shared_fk(spec, ik, q)[mask], atol=atol
    )


def test_plant_position_order_differs_from_spec_order(ik: Any) -> None:
    """Precondition for the other tests: the permutation is real."""
    order = ik.model.plant_position_indices
    assert sorted(order) == list(range(ik.nq))
    assert list(order) != list(range(ik.nq))


def test_plant_round_trip_preserves_spec_order(ik: Any, spec: dict[str, Any]) -> None:
    q = _random_q(spec, 1)
    raw = ik.model.to_plant_positions(q)
    np.testing.assert_array_equal(ik.model.from_plant_positions(raw), q)
    with pytest.raises(ValueError, match="coordinates"):
        ik.model.to_plant_positions(q[:-1])


def test_torso_twist_matches_shared_spec_fk(ik: Any, spec: dict[str, Any]) -> None:
    """Axis parity: a pure TorsoInput twist moves the markers as the spec says."""
    q = _named(spec, TranslationInputZ=1.0, TorsoInput=TWIST_RAD)
    _assert_matches_shared_fk(spec, ik, q)


def test_ik_and_model_array_fk_match_named_fk(
    ik: Any, spec: dict[str, Any], attachments: dict[str, Any]
) -> None:
    names = spec["coordinate_order"]
    for seed in range(3):
        q = _random_q(spec, seed)
        named = ik.model.marker_positions(dict(zip(names, q, strict=True)), attachments)
        np.testing.assert_allclose(
            ik.model.marker_positions(q, attachments), named, atol=1e-9
        )
        _assert_matches_shared_fk(spec, ik, q)


def test_ik_marker_jacobian_columns_follow_spec_order(
    ik: Any, spec: dict[str, Any]
) -> None:
    q = _random_q(spec, 7)
    analytic = ik._marker_jacobian(ik.marker_positions(q))
    step = 1e-6
    for index in range(ik.nq):
        dq = np.zeros(ik.nq)
        dq[index] = step
        numeric = (ik.marker_positions(q + dq) - ik.marker_positions(q - dq)) / (
            2 * step
        )
        np.testing.assert_allclose(analytic[:, :, index], numeric, atol=1e-6)


def test_ik_recovers_torso_twist_inside_spine_bounds(
    ik: Any, spec: dict[str, Any]
) -> None:
    """A -60° twist is recovered with the document bounds (SpineInputX ±35°)."""
    names = spec["coordinate_order"]
    truth = _named(spec, TranslationInputZ=1.0, TorsoInput=TWIST_RAD)
    targets = _shared_fk(spec, ik, truth)
    bounds = {
        name: (float(np.radians(lo)), float(np.radians(hi)))
        for name, (lo, hi) in spec["coordinate_ranges_deg"].items()
        if name in names
    }
    fit = ik.solve_pose(
        np.nan_to_num(targets),
        _body_mask(spec, ik),
        _named(spec, TranslationInputZ=1.0),
        ground=GroundPlane(normal=(0.0, 0.0, 1.0), height_m=-5.0),
        bounds=bounds,
        prior_weight=1e-6,
        closure_weight=0.0,
        iterations=100,
    )
    assert fit.marker_rms_m < 1e-4
    assert np.degrees(fit.q[names.index("TorsoInput")]) == pytest.approx(-60.0, abs=0.5)


def test_affine_dynamics_array_matches_mapping(ik: Any, spec: dict[str, Any]) -> None:
    names = spec["coordinate_order"]
    q = _random_q(spec, 3)
    v = np.random.default_rng(4).uniform(-0.5, 0.5, len(names))
    a_arr, b_arr = ik.model.affine_dynamics(q, v)
    a_map, b_map = ik.model.affine_dynamics(
        dict(zip(names, q, strict=True)), dict(zip(names, v, strict=True))
    )
    np.testing.assert_allclose(a_arr, a_map, atol=1e-8)
    np.testing.assert_allclose(b_arr, b_map, atol=1e-8)
    tau = np.zeros(len(names))
    tau[6:] = np.random.default_rng(5).uniform(-5.0, 5.0, len(names) - 6)
    accel = ik.model.accelerations(
        dict(zip(names, q, strict=True)),
        dict(zip(names, v, strict=True)),
        dict(zip(names, tau, strict=True)),
    )
    np.testing.assert_allclose(
        a_arr @ tau[6:] + b_arr, [accel[n] for n in names], atol=1e-6
    )


def test_step_with_mapping_matches_array(ik: Any, spec: dict[str, Any]) -> None:
    names = spec["coordinate_order"]
    q = _random_q(spec, 8)
    v = np.random.default_rng(9).uniform(-0.5, 0.5, len(names))
    tau = np.zeros(len(names))
    q_arr, v_arr = ik.model.step(q, v, tau, 1e-3)
    q_map, v_map = ik.model.step(
        dict(zip(names, q, strict=True)),
        dict(zip(names, v, strict=True)),
        dict(zip(names, tau, strict=True)),
        1e-3,
    )
    np.testing.assert_allclose(q_arr, q_map, atol=1e-12)
    np.testing.assert_allclose(v_arr, v_map, atol=1e-12)
