"""Cross-engine parity qualification for interface capability and mobility ellipsoids (CAP-10, #14050).

Part of epic #14038 (CAP-10).
Builds MuJoCo, Drake, and Pinocchio capability sources from the same anthropometric
driver specification at reference poses, comparing:
- matrices, ranks and subspaces;
- eigenvalues within 1 % and axes within 2 degrees (skipping near-repeated eigenvalues);
- absolute tolerances near zero;
- acceleration and force capacity ellipsoid radii and centres.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.loop_nullspace.capacity import CapacitySubject
from src.shared.python.spatial_algebra.interface_ellipsoids import (
    PARITY_LEDGER_PATH,
    InterfaceEllipsoidKind,
)
from src.shared.python.spatial_algebra.operational_space import CapacityEllipsoid

pytestmark = pytest.mark.parity
ROOT = Path(__file__).resolve().parents[2]
ANTHRO_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def _require_real_mujoco() -> Any:
    mj = pytest.importorskip("mujoco")
    if type(mj).__module__ == "unittest.mock" or not hasattr(mj, "MjModel"):
        pytest.skip("Real mujoco installation required (found mock)")
    from src.engines.physics_engines.mujoco.python.hand_mobility import (
        HandMobilitySource,
    )

    return HandMobilitySource


def _require_real_drake() -> Any:
    pytest.importorskip("pydrake")
    try:
        import pydrake.all as api

        if type(api).__module__ == "unittest.mock" or not hasattr(
            api, "MultibodyPlant"
        ):
            pytest.skip("Real pydrake installation required (found mock)")
    except ImportError:
        pytest.skip("pydrake not installed")
    from src.engines.physics_engines.drake.python.force_adapter import (
        DrakeCapabilitySource,
    )

    return DrakeCapabilitySource


def _require_real_pinocchio() -> Any:
    pin = pytest.importorskip("pinocchio")
    if type(pin).__module__ == "unittest.mock" or not all(
        hasattr(pin, n) for n in ("rnea", "crba", "SE3", "Model")
    ):
        pytest.skip("Real pinocchio installation required (found mock)")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioCapabilitySource,
    )

    return PinocchioCapabilitySource


@pytest.fixture(scope="module")
def anthro_spec_bytes() -> bytes:
    return ANTHRO_SPEC.read_bytes()


@pytest.fixture(scope="module")
def subject() -> CapacitySubject:
    return CapacitySubject(mass_kg=75.0, stature_m=1.75)


@pytest.fixture(scope="module")
def mujoco_source(anthro_spec_bytes: bytes, subject: CapacitySubject) -> Any | None:
    source_cls = _require_real_mujoco()
    return source_cls(anthro_spec_bytes, torque_limits=subject)


@pytest.fixture(scope="module")
def drake_source(anthro_spec_bytes: bytes, subject: CapacitySubject) -> Any | None:
    source_cls = _require_real_drake()
    return source_cls(anthro_spec_bytes, torque_limits=subject)


@pytest.fixture(scope="module")
def pinocchio_source(anthro_spec_bytes: bytes, subject: CapacitySubject) -> Any | None:
    source_cls = _require_real_pinocchio()
    return source_cls(anthro_spec_bytes, torque_limits=subject)


def _assert_subspace_and_eigenvalues_parity(
    w_ref: np.ndarray,
    w_test: np.ndarray,
    *,
    rtol: float = 0.01,
    atol: float = 1e-4,
) -> None:
    """Compare mobility matrices, ranks, subspaces, and eigenvalues within 1%."""
    assert np.allclose(w_test, w_test.T, atol=1e-10), "Matrix must be symmetric"
    assert np.linalg.matrix_rank(w_test) == np.linalg.matrix_rank(w_ref), (
        "Ranks must match"
    )

    evals_ref = np.linalg.eigvalsh(w_ref)
    evals_test = np.linalg.eigvalsh(w_test)
    assert np.allclose(evals_test, evals_ref, rtol=rtol, atol=atol), (
        f"Eigenvalues differ: {evals_test} vs {evals_ref}"
    )


def _assert_eigenvector_axes_parity(
    axes_ref: np.ndarray,
    axes_test: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    min_relative_gap: float = 0.05,
    max_angle_deg: float = 2.0,
) -> None:
    """Compare eigenvector axes within 2 degrees, skipping near-repeated eigenvalues."""
    n = len(eigenvalues)
    for i in range(n):
        lam = eigenvalues[i]
        is_isolated = all(
            abs(lam - eigenvalues[j]) > min_relative_gap * max(abs(lam), 1e-9)
            for j in range(n)
            if j != i
        )
        if not is_isolated:
            continue
        v_ref = axes_ref[:, i]
        v_test = axes_test[:, i]
        cos_theta = min(
            1.0,
            abs(float(np.dot(v_ref, v_test)))
            / (np.linalg.norm(v_ref) * np.linalg.norm(v_test)),
        )
        angle_deg = float(np.degrees(np.arccos(cos_theta)))
        assert angle_deg <= max_angle_deg, (
            f"Axis {i} angle {angle_deg:.3f} deg exceeds {max_angle_deg} deg"
        )


def _assert_capacity_ellipsoids_parity(
    ell_ref: CapacityEllipsoid,
    ell_test: CapacityEllipsoid,
    *,
    rtol: float = 0.01,
    atol: float = 1e-2,
    max_angle_deg: float = 2.0,
) -> None:
    """Compare capacity ellipsoid radii within 1% and principal axes within 2 degrees."""
    assert np.allclose(
        np.sort(ell_test.radii), np.sort(ell_ref.radii), rtol=rtol, atol=atol
    ), f"Radii mismatch: {ell_test.radii} vs {ell_ref.radii}"

    _assert_eigenvector_axes_parity(
        ell_ref.axes, ell_test.axes, ell_ref.radii**2, max_angle_deg=max_angle_deg
    )
    if ell_ref.centre is not None and ell_test.centre is not None:
        assert np.allclose(ell_test.centre, ell_ref.centre, atol=1e-3), (
            f"Centre offset mismatch: {ell_test.centre} vs {ell_ref.centre}"
        )


@pytest.mark.parametrize("hand", ["lead", "trail"])
@pytest.mark.parametrize("mode", ["constrained", "unconstrained"])
def test_cross_engine_mobility_parity(
    mujoco_source: Any,
    drake_source: Any,
    pinocchio_source: Any,
    hand: str,
    mode: str,
) -> None:
    """Compare mobility matrices, eigenvalues (1%) and axes (2 deg) across engines."""
    q = np.zeros(len(mujoco_source.coordinate_order))
    mj_hm = mujoco_source.hand_mobility(q, mode=mode)[hand]
    drake_hm = drake_source.hand_mobility(q, mode=mode)[hand]
    pin_hm = pinocchio_source.hand_mobility(q, mode=mode)[hand]

    for test_hm in (drake_hm, pin_hm):
        assert np.allclose(test_hm.centre, mj_hm.centre, atol=1e-3), (
            f"Grip centre mismatch: {test_hm.centre} vs {mj_hm.centre}"
        )
        _assert_subspace_and_eigenvalues_parity(mj_hm.mobility, test_hm.mobility)
        _assert_eigenvector_axes_parity(
            mj_hm.mobility_axes, test_hm.mobility_axes, mj_hm.mobility_eigenvalues
        )


@pytest.mark.parametrize("hand", ["lead", "trail"])
def test_cross_engine_capacity_ellipsoids_parity(
    mujoco_source: Any,
    drake_source: Any,
    pinocchio_source: Any,
    hand: str,
) -> None:
    """Compare acceleration- and force-capacity ellipsoids across engines."""
    q = np.zeros(len(mujoco_source.coordinate_order))
    mj_hm = mujoco_source.hand_mobility(q, mode="constrained")[hand]
    drake_hm = drake_source.hand_mobility(q, mode="constrained")[hand]
    pin_hm = pinocchio_source.hand_mobility(q, mode="constrained")[hand]

    assert mj_hm.acceleration_capacity is not None
    assert mj_hm.force_capacity is not None

    for test_hm in (drake_hm, pin_hm):
        assert test_hm.acceleration_capacity is not None
        assert test_hm.force_capacity is not None
        _assert_capacity_ellipsoids_parity(
            mj_hm.acceleration_capacity, test_hm.acceleration_capacity
        )
        _assert_capacity_ellipsoids_parity(mj_hm.force_capacity, test_hm.force_capacity)


def test_interface_ellipsoid_parity_ledger_has_parity_entries() -> None:
    """Ledger src/config/interface_ellipsoid_parity.json qualifies qualified engines."""
    data = json.loads(PARITY_LEDGER_PATH.read_text(encoding="utf-8"))
    engines = data["engines"]
    for engine in ("mujoco", "drake", "pinocchio"):
        assert engine in engines
        for kind in InterfaceEllipsoidKind:
            assert engines[engine][kind.value]["status"] == "parity"

    assert engines["opensim"]["mobility"]["status"] == "gap"
    assert engines["opensim"]["mobility"]["issue"] == 14051
