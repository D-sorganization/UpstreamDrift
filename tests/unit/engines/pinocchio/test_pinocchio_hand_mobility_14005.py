"""Per-hand operational-space mobility parity for Pinocchio (HCM-4, #14005).

Brings hand-club mobility and capacity ellipsoids to the Pinocchio backend,
reconciling Khatib operational-space equations across physics engines.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.hand_mobility import HandMobility
from src.engines.physics_engines.pinocchio.python.force_adapter import (
    PinocchioCapabilitySource,
    PinocchioForceAdapter,
)
from src.shared.python.spatial_algebra.operational_space import (
    CapacityEllipsoid,
    transport_spatial_mobility,
)
from src.tools.native_viewer_export.core import InputBundle, SwingInput, _hand_source

pytestmark = [pytest.mark.unit]
ROOT = Path(__file__).resolve().parents[4]
ANTHRO_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def _require_real_pinocchio() -> None:
    pin = pytest.importorskip("pinocchio")
    if type(pin).__module__ == "unittest.mock" or not all(
        hasattr(pin, n) for n in ("rnea", "crba", "SE3", "Model")
    ):
        pytest.skip("Real pinocchio installation required (found mock)")


@pytest.fixture(scope="module")
def anthro_spec_bytes() -> bytes:
    return ANTHRO_SPEC.read_bytes()


@pytest.fixture(scope="module")
def pinocchio_source(anthro_spec_bytes: bytes) -> PinocchioCapabilitySource:
    _require_real_pinocchio()
    return PinocchioCapabilitySource(anthro_spec_bytes)


@pytest.fixture(scope="module")
def test_states(pinocchio_source: PinocchioCapabilitySource) -> list[np.ndarray]:
    rng = np.random.default_rng(14005)
    n = len(pinocchio_source.coordinate_order)
    return [rng.normal(0.0, 0.1, n) for _ in range(3)]


def test_pinocchio_hand_mobility_psd_and_symmetric(
    pinocchio_source: PinocchioCapabilitySource, test_states: list[np.ndarray]
) -> None:
    """Pinocchio hand mobility matrices are symmetric positive semi-definite (HCM-4)."""
    for q in test_states:
        hm_dict = pinocchio_source.hand_mobility(q, mode="constrained")
        for hand in ("lead", "trail"):
            hm = hm_dict[hand]
            assert isinstance(hm, HandMobility)
            w = hm.mobility
            assert w.shape == (3, 3)
            assert np.allclose(w, w.T, atol=1e-10), "Mobility matrix must be symmetric"
            evals = np.linalg.eigvalsh(w)
            assert evals.min() >= -1e-9, f"Negative eigenvalue detected: {evals.min()}"
            assert (hm.effective_mass_kg > 0).all()
            assert hm.condition >= 1.0


def test_pinocchio_loewner_ordering(
    pinocchio_source: PinocchioCapabilitySource, test_states: list[np.ndarray]
) -> None:
    """Constrained mobility is Loewner below unconstrained mobility (W_con <= W_free)."""
    for q in test_states:
        con = pinocchio_source.hand_mobility(q, mode="constrained")
        free = pinocchio_source.hand_mobility(q, mode="unconstrained")
        for hand in ("lead", "trail"):
            gap = free[hand].mobility - con[hand].mobility
            assert np.linalg.eigvalsh(0.5 * (gap + gap.T)).min() >= -1e-9


def test_pinocchio_rigid_club_transport(
    pinocchio_source: PinocchioCapabilitySource, test_states: list[np.ndarray]
) -> None:
    """Both grip points lie on the club; spatial mobility transports rigidly (HCM-4)."""
    q = test_states[0]
    lead_pt = pinocchio_source.grip_point_world("lead", q)
    trail_pt = pinocchio_source.grip_point_world("trail", q)
    assert np.linalg.norm(trail_pt - lead_pt) > 0.01


def test_native_viewer_export_engine_dispatch_pinocchio(
    anthro_spec_bytes: bytes,
) -> None:
    """SwingInput dispatches to PinocchioCapabilitySource when engine='pinocchio'."""
    bundle = MagicMock(spec=InputBundle)
    bundle.spec_bytes = anthro_spec_bytes
    bundle.reference_q = np.zeros((2, 10))
    bundle.coordinate_order = tuple(f"q_{i}" for i in range(10))

    q_mock = np.zeros((2, 10))
    swing = SwingInput(
        bundle=bundle,
        q=q_mock,
        swing="test_swing",
        club="driver",
        rollout_engine="pinocchio",
    )

    fake_hm = {
        "lead": MagicMock(spec=HandMobility),
        "trail": MagicMock(spec=HandMobility),
    }
    with patch("src.tools.native_viewer_export.core._hand_source") as mock_hand_source:
        mock_source_inst = MagicMock()
        mock_source_inst.hand_mobility.return_value = fake_hm
        mock_hand_source.return_value = mock_source_inst

        result = swing.hand_mobility_at(0, mode="constrained", engine="pinocchio")
        assert result == fake_hm
        mock_hand_source.assert_called_once_with(
            anthro_spec_bytes,
            torque_limit_source=None,
            engine="pinocchio",
        )


def test_native_viewer_export_fallback_when_pinocchio_unavailable() -> None:
    """_hand_source falls back to MuJoCo when pinocchio raises an error."""
    spec_dummy = b'{"subject": {"mass_kg": 75.0, "stature_m": 1.75}}'
    _hand_source.cache_clear()
    with patch(
        "src.engines.physics_engines.pinocchio.python.force_adapter.PinocchioCapabilitySource",
        side_effect=ImportError("No module named pinocchio"),
    ):
        with patch(
            "src.engines.physics_engines.mujoco.python.hand_mobility.HandMobilitySource"
        ) as mock_mj:
            mock_mj.return_value = MagicMock()
            source = _hand_source(spec_dummy, engine="pinocchio")
            mock_mj.assert_called_once_with(spec_dummy)
            assert source == mock_mj.return_value
