from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_atlas import (
    DistributedAtlasConfig,
    _independent_engine_difference_detected,
    _project_stick_velocity,
)
from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
)
from scripts.research.proximal_distal_energy import articulated_forward_integration
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


pytestmark = pytest.mark.scientific


def test_distributed_atlas_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        DistributedAtlasConfig(station_counts=(1, 2, 3))
    with pytest.raises(ValueError, match="horizons_s"):
        DistributedAtlasConfig(horizons_s=(0.004, 0.025))
    with pytest.raises(ValueError, match="divisible"):
        DistributedAtlasConfig(
            forward=DistributedForwardConfig(
                duration_s=0.05,
                time_steps_s=(0.001, 0.0005),
            ),
            horizons_s=(0.0045, 0.01, 0.025, 0.05),
        )


def test_default_atlas_declares_nested_horizon_and_equal_total_stiffness() -> None:
    config = DistributedAtlasConfig()

    assert config.station_counts == (1, 3, 5)
    assert config.friction_coefficients == (0.0, 0.35)
    assert config.horizons_s == (0.004, 0.01, 0.025, 0.05)
    assert config.horizons_s[-1] == config.forward.duration_s
    assert config.total_stiffness_n_m == 1800.0
    with pytest.raises(ValueError, match="friction_coefficients"):
        DistributedAtlasConfig(friction_coefficients=(0.35,))


def test_mass_metric_stick_projection_has_an_analytic_solution() -> None:
    mass = np.diag([2.0, 3.0, 5.0])
    velocity = np.array([4.0, -2.0, 1.5])
    jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    projected, residual, capture_energy, impulse_norm = _project_stick_velocity(
        mass, velocity, jacobian
    )

    np.testing.assert_allclose(projected, [0.0, 0.0, 1.5], atol=1.0e-14)
    assert residual < 1.0e-14
    assert capture_energy == pytest.approx(22.0)
    assert impulse_norm == pytest.approx(np.hypot(8.0, 6.0))
    assert capture_energy == pytest.approx(
        0.5 * velocity @ mass @ velocity - 0.5 * projected @ mass @ projected
    )


def test_mass_metric_stick_projection_is_stable_for_redundant_constraints() -> None:
    """Rank-deficient station rows must not defeat the no-slip residual gate."""
    rng = np.random.default_rng(333)
    basis, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    mass = basis @ np.diag(np.geomspace(10**-3.5, 10**3.5, 20)) @ basis.T
    independent_rows = rng.normal(size=(10, 20))
    jacobian = np.vstack(
        (
            independent_rows,
            independent_rows[2],
            independent_rows[7],
        )
    )
    velocity = rng.normal(size=20)

    projected, residual, capture_energy, _ = _project_stick_velocity(
        mass, velocity, jacobian
    )

    assert np.linalg.matrix_rank(jacobian) == 10
    assert residual < 1e-11
    assert np.linalg.norm(jacobian @ projected, ord=np.inf) == pytest.approx(residual)
    assert capture_energy >= 0.0


def test_cross_engine_gate_rejects_an_identically_zero_comparison() -> None:
    zeros = np.zeros((2, 3))

    assert not _independent_engine_difference_detected(zeros, zeros, zeros)
    assert _independent_engine_difference_detected(zeros, zeros, np.array([0.0, 1e-15]))


def test_pinocchio_operator_rejects_an_impostor_without_mujoco_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pinocchio", SimpleNamespace(__version__="0.1"))

    with pytest.raises(RuntimeError, match="unrelated PyPI 'pinocchio'"):
        native_dynamics_operator("pinocchio", cast(SpatialModel, cast(Any, object())))


def test_pinocchio_operator_symmetrizes_crba_upper_triangle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinocchio leaves the CRBA lower triangle unspecified between calls."""

    class NativeModel:
        def createData(self) -> object:
            return object()

    class PinocchioModule:
        __version__ = "3.8.0"

        @staticmethod
        def crba(_model: object, _data: object, _q: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    [4.0, 1.5, -0.25],
                    [987.0, 3.0, 0.75],
                    [-654.0, 321.0, 2.0],
                ]
            )

        @staticmethod
        def nonLinearEffects(
            _model: object,
            _data: object,
            _q: np.ndarray,
            _qd: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(3)

    monkeypatch.setitem(sys.modules, "pinocchio", PinocchioModule())
    monkeypatch.setattr(
        articulated_forward_integration,
        "require_robotics_pinocchio",
        lambda _module: "3.8.0",
    )
    monkeypatch.setattr(
        articulated_forward_integration,
        "build_pinocchio_articulated_model",
        lambda _module, _model: NativeModel(),
    )

    operator = native_dynamics_operator(
        "pinocchio", cast(SpatialModel, cast(Any, object()))
    )
    mass, bias = operator(np.zeros(3), np.zeros(3))

    np.testing.assert_array_equal(
        mass,
        np.array(
            [
                [4.0, 1.5, -0.25],
                [1.5, 3.0, 0.75],
                [-0.25, 0.75, 2.0],
            ]
        ),
    )
    np.testing.assert_array_equal(bias, np.zeros(3))


def test_committed_distributed_atlas_is_complete_finite_and_qualified() -> None:
    summary = json.loads(
        (DATA / "articulated_distributed_grip_atlas.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == "articulated-distributed-grip-atlas/v3"
    assert summary["design"]["trajectory_count"] == 576
    assert summary["design"]["station_counts_per_hand"] == [1, 3, 5]
    assert summary["design"]["friction_coefficients"] == [0.0, 0.35]
    assert summary["design"]["engine_versions"]["pinocchio"] != "0.1"
    assert summary["design"]["horizons_s"] == [0.004, 0.01, 0.025, 0.05]
    assert summary["results"]["maximum_registered_event_transition_count"] > 0
    assert summary["results"]["registered_event_opening_count"] > 0
    assert summary["results"]["registered_event_reattachment_count"] > 0
    assert summary["results"]["event_active_set_parity_failures"] == 0
    assert summary["results"]["maximum_stick_projection_residual_m_s"] < 1e-10
    assert summary["results"]["maximum_stick_velocity_relative_error"] < 1e-7
    assert summary["results"]["stick_active_set_parity_failures"] == 0
    assert summary["results"]["failed_numerical_cell_count"] == 0
    assert summary["results"]["failed_parity_cell_count"] == 0
    assert summary["results"]["active_set_parity_failures"] == 0
    assert summary["results"]["time_refinement_passed"]
    assert summary["results"]["station_refinement_passed"]
    assert summary["results"]["independent_engine_difference_detected"]
    assert summary["results"]["all_registered_gates_passed"]

    with np.load(DATA / "articulated_distributed_grip_atlas.npz") as arrays:
        assert arrays["peak_station_force_n"].shape == (12, 3, 2, 2, 2, 2, 4)
        assert arrays["trajectory_relative_error"].shape == (12, 3, 2, 2, 2, 4)
        assert arrays["event_transition_count"].shape == (3, 2, 2, 2)
        assert np.any(arrays["event_opening_count"] > 0)
        assert np.any(arrays["event_reattachment_count"] > 0)
        assert np.all(arrays["event_active_set_parity"])
        assert arrays["stick_projection_residual_m_s"].shape == (12, 3, 2, 2)
        assert np.all(arrays["stick_projection_residual_m_s"] < 1e-10)
        assert np.all(arrays["stick_capture_energy_j"] >= 0.0)
        assert np.all(arrays["stick_active_set_parity"])
        assert np.all(np.isfinite(arrays["peak_station_force_n"]))
        assert np.all(np.isfinite(arrays["final_q"]))
        assert np.all(arrays["numerical_gates_passed"])
        assert np.all(arrays["parity_gates_passed"])
