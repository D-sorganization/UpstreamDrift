from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.simscape import SimscapeAdapter, SimscapeStateError
from src.shared.python.core.contracts.exceptions import PreconditionError
from src.shared.python.engine_core.interfaces import PhysicsEngine

pytestmark = pytest.mark.unit


def test_simscape_adapter_satisfies_physics_engine_protocol() -> None:
    adapter = SimscapeAdapter()
    assert isinstance(adapter, PhysicsEngine)
    assert hasattr(adapter, "compute_gravity_only_acceleration")
    assert hasattr(adapter, "compute_control_acceleration_at_state")


def test_simscape_counterfactual_methods_have_precondition_guards() -> None:
    """Guards must be on the class methods, not only on delegated defaults (#14943)."""
    for name in (
        "compute_gravity_only_acceleration",
        "compute_control_acceleration_at_state",
    ):
        method = getattr(SimscapeAdapter, name)
        assert getattr(method, "__wrapped__", None) is not None or hasattr(
            method, "__preconditions__"
        )


@pytest.mark.parametrize(
    ("q", "tau"),
    [
        (np.array([0.1, 0.2]), None),
        (np.array([[0.1, 0.2]]), None),
        (np.zeros(0), None),
    ],
    ids=["valid_1d", "non_1d_q", "empty_1d_q"],
)
def test_compute_gravity_only_acceleration_input_guards(
    q: np.ndarray, tau: np.ndarray | None
) -> None:
    adapter = SimscapeAdapter()
    try:
        if q.ndim != 1:
            with pytest.raises(PreconditionError):
                adapter.compute_gravity_only_acceleration(q)
            return
        with pytest.raises(SimscapeStateError):
            adapter.compute_gravity_only_acceleration(q)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("q", "tau", "expect"),
    [
        (np.array([0.1, 0.2]), np.array([1.0, 2.0]), SimscapeStateError),
        (np.zeros((2, 2)), np.array([1.0, 2.0]), PreconditionError),
        (np.array([0.1, 0.2]), np.zeros((2, 2)), PreconditionError),
        (np.array([0.1, 0.2]), np.zeros(0), SimscapeStateError),
    ],
    ids=[
        "unloaded_lifecycle",
        "non_1d_q",
        "non_1d_tau",
        "empty_tau",
    ],
)
def test_compute_control_acceleration_at_state_input_guards(
    q: np.ndarray,
    tau: np.ndarray,
    expect: type[Exception],
) -> None:
    adapter = SimscapeAdapter()
    try:
        with pytest.raises(expect):
            adapter.compute_control_acceleration_at_state(q, tau)
    finally:
        adapter.close()


def test_counterfactual_rejects_stopped_adapter(slx_pair: Path) -> None:
    adapter = SimscapeAdapter()
    adapter.load_from_path(str(slx_pair))
    dof = adapter.dof
    q = np.zeros(dof)
    tau = np.zeros(dof)
    adapter.close()
    assert not adapter.model_loaded
    with pytest.raises(SimscapeStateError):
        adapter.compute_gravity_only_acceleration(q)
    with pytest.raises(SimscapeStateError):
        adapter.compute_control_acceleration_at_state(q, tau)


@pytest.fixture
def slx_pair(tmp_path: Path) -> Path:
    slx = tmp_path / "GolfSwing3D_Kinetic.slx"
    slx.write_bytes(b"PK\x03\x04")
    (tmp_path / "PolynomialInputValues.mat").write_bytes(b"\x00")
    return slx


def test_counterfactual_rejects_wrong_length_vectors(slx_pair: Path) -> None:
    adapter = SimscapeAdapter()
    adapter.load_from_path(str(slx_pair))
    dof = adapter.dof
    short_q = np.zeros(dof - 1, dtype=float)
    tau = np.zeros(dof, dtype=float)
    try:
        with pytest.raises(ValueError, match=r"shape \(16,\)"):
            adapter.compute_gravity_only_acceleration(short_q)
        with pytest.raises(ValueError, match=r"shape \(16,\)"):
            adapter.compute_control_acceleration_at_state(short_q, tau)
    finally:
        adapter.close()
