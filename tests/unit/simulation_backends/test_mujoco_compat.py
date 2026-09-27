"""Unit tests for the MuJoCo full_mass_matrix compatibility helper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.simulation_backends import has_mujoco
from src.shared.python.simulation_backends.mujoco_compat import full_mass_matrix

pytestmark = pytest.mark.unit

_skip_no_mujoco = pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_full_mass_matrix_one_hinge_model() -> None:
    """The helper returns the analytic hinge inertia on the installed MuJoCo."""
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="body1">
          <joint name="hinge1" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    M = full_mass_matrix(mujoco, model, data)

    # Solid sphere spinning about a diameter: I = 2/5 m r^2 = 0.4 * 1.0 * 0.1^2.
    np.testing.assert_allclose(M, [[0.004]], rtol=1e-12)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_full_mass_matrix_one_hinge_model_preallocated_dst() -> None:
    """Helper populates preallocated dst in place."""
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="body1">
          <joint name="hinge1" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dst = np.zeros((model.nv, model.nv), dtype=float)
    out = full_mass_matrix(mujoco, model, data, dst=dst)

    assert out is dst
    assert np.all(np.isfinite(dst))
    assert dst[0, 0] > 0.0


def test_full_mass_matrix_mock_with_qm() -> None:
    """Exercise old MuJoCo branch (< 3.13) where data has qM."""
    fake_mj = MagicMock()

    def fake_fullM(model, dst, qM):
        dst[:] = np.eye(model.nv) * 2.5

    fake_mj.mj_fullM.side_effect = fake_fullM

    model = SimpleNamespace(nv=3)
    data = SimpleNamespace(qM=np.array([1.0, 2.0, 3.0]))

    M = full_mass_matrix(fake_mj, model, data)

    assert M.shape == (3, 3)
    np.testing.assert_allclose(M, np.eye(3) * 2.5)
    fake_mj.mj_fullM.assert_called_once()
    call_args = fake_mj.mj_fullM.call_args[0]
    assert call_args[0] is model
    assert isinstance(call_args[1], np.ndarray)
    assert call_args[2] is data.qM


def test_full_mass_matrix_mock_without_qm() -> None:
    """Exercise new MuJoCo branch (>= 3.13) where data has no qM."""
    fake_mj = MagicMock()

    def fake_fullM(model, data, dst):
        dst[:] = np.eye(model.nv) * 3.5

    fake_mj.mj_fullM.side_effect = fake_fullM

    model = SimpleNamespace(nv=2)
    data = SimpleNamespace()  # no qM attribute

    M = full_mass_matrix(fake_mj, model, data)

    assert M.shape == (2, 2)
    np.testing.assert_allclose(M, np.eye(2) * 3.5)
    fake_mj.mj_fullM.assert_called_once()
    call_args = fake_mj.mj_fullM.call_args[0]
    assert call_args[0] is model
    assert call_args[1] is data
    assert isinstance(call_args[2], np.ndarray)


def test_full_mass_matrix_precondition_negative_nv() -> None:
    """Precondition: model.nv >= 0."""
    fake_mj = MagicMock()
    model = SimpleNamespace(nv=-1)
    data = SimpleNamespace()

    with pytest.raises(ValueError, match="nv must be >= 0"):
        full_mass_matrix(fake_mj, model, data)


def test_full_mass_matrix_precondition_missing_nv() -> None:
    """Precondition: model must have nv attribute."""
    fake_mj = MagicMock()
    model = SimpleNamespace()
    data = SimpleNamespace()

    with pytest.raises(ValueError, match="model must have 'nv' attribute"):
        full_mass_matrix(fake_mj, model, data)


def test_full_mass_matrix_precondition_dst_shape_mismatch() -> None:
    """Precondition: dst shape must match (nv, nv)."""
    fake_mj = MagicMock()
    model = SimpleNamespace(nv=2)
    data = SimpleNamespace()
    bad_dst = np.zeros((3, 3))

    with pytest.raises(ValueError, match="dst must have shape"):
        full_mass_matrix(fake_mj, model, data, dst=bad_dst)
