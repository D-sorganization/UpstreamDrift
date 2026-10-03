"""Tests for deprecated plotting renderer shims (ADR-0052, #11292)."""

from __future__ import annotations

import importlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure


def test_force_vectors_shim_deprecation_warning() -> None:
    if "src.shared.python.plotting.renderers.force_vectors" in sys.modules:
        del sys.modules["src.shared.python.plotting.renderers.force_vectors"]

    with pytest.deprecated_call():
        importlib.import_module("src.shared.python.plotting.renderers.force_vectors")


def test_vectors_shim_deprecation_warning() -> None:
    if "src.shared.python.plotting.renderers.vectors" in sys.modules:
        del sys.modules["src.shared.python.plotting.renderers.vectors"]

    with pytest.deprecated_call():
        importlib.import_module("src.shared.python.plotting.renderers.vectors")


from unittest.mock import MagicMock


def test_force_vectors_public_methods() -> None:
    from src.shared.python.plotting.renderers.force_vectors import (
        ForceVectorRenderer,
    )

    renderer = ForceVectorRenderer(MagicMock())
    fig = Figure()
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    forces = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])

    renderer.plot_joint_force_vectors(fig, positions=pos, forces=forces)
    assert len(fig.get_axes()) == 1

    fig_ztcf = Figure()
    renderer.plot_ztcf_force_vectors(fig_ztcf, positions=pos, ztcf_forces=forces)
    assert len(fig_ztcf.get_axes()) == 1


def test_vectors_public_methods() -> None:
    from src.shared.python.plotting.renderers.vectors import (
        VectorOverlayRenderer,
    )

    renderer = VectorOverlayRenderer(MagicMock())
    fig = Figure()
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    forces = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])

    renderer.plot_contact_force_vectors(fig, positions=pos, forces=forces)
    assert len(fig.get_axes()) == 1

    fig_torque = Figure()
    renderer.plot_joint_torque_vectors(
        fig_torque, joint_positions=pos, torque_magnitudes=np.array([5.0, 10.0])
    )
    assert len(fig_torque.get_axes()) == 1
