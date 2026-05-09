"""Tests for plot_engine.contour and plot_engine.protocols (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.plot_engine.contour import correlation_matrix, scatter_to_grid
from src.shared.python.plot_engine.protocols import PlotConverter, PlotRenderer


class TestScatterToGrid:
    def _scatter_paraboloid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        n = 50
        x = rng.uniform(0, 10, n)
        y = rng.uniform(0, 10, n)
        z = x**2 + y**2
        return x, y, z

    def test_plot_engine_extras_returns_tuple_of_three(self) -> None:
        x, y, z = self._scatter_paraboloid()
        result = scatter_to_grid(x, y, z)
        assert len(result) == 3

    def test_x_grid_1d(self) -> None:
        x, y, z = self._scatter_paraboloid()
        x_g, y_g, z_g = scatter_to_grid(x, y, z, resolution=20)
        assert x_g.ndim == 1
        assert len(x_g) == 20

    def test_z_grid_2d(self) -> None:
        x, y, z = self._scatter_paraboloid()
        x_g, y_g, z_g = scatter_to_grid(x, y, z, resolution=20)
        assert z_g.ndim == 2
        assert z_g.shape == (20, 20)

    def test_too_few_points_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            scatter_to_grid(
                np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])
            )


class TestCorrelationMatrix:
    def test_plot_engine_extras_returns_tuple(self) -> None:
        data = np.random.default_rng(42).normal(0, 1, (20, 3))
        corr, labels = correlation_matrix(data)
        assert isinstance(corr, np.ndarray)
        assert isinstance(labels, list)

    def test_square_matrix(self) -> None:
        data = np.random.default_rng(42).normal(0, 1, (20, 4))
        corr, labels = correlation_matrix(data)
        assert corr.shape == (4, 4)

    def test_diagonal_ones(self) -> None:
        data = np.random.default_rng(42).normal(0, 1, (50, 3))
        corr, _ = correlation_matrix(data)
        np.testing.assert_allclose(np.diag(corr), np.ones(3), atol=1e-10)

    def test_auto_labels(self) -> None:
        data = np.random.default_rng(42).normal(0, 1, (20, 2))
        _, labels = correlation_matrix(data)
        assert len(labels) == 2

    def test_custom_labels(self) -> None:
        data = np.random.default_rng(42).normal(0, 1, (20, 2))
        _, labels = correlation_matrix(data, labels=["A", "B"])
        assert labels == ["A", "B"]

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError):
            correlation_matrix(np.array([1.0, 2.0, 3.0]))


class TestProtocols:
    def test_plot_renderer_protocol_exists(self) -> None:
        assert PlotRenderer is not None

    def test_plot_converter_protocol_exists(self) -> None:
        assert PlotConverter is not None
