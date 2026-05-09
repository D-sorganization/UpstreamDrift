"""Tests for src.shared.python.analysis.pca_analysis (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.analysis.pca_analysis import PCAAnalysisMixin


class _Concrete(PCAAnalysisMixin):
    def __init__(self, n: int = 100, n_joints: int = 6) -> None:
        rng = np.random.default_rng(42)
        self.times = np.linspace(0.0, 1.0, n)
        # Create correlated joint motion (dominated by a few modes)
        t = self.times
        base = np.column_stack(
            [
                np.sin(2 * np.pi * t),
                np.cos(2 * np.pi * t),
                0.5 * np.sin(4 * np.pi * t),
                np.zeros(n),
                0.3 * np.cos(4 * np.pi * t),
                rng.standard_normal(n) * 0.01,
            ]
        )
        self.joint_positions = base
        self.joint_velocities = np.gradient(base, t[1] - t[0], axis=0)


class TestPCAAnalysisMixin:
    def setup_method(self) -> None:
        self.obj = _Concrete(n=100, n_joints=6)

    def test_returns_pca_result(self) -> None:
        result = self.obj.compute_principal_component_analysis()
        assert result is not None

    def test_components_shape(self) -> None:
        result = self.obj.compute_principal_component_analysis()
        # components should be (n_components, n_features)
        assert result.components.ndim == 2
        assert result.components.shape[1] == 6

    def test_explained_variance_non_negative(self) -> None:
        result = self.obj.compute_principal_component_analysis()
        assert np.all(result.explained_variance >= 0.0)

    def test_explained_variance_ratio_sums_to_one(self) -> None:
        result = self.obj.compute_principal_component_analysis()
        assert np.sum(result.explained_variance_ratio) == pytest.approx(1.0, abs=1e-8)

    def test_explained_variance_ratio_all_positive(self) -> None:
        result = self.obj.compute_principal_component_analysis()
        assert np.all(result.explained_variance_ratio >= 0.0)

    def test_n_components_truncates_result(self) -> None:
        result = self.obj.compute_principal_component_analysis(n_components=3)
        assert result.components.shape[0] == 3
        assert len(result.explained_variance) == 3

    def test_n_components_zero_raises(self) -> None:
        from src.shared.python.core.contracts.exceptions import PreconditionError

        with pytest.raises(PreconditionError):
            self.obj.compute_principal_component_analysis(n_components=0)

    def test_velocity_data_type(self) -> None:
        result = self.obj.compute_principal_component_analysis(data_type="velocity")
        assert result is not None

    def test_pca_analysis_all_values_finite(self) -> None:
        result = self.obj.compute_principal_component_analysis()
        assert np.all(np.isfinite(result.components))
        assert np.all(np.isfinite(result.explained_variance))
        assert np.all(np.isfinite(result.explained_variance_ratio))

    def test_first_component_captures_most_variance(self) -> None:
        result = self.obj.compute_principal_component_analysis()
        # First component should have the largest explained variance
        ratios = result.explained_variance_ratio
        assert ratios[0] >= ratios[1]

    def test_returns_none_for_empty_data(self) -> None:
        obj = _Concrete(n=10, n_joints=6)
        obj.joint_positions = np.zeros((0, 6))
        result = obj.compute_principal_component_analysis()
        assert result is None
