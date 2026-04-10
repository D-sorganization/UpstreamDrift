"""Regression coverage for the data_fitting facade split."""

from src.shared.python.validation_pkg.data_fitting import (
    A3FittingPipeline,
    BodySegmentParams,
    InverseKinematicsSolver,
    ParameterEstimator,
    SensitivityAnalyzer,
    convert_poses_to_markers,
)


def test_facade_reexports_core_types() -> None:
    """The legacy facade should keep exposing the public A3 API."""
    assert BodySegmentParams.__name__ == "BodySegmentParams"
    assert InverseKinematicsSolver.__name__ == "InverseKinematicsSolver"
    assert ParameterEstimator.__name__ == "ParameterEstimator"
    assert SensitivityAnalyzer.__name__ == "SensitivityAnalyzer"
    assert A3FittingPipeline.__name__ == "A3FittingPipeline"
    assert callable(convert_poses_to_markers)
