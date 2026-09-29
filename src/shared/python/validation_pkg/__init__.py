"""Physics validation, statistical analysis, data fitting, and workflow diagnostics."""

from .comparative_analysis import (
    AlignedSignals,
    ComparativeSwingAnalyzer,
    ComparisonMetric,
)
from .comparative_plotting import ComparativePlotter
from .data_fitting import (
    A3FittingPipeline,
    BodySegmentParams,
    FitResult,
    InverseKinematicsSolver,
    KinematicState,
    ParameterEstimationReport,
    ParameterEstimator,
    SensitivityAnalyzer,
    SensitivityResult,
    convert_poses_to_markers,
)
from .kaggle_validation import (
    ShotRecord,
    compare_all_models_to_dataset,
    get_clean_shots,
    get_dataset_statistics,
    load_kaggle_dataset,
    print_validation_report,
    validate_model_against_dataset,
)
from .statistical_analysis import StatisticalAnalyzer
from .validation import (
    PhysicalValidationError,
    validate_friction_coefficient,
    validate_inertia_matrix,
    validate_joint_limits,
    validate_mass,
    validate_physical_bounds,
    validate_timestep,
)
from .validation_data import (
    DataSource,
    ValidationDataPoint,
    get_validation_data_for_club,
    print_validation_summary,
)
from .validation_helpers import (
    PhysicsValidationError,
    ValidationLevel,
    validate_cartesian_state,
    validate_finite,
    validate_joint_state,
    validate_magnitude,
    validate_model_parameters,
)
from .validation_utils import (
    validate_all,
    validate_array_dimensions,
    validate_array_length,
    validate_array_shape,
    validate_directory_exists,
    validate_extension,
    validate_file_exists,
    validate_not_none,
    validate_numeric,
    validate_positive,
    validate_range,
    validate_type,
)
from .workflow_diagnostics import WorkflowDiagnosticContext

__all__: list[str] = [
    # comparative_analysis
    "AlignedSignals",
    "ComparativeSwingAnalyzer",
    "ComparisonMetric",
    # comparative_plotting
    "ComparativePlotter",
    # data_fitting
    "A3FittingPipeline",
    "BodySegmentParams",
    "FitResult",
    "InverseKinematicsSolver",
    "KinematicState",
    "ParameterEstimationReport",
    "ParameterEstimator",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "convert_poses_to_markers",
    # kaggle_validation
    "ShotRecord",
    "compare_all_models_to_dataset",
    "get_clean_shots",
    "get_dataset_statistics",
    "load_kaggle_dataset",
    "print_validation_report",
    "validate_model_against_dataset",
    # statistical_analysis
    "StatisticalAnalyzer",
    # validation
    "PhysicalValidationError",
    "validate_friction_coefficient",
    "validate_inertia_matrix",
    "validate_joint_limits",
    "validate_mass",
    "validate_physical_bounds",
    "validate_timestep",
    # validation_data
    "DataSource",
    "ValidationDataPoint",
    "get_validation_data_for_club",
    "print_validation_summary",
    # validation_helpers
    "PhysicsValidationError",
    "ValidationLevel",
    "validate_cartesian_state",
    "validate_finite",
    "validate_joint_state",
    "validate_magnitude",
    "validate_model_parameters",
    # validation_utils
    "validate_all",
    "validate_array_dimensions",
    "validate_array_length",
    "validate_array_shape",
    "validate_directory_exists",
    "validate_extension",
    "validate_file_exists",
    "validate_not_none",
    "validate_numeric",
    "validate_positive",
    "validate_range",
    "validate_type",
    # workflow_diagnostics
    "WorkflowDiagnosticContext",
]
