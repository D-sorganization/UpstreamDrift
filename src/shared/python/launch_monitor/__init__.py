"""Vendor-neutral launch-monitor ingestion and analytics.

The package keeps PyQt6 and scikit-learn optional. Importing this façade needs
only the core scientific/data dependencies; the shallow MLP imports
scikit-learn lazily when selected.
"""

from src.shared.python.launch_monitor.comparison import (
    MonitorComparisonResult,
    MonitorSummary,
    PairwiseMonitorComparison,
    compare_monitors,
)
from src.shared.python.launch_monitor.dispersion import (
    DispersionResult,
    analyze_dispersion,
)
from src.shared.python.launch_monitor.flexible_analysis import (
    CoefficientEstimate,
    CorrelationEstimate,
    DatasetSummary,
    FlexibleAnalysisRequest,
    FlexibleAnalysisResult,
    GroupAnalysis,
    RegressionEstimate,
    ResidualDiagnostics,
    analyze_variables,
)
from src.shared.python.launch_monitor.importer import import_session
from src.shared.python.launch_monitor.modeling import (
    PredictiveModelResult,
    fit_predictive_model,
)
from src.shared.python.launch_monitor.multivariate import (
    PCAResult,
    VIFResult,
    compute_pca,
    compute_vif,
)
from src.shared.python.launch_monitor.profiles import (
    PROFILES,
    ImportProfile,
    ProfileDetection,
    detect_profile,
    normalize_header,
)
from src.shared.python.launch_monitor.project import LaunchMonitorProject
from src.shared.python.launch_monitor.relationships import (
    CorrelationResult,
    DependencyEdge,
    compute_correlations,
)
from src.shared.python.launch_monitor.schema import (
    IDENTITY_COLUMNS,
    METRICS,
    ColumnMapping,
    ImportedSession,
    ImportManifest,
    ImportOptions,
    MetricDefinition,
    numeric_metric_columns,
)
from src.shared.python.launch_monitor.trends import (
    ChangeCandidate,
    TrendResult,
    analyze_trend,
)
from src.shared.python.launch_monitor.treatment import (
    TreatmentConfig,
    TreatmentResult,
    FilterRule,
    apply_treatment,
)

__all__ = [
    "IDENTITY_COLUMNS",
    "METRICS",
    "PROFILES",
    "ChangeCandidate",
    "CoefficientEstimate",
    "ColumnMapping",
    "CorrelationEstimate",
    "CorrelationResult",
    "DatasetSummary",
    "DependencyEdge",
    "DispersionResult",
    "FlexibleAnalysisRequest",
    "FlexibleAnalysisResult",
    "GroupAnalysis",
    "ImportManifest",
    "ImportOptions",
    "ImportProfile",
    "ImportedSession",
    "LaunchMonitorProject",
    "MetricDefinition",
    "MonitorComparisonResult",
    "MonitorSummary",
    "PairwiseMonitorComparison",
    "PredictiveModelResult",
    "PCAResult",
    "ProfileDetection",
    "RegressionEstimate",
    "ResidualDiagnostics",
    "TreatmentConfig",
    "TreatmentResult",
    "FilterRule",
    "TrendResult",
    "VIFResult",
    "analyze_dispersion",
    "analyze_variables",
    "analyze_trend",
    "apply_treatment",
    "compare_monitors",
    "compute_correlations",
    "compute_pca",
    "compute_vif",
    "detect_profile",
    "fit_predictive_model",
    "import_session",
    "normalize_header",
    "numeric_metric_columns",
]
