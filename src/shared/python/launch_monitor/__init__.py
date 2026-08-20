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
from src.shared.python.launch_monitor.corpus import (
    CORPUS_COLUMN_MAP,
    corpus_dataset_path,
    load_private_corpus,
)
from src.shared.python.launch_monitor.contract_v2 import (
    CONTRACT_VERSION_V2,
    AnalysisContextV2,
    AnalysisLineageV2,
    AvailabilityV2,
    BackingRecordV2,
    ClaimsV2,
    DatasetAuthorityV2,
    LaunchMonitorAnalysisResultV2,
    MetricUnitsV2,
    ModelProvenanceV2,
    PlayerIdentityV2,
    SourceFileReferenceV2,
    TransformRecordV2,
    UncertaintyV2,
    VendorProvenanceV2,
    adapt_v2_to_v1,
    analyze_variables_v2,
    contract_v2_json_schema,
)
from src.shared.python.launch_monitor.dispersion import (
    DispersionResult,
    analyze_dispersion,
)
from src.shared.python.launch_monitor.flexible_analysis import (
    CONTRACT_VERSION,
    AnalysisMode,
    CoefficientEstimate,
    CorrelationMethod,
    CorrelationEstimate,
    DatasetSummary,
    FlexibleAnalysisRequest,
    FlexibleAnalysisResult,
    GroupAnalysis,
    MissingPolicy,
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
    "AnalysisContextV2",
    "AnalysisLineageV2",
    "ChangeCandidate",
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_V2",
    "AnalysisMode",
    "AvailabilityV2",
    "BackingRecordV2",
    "ClaimsV2",
    "CoefficientEstimate",
    "ColumnMapping",
    "CorrelationEstimate",
    "CorrelationMethod",
    "CorrelationResult",
    "DatasetSummary",
    "DatasetAuthorityV2",
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
    "LaunchMonitorAnalysisResultV2",
    "MetricDefinition",
    "MetricUnitsV2",
    "MonitorComparisonResult",
    "MonitorSummary",
    "ModelProvenanceV2",
    "MissingPolicy",
    "PairwiseMonitorComparison",
    "PredictiveModelResult",
    "PCAResult",
    "ProfileDetection",
    "PlayerIdentityV2",
    "SourceFileReferenceV2",
    "RegressionEstimate",
    "ResidualDiagnostics",
    "TreatmentConfig",
    "TreatmentResult",
    "TransformRecordV2",
    "FilterRule",
    "TrendResult",
    "VIFResult",
    "UncertaintyV2",
    "VendorProvenanceV2",
    "adapt_v2_to_v1",
    "analyze_dispersion",
    "analyze_variables",
    "analyze_variables_v2",
    "analyze_trend",
    "apply_treatment",
    "compare_monitors",
    "CORPUS_COLUMN_MAP",
    "corpus_dataset_path",
    "compute_correlations",
    "compute_pca",
    "compute_vif",
    "contract_v2_json_schema",
    "detect_profile",
    "fit_predictive_model",
    "import_session",
    "load_private_corpus",
    "normalize_header",
    "numeric_metric_columns",
]
