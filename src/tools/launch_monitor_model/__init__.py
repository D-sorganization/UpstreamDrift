"""Vendor-neutral launch-monitor ingestion and analytics.

The package keeps PyQt6 and scikit-learn optional. Importing this façade needs
only the core scientific/data dependencies; the shallow MLP imports
scikit-learn lazily when selected.

ADR-0046 Stage 2 retires modules from here onto the canonical layer Tools
owns, imported as ``shared.python.launch_monitor.<module>``. That name only
resolves where ``vendor/ud-tools/src`` is on ``sys.path``. The test session
gets it from ``pyproject.toml``'s pytest ``pythonpath``; an installed wheel
gets it from ``build_hooks.py``, which force-includes the canonical ``shared``
package at the top level. A process started from a source checkout -- the API
server, the launcher, a companion workflow -- gets it from neither, so this
module puts it there before the first canonical import runs. See
:func:`_ensure_canonical_layer_importable`.
"""

import importlib
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_canonical_layer_importable() -> None:
    """Make ``shared.python.launch_monitor`` resolvable before it is imported.

    A no-op wherever the vendored Tools source is already reachable, which is
    every test run and every installed wheel. Where it is not, the resolution
    facade in :mod:`src.launchers.tools_repo_path` owns the precedence order
    (``TOOLS_REPO_PATH``, then the pinned ``vendor/ud-tools`` gitlink, then
    dev-mode sibling discovery) and puts the Tools ``src`` directory on
    ``sys.path``. Duplicating that precedence here would be a second answer to
    a question that already has one.

    Adding the path is not sufficient on its own. ``shared`` is a regular
    package, so its ``__path__`` is fixed at first import, and the probe above
    binds ``shared.python`` to UpstreamDrift's own ``src/shared/python`` on its
    way to discovering that ``launch_monitor`` is not there. The vendored
    directory is therefore *appended* to that ``__path__`` -- the same thing
    ``tests/conftest.py`` does for the test session, and the reason the
    canonical import resolves under pytest today. Appending rather than
    prepending keeps UpstreamDrift's own shared modules first; only names
    UpstreamDrift does not provide fall through, and ``launch_monitor`` is now
    exactly such a name.

    Raises:
        ModuleNotFoundError: If no Tools checkout resolves. Failing at import
            is deliberate: the modules ADR-0046 Stage 2 retired are not served
            by UpstreamDrift any more, so a façade that imported without them
            would be a façade over nothing.
    """
    try:
        importlib.import_module("shared.python.launch_monitor")
        return
    except ImportError:
        pass

    # Deferred: only a source-checkout process reaches this branch, and the
    # canonical resolution facade lives in the launcher package.
    from src.launchers.tools_repo_path import ensure_tools_importable

    resolution = ensure_tools_importable(
        _REPO_ROOT,
        os.environ.get("TOOLS_REPO_PATH"),
        error_cls=ModuleNotFoundError,
    )

    vendored = str(resolution.path / "src" / "shared" / "python")
    shared_python = sys.modules.get("shared.python")
    search_path = getattr(shared_python, "__path__", None)
    if search_path is not None and vendored not in search_path:
        search_path.append(vendored)

    importlib.import_module("shared.python.launch_monitor")


_ensure_canonical_layer_importable()

from shared.python.launch_monitor.comparison import (
    MonitorComparisonResult,
    MonitorSummary,
    PairwiseMonitorComparison,
    compare_monitors,
)
from src.tools.launch_monitor_model.conformance_bundle import (
    LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
    LaunchMonitorConformanceBundleV1,
    LaunchMonitorConformanceScenarioV1,
    launch_monitor_conformance_bundle_json_schema,
    launch_monitor_conformance_bundle_sha256,
    launch_monitor_conformance_scenario_sha256,
)
from src.tools.launch_monitor_model.corpus import (
    CORPUS_COLUMN_MAP,
    corpus_dataset_path,
    load_private_corpus,
)
from src.tools.launch_monitor_model.contract_v2 import (
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
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
    SourceFileReferenceV2,
    TransformRecordV2,
    UncertaintyV2,
    VendorProvenanceV2,
    adapt_v2_to_v1,
    analyze_variables_v2,
    build_analysis_lineage_v2,
    contract_v2_json_schema,
)
from shared.python.launch_monitor.dispersion import (
    DispersionResult,
    analyze_dispersion,
)
from src.tools.launch_monitor_model.dataset_reference import (
    DATASET_JOB_CONTRACT_VERSION,
    DatasetJobRequestV1,
    DatasetOperationV1,
    DatasetReferenceV1,
    DatasetUnavailableStateV1,
    dataset_content_sha256,
    dataset_job_contract_json_schema,
)
from src.tools.launch_monitor_model.flexible_analysis import (
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
from src.tools.launch_monitor_model.importer import import_session
from src.tools.launch_monitor_model.longitudinal import (
    analyze_longitudinal_sessions,
    longitudinal_session_contract_json_schema,
)
from src.tools.launch_monitor_model.longitudinal_types import (
    LONGITUDINAL_SESSION_CONTRACT_VERSION,
    LongitudinalClaimsV1,
    LongitudinalDesignV1,
    LongitudinalMissingnessV1,
    LongitudinalPlayerAssociationV1,
    LongitudinalSessionRequestV1,
    LongitudinalSessionResultV1,
    PooledAssociationV1,
    SessionAggregateV1,
)
from src.tools.launch_monitor_model.modeling import (
    PredictiveModelResult,
    fit_predictive_model,
)
from shared.python.launch_monitor.multivariate import (
    PCAResult,
    VIFResult,
    compute_pca,
    compute_vif,
)
from src.tools.launch_monitor_model.outcome_proxy import analyze_outcome_proxy
from src.tools.launch_monitor_model.player_covariation import (
    analyze_player_covariation_v1,
    player_covariation_contract_json_schema,
    scan_player_covariation_v1,
)
from src.tools.launch_monitor_model.player_covariation_types import (
    PLAYER_COVARIATION_CONTRACT_VERSION,
    AssociationEstimateV1,
    CovariationMissingnessV1,
    CovariationPairRankV1,
    CovariationUncertaintyV1,
    MetaAnalysisSummaryV1,
    PlayerAssociationV1,
    PlayerCovariationContractV1,
    PlayerCovariationRequestV1,
    PlayerCovariationResultV1,
    PlayerCovariationScanRequestV1,
    PlayerCovariationScanResultV1,
)
from src.tools.launch_monitor_model.profiles import (
    PROFILES,
    ImportProfile,
    ProfileDetection,
    detect_profile,
    normalize_header,
)
from src.tools.launch_monitor_model.project import LaunchMonitorProject
from src.tools.launch_monitor_model.relationships import (
    CorrelationResult,
    DependencyEdge,
    compute_correlations,
)
from shared.python.launch_monitor.schema import (
    IDENTITY_COLUMNS,
    METRICS,
    ColumnMapping,
    ImportedSession,
    ImportManifest,
    ImportOptions,
    MetricDefinition,
    numeric_metric_columns,
)
from shared.python.launch_monitor.trends import (
    ChangeCandidate,
    TemporalTrendResult,
    analyze_trend,
)
from src.tools.launch_monitor_model.strokes_gained import (
    analyze_source_backed_strokes_gained,
    strokes_gained_contract_json_schema,
)
from src.tools.launch_monitor_model.strokes_gained_types import (
    BASELINE_CONTRACT_VERSION,
    OUTCOME_PROXY_CONTRACT_VERSION,
    STROKES_GAINED_CONTRACT_VERSION,
    CourseStateColumnsV1,
    ExpectedStrokesBaselineV2,
    ExpectedStrokesStateV2,
    GroupingDimensionV1,
    LongitudinalDimensionV1,
    OutcomeProxyRequestV1,
    OutcomeProxyResultV1,
    StrokesGainedAnalysisResultV1,
    StrokesGainedRequestV1,
    baseline_table_sha256,
)
from shared.python.launch_monitor.treatment import (
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
    "AssociationEstimateV1",
    "AvailabilityV2",
    "BASELINE_CONTRACT_VERSION",
    "BackingRecordV2",
    "ClaimsV2",
    "CoefficientEstimate",
    "ColumnMapping",
    "CorrelationEstimate",
    "CorrelationMethod",
    "CorrelationResult",
    "CourseStateColumnsV1",
    "CovariationMissingnessV1",
    "CovariationPairRankV1",
    "CovariationUncertaintyV1",
    "DatasetSummary",
    "DATASET_JOB_CONTRACT_VERSION",
    "DatasetJobRequestV1",
    "DatasetOperationV1",
    "DatasetReferenceV1",
    "DatasetUnavailableStateV1",
    "DatasetAuthorityV2",
    "DependencyEdge",
    "DispersionResult",
    "FlexibleAnalysisRequest",
    "FlexibleAnalysisResult",
    "ExpectedStrokesBaselineV2",
    "ExpectedStrokesStateV2",
    "GroupingDimensionV1",
    "GroupAnalysis",
    "ImportManifest",
    "ImportOptions",
    "ImportProfile",
    "ImportedSession",
    "LaunchMonitorProject",
    "LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION",
    "LaunchMonitorConformanceBundleV1",
    "LaunchMonitorConformanceScenarioV1",
    "LONGITUDINAL_SESSION_CONTRACT_VERSION",
    "LongitudinalDimensionV1",
    "LongitudinalClaimsV1",
    "LongitudinalDesignV1",
    "LongitudinalMissingnessV1",
    "LongitudinalPlayerAssociationV1",
    "LongitudinalSessionRequestV1",
    "LongitudinalSessionResultV1",
    "LaunchMonitorAnalysisResultV2",
    "MetricDefinition",
    "MetricUnitsV2",
    "MonitorComparisonResult",
    "MonitorSummary",
    "MetaAnalysisSummaryV1",
    "ModelProvenanceV2",
    "OrderEvidenceV2",
    "MissingPolicy",
    "OUTCOME_PROXY_CONTRACT_VERSION",
    "OutcomeProxyRequestV1",
    "OutcomeProxyResultV1",
    "PairwiseMonitorComparison",
    "PredictiveModelResult",
    "PCAResult",
    "ProfileDetection",
    "PLAYER_COVARIATION_CONTRACT_VERSION",
    "PlayerAssociationV1",
    "PlayerCovariationContractV1",
    "PlayerCovariationRequestV1",
    "PlayerCovariationResultV1",
    "PlayerCovariationScanRequestV1",
    "PlayerCovariationScanResultV1",
    "PlayerIdentityV2",
    "PooledAssociationV1",
    "SessionIdentityV2",
    "SourceFileReferenceV2",
    "RegressionEstimate",
    "ResidualDiagnostics",
    "STROKES_GAINED_CONTRACT_VERSION",
    "StrokesGainedAnalysisResultV1",
    "StrokesGainedRequestV1",
    "SessionAggregateV1",
    "TreatmentConfig",
    "TreatmentResult",
    "TransformRecordV2",
    "FilterRule",
    "TemporalTrendResult",
    "VIFResult",
    "UncertaintyV2",
    "VendorProvenanceV2",
    "adapt_v2_to_v1",
    "analyze_outcome_proxy",
    "analyze_longitudinal_sessions",
    "analyze_player_covariation_v1",
    "analyze_source_backed_strokes_gained",
    "analyze_dispersion",
    "analyze_variables",
    "analyze_variables_v2",
    "analyze_trend",
    "apply_treatment",
    "baseline_table_sha256",
    "build_analysis_lineage_v2",
    "compare_monitors",
    "CORPUS_COLUMN_MAP",
    "corpus_dataset_path",
    "compute_correlations",
    "compute_pca",
    "compute_vif",
    "contract_v2_json_schema",
    "detect_profile",
    "dataset_content_sha256",
    "dataset_job_contract_json_schema",
    "fit_predictive_model",
    "import_session",
    "load_private_corpus",
    "launch_monitor_conformance_bundle_json_schema",
    "launch_monitor_conformance_bundle_sha256",
    "launch_monitor_conformance_scenario_sha256",
    "longitudinal_session_contract_json_schema",
    "normalize_header",
    "numeric_metric_columns",
    "player_covariation_contract_json_schema",
    "scan_player_covariation_v1",
    "strokes_gained_contract_json_schema",
]
