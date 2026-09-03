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
_CANONICAL_RELATIVE_PATH = Path("src/shared/python/launch_monitor/__init__.py")


def _bind_canonical_search_path(tools_root: Path) -> None:
    """Point the ``shared.python`` package at *tools_root*'s canonical layer.

    Putting the directory on ``sys.path`` is not enough on its own. ``shared``
    is a regular package, so its ``__path__`` is fixed at first import, and the
    probe in :func:`_ensure_canonical_layer_importable` binds ``shared.python``
    to UpstreamDrift's own ``src/shared/python`` on its way to discovering that
    ``launch_monitor`` is not there. The vendored directory is therefore
    *appended* to that ``__path__`` -- the same thing ``tests/conftest.py``
    does for the test session, and the reason the canonical import resolves
    under pytest today. Appending rather than prepending keeps UpstreamDrift's
    own shared modules first; only names UpstreamDrift does not provide fall
    through, and ``launch_monitor`` is now exactly such a name.
    """
    src_dir = str(tools_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    vendored = str(tools_root / "src" / "shared" / "python")
    search_path = getattr(sys.modules.get("shared.python"), "__path__", None)
    if search_path is not None and vendored not in search_path:
        search_path.append(vendored)


def _ensure_canonical_layer_importable() -> None:
    """Make ``shared.python.launch_monitor`` resolvable before it is imported.

    A no-op wherever the vendored Tools source is already reachable, which is
    every test run and every installed wheel.

    Where it is not, the pinned ``vendor/ud-tools`` tree is used directly when
    its files are present. That deliberately skips the git-level validation in
    :func:`~src.launchers.tools_repo_path.require_tools_repo`, because several
    CI lanes materialise the pinned Tools source as a plain checkout rather
    than a real submodule, which that validation rejects even though the files
    are present and importable. The same "already-present wins, validate only
    when searching" shape is what
    ``src/api/routes/_ball_flight_trajectory_import.py`` uses, and for the same
    reason.

    Only when the vendored tree is genuinely absent does the resolution facade
    take over: it owns ``TOOLS_REPO_PATH`` and dev-mode sibling discovery, and
    duplicating that precedence here would be a second answer to a question
    that already has one.

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

    vendor_root = _REPO_ROOT / "vendor" / "ud-tools"
    if (vendor_root / _CANONICAL_RELATIVE_PATH).is_file():
        _bind_canonical_search_path(vendor_root)
    else:
        # Deferred: only a source checkout with no vendored tree reaches this,
        # and the canonical resolution facade lives in the launcher package.
        from src.launchers.tools_repo_path import require_tools_repo

        try:
            resolution = require_tools_repo(
                _REPO_ROOT, os.environ.get("TOOLS_REPO_PATH")
            )
        except RuntimeError as exc:
            raise ModuleNotFoundError(str(exc)) from exc
        _bind_canonical_search_path(resolution.path)

    importlib.import_module("shared.python.launch_monitor")


_ensure_canonical_layer_importable()

from shared.python.launch_monitor.comparison import (
    MonitorComparisonResult,
    MonitorSummary,
    PairwiseMonitorComparison,
    compare_monitors,
)
from shared.python.launch_monitor.conformance_bundle import (
    LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
    LaunchMonitorConformanceBundleV1,
    LaunchMonitorConformanceScenarioV1,
    launch_monitor_conformance_bundle_json_schema,
    launch_monitor_conformance_bundle_sha256,
    launch_monitor_conformance_scenario_sha256,
)
from shared.python.launch_monitor.corpus import (
    CORPUS_COLUMN_MAP,
    corpus_dataset_path,
    load_private_corpus,
)
from shared.python.launch_monitor.contract_v2 import (
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
from shared.python.launch_monitor.dataset_reference import (
    DATASET_JOB_CONTRACT_VERSION,
    DatasetJobRequestV1,
    DatasetOperationV1,
    DatasetReferenceV1,
    DatasetUnavailableStateV1,
    dataset_content_sha256,
    dataset_job_contract_json_schema,
)
from shared.python.launch_monitor.flexible_analysis import (
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
from shared.python.launch_monitor.importer import import_session
from shared.python.launch_monitor.longitudinal import (
    analyze_longitudinal_sessions,
    longitudinal_session_contract_json_schema,
)
from shared.python.launch_monitor.longitudinal_types import (
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
from shared.python.launch_monitor.modeling import (
    PredictiveModelResult,
    fit_predictive_model,
)
from shared.python.launch_monitor.multivariate import (
    PCAResult,
    VIFResult,
    compute_pca,
    compute_vif,
)
from shared.python.launch_monitor.outcome_proxy import analyze_outcome_proxy
from shared.python.launch_monitor.player_covariation import (
    analyze_player_covariation_v1,
    player_covariation_contract_json_schema,
    scan_player_covariation_v1,
)
from shared.python.launch_monitor.player_covariation_types import (
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
from shared.python.launch_monitor.profiles import (
    PROFILES,
    ImportProfile,
    ProfileDetection,
    detect_profile,
    normalize_header,
)
from src.tools.launch_monitor_model.project import LaunchMonitorProject
from shared.python.launch_monitor.relationships import (
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
from shared.python.launch_monitor.strokes_gained import (
    analyze_source_backed_strokes_gained,
    strokes_gained_contract_json_schema,
)
from shared.python.launch_monitor.strokes_gained_types import (
    BASELINE_CONTRACT_VERSION,
    OUTCOME_PROXY_CONTRACT_VERSION,
    STROKES_GAINED_CONTRACT_VERSION,
    CourseStateColumnsV1,
    ExpectedStrokesBaselineLike,
    ExpectedStrokesStateLike,
    GroupingDimensionV1,
    LongitudinalDimensionV1,
    LongitudinalMethod,
    OutcomeProxyRequestV1,
    OutcomeProxyResultV1,
    StrokesGainedAnalysisResultV1,
    StrokesGainedRequestV1,
)

# ADR-0048 step P12 deliberately left the expected-strokes baseline half out of
# the canonical port: Tools' ``rate_of_closure.launch_monitor_strokes_gained_
# baseline`` is already the authority for loading and digest-verifying that
# artifact, so the canonical layer types its ``baseline`` argument against a
# protocol instead. UpstreamDrift still needs a *validating* model, because the
# analytics API parses one off the wire. See ``strokes_gained_baseline``.
from src.tools.launch_monitor_model.strokes_gained_baseline import (
    ExpectedStrokesBaselineV2,
    ExpectedStrokesStateV2,
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
    "ExpectedStrokesBaselineLike",
    "ExpectedStrokesBaselineV2",
    "ExpectedStrokesStateLike",
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
    "LongitudinalMethod",
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
