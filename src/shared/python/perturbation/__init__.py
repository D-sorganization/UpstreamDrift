"""Perturbation analysis package for upstream drift.

Unified platform utilities for conducting Monte Carlo sensitivity
analysis on engine dynamics.
"""

from .config import PerturbationAnalyzer, PerturbationConfig, PerturbationSummary
from .cross_engine_trial_parity import (
    CrossEngineCompatibilityError,
    CrossEngineParityMetrics,
    CrossEngineTolerances,
    compare_cross_engine_trials,
)
from .canonical_trial_executor import (
    TrialEvidenceCollector,
    execute_batched_variation,
    execute_serial_variation,
)
from .noise import generate_noise
from .robustness_score import compute_robustness_score
from .statistics import MetricStatistics, compute_metric_statistics
from .tools_variation_adapter import (
    ToolsVariationCapabilities,
    ToolsVariationCompatibilityError,
    ToolsVariationGateway,
    ToolsVariationUnavailableError,
    load_tools_variation_gateway,
    probe_tools_variation,
)
from .trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    ImpactObservation,
    SampledInput,
    TrialTrace,
)

__all__ = [
    "PerturbationAnalyzer",
    "PerturbationConfig",
    "PerturbationSummary",
    "generate_noise",
    "compute_robustness_score",
    "MetricStatistics",
    "compute_metric_statistics",
    "ToolsVariationCapabilities",
    "ToolsVariationCompatibilityError",
    "ToolsVariationGateway",
    "ToolsVariationUnavailableError",
    "load_tools_variation_gateway",
    "probe_tools_variation",
    "CanonicalTrialEvidence",
    "ClosestApproach",
    "ImpactObservation",
    "SampledInput",
    "TrialTrace",
    "TrialEvidenceCollector",
    "execute_batched_variation",
    "execute_serial_variation",
    "CrossEngineCompatibilityError",
    "CrossEngineParityMetrics",
    "CrossEngineTolerances",
    "compare_cross_engine_trials",
]
