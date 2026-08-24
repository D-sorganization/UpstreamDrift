"""Perturbation analysis package for upstream drift.

Unified platform utilities for conducting Monte Carlo sensitivity
analysis on engine dynamics.
"""

from .config import PerturbationAnalyzer, PerturbationConfig, PerturbationSummary
from .articulated_mujoco_trial_adapter import (
    ArticulatedMujocoTrialAdapter,
    ArticulatedMujocoTrialConfig,
    ArticulatedMujocoTrialResult,
    MujocoVariationBinding,
    NamedJointTorque,
)
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
from .canonical_trial_campaign import (
    CanonicalVariationCampaignResult,
    execute_batched_variation_campaign,
    execute_serial_variation_campaign,
)
from .double_pendulum_trial_adapter import (
    DoublePendulumTrialAdapter,
    DoublePendulumTrialConfig,
    DoublePendulumTrialResult,
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
from .trial_evidence_bundle import (
    TrialEvidenceBundleSummary,
    load_trial_evidence_bundle,
    validate_trial_evidence_bundle,
    write_trial_evidence_bundle,
)

__all__ = [
    "PerturbationAnalyzer",
    "PerturbationConfig",
    "PerturbationSummary",
    "ArticulatedMujocoTrialAdapter",
    "ArticulatedMujocoTrialConfig",
    "ArticulatedMujocoTrialResult",
    "MujocoVariationBinding",
    "NamedJointTorque",
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
    "TrialEvidenceBundleSummary",
    "load_trial_evidence_bundle",
    "validate_trial_evidence_bundle",
    "write_trial_evidence_bundle",
    "TrialEvidenceCollector",
    "execute_batched_variation",
    "execute_serial_variation",
    "CanonicalVariationCampaignResult",
    "execute_batched_variation_campaign",
    "execute_serial_variation_campaign",
    "DoublePendulumTrialAdapter",
    "DoublePendulumTrialConfig",
    "DoublePendulumTrialResult",
    "CrossEngineCompatibilityError",
    "CrossEngineParityMetrics",
    "CrossEngineTolerances",
    "compare_cross_engine_trials",
]
