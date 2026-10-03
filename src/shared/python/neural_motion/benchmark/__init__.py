"""Neural motion matching benchmark suite (NM-10, #10625)."""

from __future__ import annotations

from .types import (
    AcceptedMatchMetrics,
    BenchmarkMethod,
    BreakEvenReceipt,
    DataEfficiencyCurve,
    LatencySummary,
    MeasuredBaseline,
    ModelBenchmarkCard,
    MultiSeedEfficiencySummary,
    PromotionDecision,
    PromotionGateResult,
    TimingBreakdown,
)
from .economics import compute_break_even
from .efficiency import (
    compute_budget_to_target_ratio,
    evaluate_data_efficiency,
    evaluate_multi_seed_data_efficiency,
    trapezoidal_auc,
)
from .gates import evaluate_promotion_gate
from .runner import (
    measure_synchronous_latency,
    run_model_comparative_benchmark,
    synchronize_device,
)

__all__ = [
    "AcceptedMatchMetrics",
    "BenchmarkMethod",
    "BreakEvenReceipt",
    "DataEfficiencyCurve",
    "LatencySummary",
    "MeasuredBaseline",
    "ModelBenchmarkCard",
    "MultiSeedEfficiencySummary",
    "PromotionDecision",
    "PromotionGateResult",
    "TimingBreakdown",
    "compute_break_even",
    "compute_budget_to_target_ratio",
    "evaluate_data_efficiency",
    "evaluate_multi_seed_data_efficiency",
    "evaluate_promotion_gate",
    "measure_synchronous_latency",
    "run_model_comparative_benchmark",
    "synchronize_device",
    "trapezoidal_auc",
]
