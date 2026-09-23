"""Neural motion matching benchmark suite (NM-10, #10625)."""

from __future__ import annotations

from .types import (
    AcceptedMatchMetrics,
    BenchmarkMethod,
    BreakEvenReceipt,
    DataEfficiencyCurve,
    LatencySummary,
    ModelBenchmarkCard,
    PromotionDecision,
    PromotionGateResult,
    TimingBreakdown,
)
from .economics import compute_break_even
from .efficiency import evaluate_data_efficiency
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
    "ModelBenchmarkCard",
    "PromotionDecision",
    "PromotionGateResult",
    "TimingBreakdown",
    "compute_break_even",
    "evaluate_data_efficiency",
    "evaluate_promotion_gate",
    "measure_synchronous_latency",
    "run_model_comparative_benchmark",
    "synchronize_device",
]
