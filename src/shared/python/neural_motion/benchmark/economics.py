"""Break-even economics and amortization calculations for NM-10.

Governing Issue: #10625
Parent Epic: #10603
"""

from __future__ import annotations

import math
from typing import Final

from .types import BreakEvenReceipt


def compute_break_even(
    training_wall_time_s: float,
    training_cost_usd: float,
    baseline_latency_s: float,
    candidate_latency_s: float,
    baseline_query_cost_usd: float,
    candidate_query_cost_usd: float,
) -> BreakEvenReceipt:
    """Compute wall-time and economic break-even points.

    Design by Contract:
        If savings <= 0, strictly reports has_break_even=False with None queries.
    """
    for arg_name, arg_val in (
        ("training_wall_time_s", training_wall_time_s),
        ("training_cost_usd", training_cost_usd),
        ("baseline_latency_s", baseline_latency_s),
        ("candidate_latency_s", candidate_latency_s),
        ("baseline_query_cost_usd", baseline_query_cost_usd),
        ("candidate_query_cost_usd", candidate_query_cost_usd),
    ):
        if (
            not isinstance(arg_val, (int, float))
            or not math.isfinite(arg_val)
            or arg_val < 0.0
        ):
            raise ValueError(
                f"{arg_name} must be non-negative and finite, got {arg_val}"
            )

    latency_savings_s = baseline_latency_s - candidate_latency_s
    dollar_savings_usd = baseline_query_cost_usd - candidate_query_cost_usd

    # Threshold for positive savings
    has_wall_time_savings = latency_savings_s > 1e-9
    has_dollar_savings = dollar_savings_usd > 1e-9

    has_break_even = has_wall_time_savings and has_dollar_savings

    if not has_break_even:
        reasons = []
        if not has_wall_time_savings:
            reasons.append(
                f"candidate latency ({candidate_latency_s:.4f}s) >= baseline ({baseline_latency_s:.4f}s)"
            )
        if not has_dollar_savings:
            reasons.append(
                f"candidate query cost (${candidate_query_cost_usd:.6f}) >= baseline (${baseline_query_cost_usd:.6f})"
            )
        explanation = f"No break-even: {'; '.join(reasons)}"
        return BreakEvenReceipt(
            training_wall_time_s=training_wall_time_s,
            training_cost_usd=training_cost_usd,
            baseline_query_latency_s=baseline_latency_s,
            candidate_query_latency_s=candidate_latency_s,
            per_query_latency_savings_s=latency_savings_s,
            baseline_query_cost_usd=baseline_query_cost_usd,
            candidate_query_cost_usd=candidate_query_cost_usd,
            per_query_dollar_savings_usd=dollar_savings_usd,
            has_break_even=False,
            wall_time_break_even_queries=None,
            dollar_break_even_queries=None,
            explanation=explanation,
        )

    # Compute positive break-even queries
    wall_time_queries = int(math.ceil(training_wall_time_s / latency_savings_s))
    dollar_queries = int(math.ceil(training_cost_usd / dollar_savings_usd))

    explanation = (
        f"Break-even achieved: wall time amortizes in {wall_time_queries} queries "
        f"({latency_savings_s:.4f}s/query savings), compute cost amortizes in "
        f"{dollar_queries} queries (${dollar_savings_usd:.6f}/query savings)."
    )

    return BreakEvenReceipt(
        training_wall_time_s=training_wall_time_s,
        training_cost_usd=training_cost_usd,
        baseline_query_latency_s=baseline_latency_s,
        candidate_query_latency_s=candidate_latency_s,
        per_query_latency_savings_s=latency_savings_s,
        baseline_query_cost_usd=baseline_query_cost_usd,
        candidate_query_cost_usd=candidate_query_cost_usd,
        per_query_dollar_savings_usd=dollar_savings_usd,
        has_break_even=True,
        wall_time_break_even_queries=wall_time_queries,
        dollar_break_even_queries=dollar_queries,
        explanation=explanation,
    )
