"""Shared utility for generating reports with agentic insights."""

import json
from typing import Any, Protocol


class InsightsProvider(Protocol):
    """Protocol for providing AI insights."""

    async def generate_insights(self, prompt: str) -> str:
        """Generate insights given a prompt."""
        ...


class ReportGenerator:
    """Shared report generator for injecting agentic insights into reports."""

    def __init__(self, insights_provider: InsightsProvider | None = None) -> None:
        self.insights_provider = insights_provider

    async def generate_agentic_insights(self, data: dict[str, Any]) -> str:
        """Analyze simulation data and generate human-readable insights."""
        if not self.insights_provider:
            return "Agentic Insights: AI provider not configured."

        prompt = (
            "Analyze the following simulation output data. "
            "Identify any anomalies, convergence failures, boundary conditions, "
            "or mass balance issues. Provide a clear, human-readable summary "
            "of the insights.\n\n"
            f"Data:\n{json.dumps(data, indent=2, default=str)}\n\n"
            "Insights:"
        )

        try:
            return await self.insights_provider.generate_insights(prompt)
        except Exception as e:
            return f"Agentic Insights Error: {e}"
