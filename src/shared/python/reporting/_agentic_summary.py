"""Agentic summary generator — LLM-backed markdown insights for simulation runs.

Design-by-Contract invariants
------------------------------
- ``run_id`` must be a non-empty alphanumeric/hyphen/underscore string.
- ``generate`` postcondition: result is a non-empty string.

Law of Demeter
--------------
``AgenticSummaryGenerator`` delegates AI calls to ``_AIClient`` and template
rendering to ``_FallbackRenderer``.  It never reaches through more than one
layer into either dependency.

Implements part of Issue #5423: Global Report Templates and Agentic Summaries.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

from src.shared.python.contracts import ensure, require
from src.shared.python.simulation_store import SimulationDataStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
_FALLBACK_TEMPLATE = """\
# Simulation Run Summary — {run_id}

## Overview
Run ID: {run_id}

## Data
{data_summary}

## Insights
*AI assistant unavailable — template-based summary generated.*

{template_insights}
"""

_PROMPT_TEMPLATE = """\
You are an expert golf biomechanics simulation analyst.

Analyse the following simulation run data and produce a concise markdown
"Insights" section (150–300 words) covering:
- Key performance indicators (e.g. club head speed, ball speed, launch angle)
- Notable anomalies or convergence issues
- Actionable recommendations

Run ID: {run_id}
Simulation Data:
{data_json}

Respond with ONLY the markdown "## Insights" section and its content.
"""


# ---------------------------------------------------------------------------
# Protocol — AI client (dependency injection)
# ---------------------------------------------------------------------------


@runtime_checkable
class AIClient(Protocol):
    """Minimal protocol for an AI completion client.

    Any object implementing ``complete(prompt: str) -> str`` satisfies
    this protocol, enabling easy mocking in tests.
    """

    def complete(self, prompt: str) -> str:
        """Return AI-generated text for *prompt*."""
        ...


# ---------------------------------------------------------------------------
# Private helpers — Law of Demeter boundary
# ---------------------------------------------------------------------------


class _FallbackRenderer:
    """Renders a template-based summary when the AI client is unavailable."""

    def render(self, run_id: str, data: dict[str, Any]) -> str:
        """Return a template-based markdown summary."""
        data_summary_lines = [f"- **{k}**: {v}" for k, v in data.items()]
        data_summary = (
            "\n".join(data_summary_lines) if data_summary_lines else "*(empty)*"
        )
        # Extract numeric values for simple template insights
        numeric_values = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        if numeric_values:
            template_insights = "Key numeric parameters: " + ", ".join(
                f"{k}={v}" for k, v in list(numeric_values.items())[:5]
            )
        else:
            template_insights = "No numeric metrics found in run data."
        return _FALLBACK_TEMPLATE.format(
            run_id=run_id,
            data_summary=data_summary,
            template_insights=template_insights,
        )


class _AIInsightsExtractor:
    """Wraps an ``AIClient`` and extracts the insights section from its output."""

    def __init__(self, client: AIClient) -> None:
        self._client = client

    def extract(self, run_id: str, data: dict[str, Any]) -> str:
        """Call the AI client and return the insights markdown."""
        import json

        prompt = _PROMPT_TEMPLATE.format(
            run_id=run_id,
            data_json=json.dumps(data, indent=2, default=str),
        )
        return self._client.complete(prompt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AgenticSummaryGenerator:
    """Generates structured markdown summaries for simulation runs.

    Uses an AI assistant when available, falls back to a template-based
    summary otherwise.

    Args:
        store: ``SimulationDataStore`` instance for loading run data.
        ai_client: Optional object implementing ``AIClient`` protocol.
            When ``None`` or unavailable, the fallback template is used.

    Examples::

        store = SimulationDataStore()
        gen = AgenticSummaryGenerator(store)
        md = gen.generate("run_001")

        # With a custom AI client:
        gen = AgenticSummaryGenerator(store, ai_client=my_llm)
        md = gen.generate("run_001")
    """

    def __init__(
        self,
        store: SimulationDataStore,
        ai_client: AIClient | None = None,
    ) -> None:
        require(
            isinstance(store, SimulationDataStore),
            "store must be a SimulationDataStore instance",
            store,
        )
        self._store = store
        self._ai_client = ai_client
        self._fallback = _FallbackRenderer()
        logger.debug(
            "agentic_summary_generator_created ai_available=%s",
            ai_client is not None,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def has_ai_client(self) -> bool:
        """Return ``True`` if an AI client is configured."""
        return self._ai_client is not None

    def generate(self, run_id: str) -> str:
        """Generate a structured markdown summary for *run_id*.

        Preconditions:
            - ``run_id`` is a non-empty alphanumeric/hyphen/underscore string
              (max 256 chars).

        Postcondition:
            Returns a non-empty string.

        Raises:
            ValueError: If ``run_id`` violates the naming contract.
            KeyError: If no run with ``run_id`` exists in the store.

        Args:
            run_id: Identifier of a saved simulation run.

        Returns:
            Markdown-formatted summary string.
        """
        require(
            isinstance(run_id, str) and bool(_RUN_ID_PATTERN.match(run_id)),
            "run_id must be a non-empty alphanumeric/hyphen/underscore string (max 256 chars)",
            run_id,
        )

        data = self._store.load_run(run_id)
        result = self._generate_summary(run_id, data)

        ensure(
            isinstance(result, str) and bool(result.strip()),
            "generate postcondition: result must be a non-empty string",
        )
        logger.info("agentic_summary_generated run_id=%s", run_id)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_summary(self, run_id: str, data: dict[str, Any]) -> str:
        """Attempt AI generation; fall back to template on any failure."""
        if self._ai_client is None:
            logger.debug("agentic_summary_using_fallback ai_client=None")
            return self._fallback.render(run_id, data)
        try:
            extractor = _AIInsightsExtractor(self._ai_client)
            insights = extractor.extract(run_id, data)
            return self._assemble_full_report(run_id, data, insights)
        except Exception:
            logger.warning(
                "agentic_summary_ai_failed run_id=%s falling_back=True",
                run_id,
                exc_info=True,
            )
            return self._fallback.render(run_id, data)

    def _assemble_full_report(
        self,
        run_id: str,
        data: dict[str, Any],
        insights: str,
    ) -> str:
        """Combine run metadata with AI-generated insights into a full report."""
        data_lines = [f"- **{k}**: {v}" for k, v in data.items()]
        data_section = "\n".join(data_lines) if data_lines else "*(empty)*"
        return (
            f"# Simulation Run Summary — {run_id}\n\n"
            f"## Overview\nRun ID: {run_id}\n\n"
            f"## Data\n{data_section}\n\n"
            f"{insights}\n"
        )
