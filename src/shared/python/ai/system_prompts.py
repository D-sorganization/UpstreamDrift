"""Application-agnostic system prompt construction.

Replaces the hardcoded "Golf Modeling Suite" system prompts across all
adapters with configurable, app-context-aware prompt builders.

Usage::

    from src.shared.python.ai.system_prompts import build_system_prompt

    prompt = build_system_prompt(
        app_context="gasification",
        expertise_level="advanced",
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Application context registry ────────────────────────────────────

_APP_CONTEXTS: dict[str, dict[str, Any]] = {
    "gasification": {
        "name": "Integrated Process Simulator",
        "description": (
            "a research-grade thermodynamic modeling environment for "
            "coal gasification, quench systems, and integrated process simulation"
        ),
        "capabilities": [
            "Running thermodynamic equilibrium calculations",
            "Performing Gibbs free energy minimization",
            "Computing quench system parameters",
            "Analyzing syngas composition and properties",
            "Managing multi-zone reactor models",
            "Exporting results to CSV/JSON",
        ],
    },
    "upstream_drift": {
        "name": "UpstreamDrift Analysis Suite",
        "description": (
            "a research-grade biomechanics simulation platform for "
            "analyzing golf swings using multi-physics engine orchestration"
        ),
        "capabilities": [
            "Analyzing C3D motion capture data",
            "Running physics simulations (MuJoCo, Drake, Pinocchio)",
            "Computing inverse dynamics and joint torques",
            "Performing drift-control decomposition",
            "Generating visualizations and reports",
            "Cross-engine comparison analysis",
            "Summarizing stored simulation runs with summarize_simulation_run",
        ],
    },
    "tools": {
        "name": "Engineering Tools Suite",
        "description": (
            "a collection of engineering analysis and simulation tools "
            "including pendulum models, glass bath FEA, and utilities"
        ),
        "capabilities": [
            "Running pendulum simulations",
            "Performing finite element analysis",
            "Glass bath thermal modeling",
            "Engineering calculations and unit conversions",
        ],
    },
}

# Fallback for unknown contexts
_DEFAULT_CONTEXT = {
    "name": "AI Assistant",
    "description": "a helpful engineering AI assistant",
    "capabilities": [
        "Answering technical questions",
        "Analyzing data and results",
        "Providing step-by-step guidance",
        "Generating reports and visualizations",
    ],
}


def register_app_context(
    context_key: str,
    name: str,
    description: str,
    capabilities: list[str],
) -> None:
    """Register a new application context for system prompts.

    Allows consuming applications to register their own context
    without modifying this module.

    Args:
        context_key: Short identifier (e.g., "gasification").
        name: Human-readable application name.
        description: One-sentence description.
        capabilities: List of capability descriptions.
    """
    if not context_key or not context_key.strip():
        raise ValueError("context_key must be a non-empty string")
    _APP_CONTEXTS[context_key.lower()] = {
        "name": name,
        "description": description,
        "capabilities": capabilities,
    }
    logger.debug("Registered app context: %s", context_key)


def build_system_prompt(
    app_context: str = "assistant",
    expertise_level: str = "beginner",
    extra_instructions: str | None = None,
) -> str:
    """Build a context-aware system prompt for the AI assistant.

    Args:
        app_context: Application context key (e.g., "gasification").
        expertise_level: User expertise level.
        extra_instructions: Additional instructions to append.

    Returns:
        Complete system prompt string.
    """
    ctx = _APP_CONTEXTS.get(app_context.lower(), _DEFAULT_CONTEXT)
    name = ctx["name"]
    desc = ctx["description"]
    caps = ctx["capabilities"]

    caps_text = "\n".join(f"- {cap}" for cap in caps)

    prompt = (
        f"You are an AI assistant for {name}, {desc}.\n\n"
        f"Current user expertise level: {expertise_level}\n\n"
        f"Your capabilities include:\n{caps_text}\n\n"
        f"Guidelines:\n"
        f"1. Use tools to perform analyses — never fabricate numerical results\n"
        f"2. Explain concepts at the {expertise_level} level\n"
        f"3. Validate scientific claims before presenting them\n"
        f"4. Guide users through workflows step by step\n"
        f"5. Acknowledge uncertainty and cite limitations\n"
        f"6. Be precise about physical units (SI: m, kg, s, rad, N, N·m)"
    )

    if extra_instructions:
        prompt += f"\n\nAdditional context:\n{extra_instructions}"

    return prompt


def get_registered_contexts() -> list[str]:
    """Return list of registered application context keys.

    Returns:
        Sorted list of context keys.
    """
    return sorted(_APP_CONTEXTS.keys())
