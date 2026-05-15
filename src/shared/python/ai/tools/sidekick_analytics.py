"""Sidekick analytics tools.

Exposes a deterministic summarizer for simulation runs so the AI chat
assistant can produce objective baseline reports without ad-hoc LLM
prompts. Today only one tool is registered:

    summarize_simulation_run(run_id) -> dict

The summary is built from a JSON ``manifest.json`` stored under
``<runs_dir>/<run_id>/manifest.json``. The default ``runs_dir`` is
``~/.golf_modeling_suite/runs`` but can be overridden via the
``UPSTREAMDRIFT_SIM_RUNS_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────

_DEFAULT_RUNS_DIR = Path.home() / ".golf_modeling_suite" / "runs"
_RUNS_DIR_ENV_VAR = "UPSTREAMDRIFT_SIM_RUNS_DIR"
_PATH_SEPARATORS: tuple[str, ...] = ("/", "\\")


def _get_runs_dir() -> Path:
    """Return the configured simulation-runs directory."""
    override = os.environ.get(_RUNS_DIR_ENV_VAR)
    return Path(override) if override else _DEFAULT_RUNS_DIR


def _validate_run_id(run_id: Any) -> str:
    """Validate ``run_id`` against type, emptiness, and path traversal.

    Returns:
        The validated ``run_id`` as a stripped string.

    Raises:
        TypeError: If ``run_id`` is not a string.
        ValueError: If ``run_id`` is empty or contains path separators
            (``/``, ``\\``) or path traversal segments.
    """
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    cleaned = run_id.strip()
    if not cleaned:
        raise ValueError("run_id must be a non-empty string")
    if any(sep in cleaned for sep in _PATH_SEPARATORS):
        raise ValueError("run_id must not contain path separators ('/' or '\\\\')")
    # Defence in depth: reject explicit parent-dir tokens too.
    if cleaned in {".", ".."} or cleaned.startswith(".."):
        raise ValueError("run_id must not be a path traversal token")
    return cleaned


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and parse a manifest JSON file.

    Raises:
        ValueError: If the file cannot be read or is not a JSON object.
    """
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read manifest at {manifest_path}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Manifest at {manifest_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Manifest at {manifest_path} is not a JSON object")
    return data


def _build_summary_sentence(
    run_id: str,
    engine: str,
    duration_s: float | None,
    n_frames: int | None,
    key_metrics: dict[str, Any],
) -> str:
    """Build the deterministic, one-paragraph natural-language summary."""
    parts: list[str] = [f"Run {run_id} executed on the {engine} engine"]

    if duration_s is not None:
        parts.append(f"for {duration_s:.3f} s")
    if n_frames is not None:
        parts.append(f"across {n_frames} frame{'s' if n_frames != 1 else ''}")

    head = " ".join(parts).rstrip()

    if key_metrics:
        # Sort for determinism.
        metric_bits = [
            f"{k}={v}" for k, v in sorted(key_metrics.items(), key=lambda kv: kv[0])
        ]
        tail = "; key metrics — " + ", ".join(metric_bits)
    else:
        tail = "; no key metrics recorded"

    return head + tail + "."


# ── Public API ───────────────────────────────────────────────────────


def summarize_simulation_run(run_id: str) -> dict[str, Any]:
    """Return a structured summary of a simulation run.

    Args:
        run_id: Identifier of the run. Must be a non-empty string with
            no path separators (``/`` or ``\\``) or path traversal
            tokens. The run is located at ``<runs_dir>/<run_id>/``.

    Returns:
        Dict::

            {
                "run_id": str,
                "engine": str,
                "duration_s": float | None,
                "n_frames": int | None,
                "key_metrics": dict[str, Any],
                "summary": str,
            }

    Raises:
        TypeError: If ``run_id`` is not a string.
        ValueError: If ``run_id`` fails validation OR the run does not
            exist OR its manifest cannot be parsed.
    """
    validated = _validate_run_id(run_id)

    runs_dir = _get_runs_dir()
    manifest_path = runs_dir / validated / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Unknown run_id: {validated}")

    manifest = _load_manifest(manifest_path)

    engine_raw = manifest.get("engine", "unknown")
    engine = str(engine_raw) if engine_raw is not None else "unknown"

    duration_raw = manifest.get("duration_s")
    duration_s: float | None
    try:
        duration_s = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration_s = None

    n_frames_raw = manifest.get("n_frames")
    n_frames: int | None
    try:
        n_frames = int(n_frames_raw) if n_frames_raw is not None else None
    except (TypeError, ValueError):
        n_frames = None

    key_metrics_raw = manifest.get("key_metrics") or {}
    key_metrics: dict[str, Any] = (
        dict(key_metrics_raw) if isinstance(key_metrics_raw, dict) else {}
    )

    summary = _build_summary_sentence(
        run_id=validated,
        engine=engine,
        duration_s=duration_s,
        n_frames=n_frames,
        key_metrics=key_metrics,
    )

    return {
        "run_id": validated,
        "engine": engine,
        "duration_s": duration_s,
        "n_frames": n_frames,
        "key_metrics": key_metrics,
        "summary": summary,
    }


# ── Registry hookup ──────────────────────────────────────────────────


_TOOL_NAME = "summarize_simulation_run"
_TOOL_DESCRIPTION = (
    "Summarize a stored simulation run by id. Returns engine, duration, "
    "frame count, key metrics, and a deterministic natural-language "
    "summary suitable for the chat assistant to rephrase."
)


def register_sidekick_analytics_tools(registry: ToolRegistry) -> None:
    """Register Sidekick analytics tools with ``registry``.

    Args:
        registry: Target :class:`ToolRegistry`.

    Raises:
        TypeError: If ``registry`` is not a :class:`ToolRegistry`.
    """
    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be a ToolRegistry instance")

    registry.register(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        category=ToolCategory.ANALYSIS,
        expertise_level=1,
    )(summarize_simulation_run)

    logger.debug("Registered Sidekick analytics tool: %s", _TOOL_NAME)
