"""Glossary routes.

Provides endpoints for looking up physics and biomechanics terms
from the golf modeling suite glossary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

_GLOSSARY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "shared"
    / "python"
    / "ai"
    / "data"
    / "glossary_core.json"
)


def _load_glossary() -> list[dict[str, Any]]:
    """Load glossary entries from disk.

    Returns:
        List of glossary entry dicts.
    """
    if not _GLOSSARY_PATH.exists():
        logger.warning("Glossary file not found at %s", _GLOSSARY_PATH)
        return []
    with open(_GLOSSARY_PATH, encoding="utf-8") as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def _build_glossary_index() -> dict[str, dict[str, Any]]:
    """Build a lookup dict keyed by term id.

    Returns:
        Mapping of term key -> entry dict.
    """
    return {entry["key"]: entry for entry in _load_glossary()}


_LEVEL_FIELD = {
    "beginner": "b",
    "intermediate": "i",
    "advanced": "a",
    "expert": "a",
}


@router.get("/glossary/{term_id}")
async def get_glossary_term(term_id: str, level: str = "beginner") -> dict[str, Any]:
    """Return the glossary entry for a physics or biomechanics term.

    Args:
        term_id: The snake_case identifier for the term (e.g. ``equations_of_motion``).
        level: Optional expertise level (``beginner``, ``intermediate``,
            ``advanced``). Selects which definition text populates the
            ``definition`` field. Raw entry fields ``b``/``i``/``a`` are
            always included for backwards compatibility.

    Returns:
        Glossary entry with fields: key, term, cat, b (beginner),
        i (intermediate), a (advanced), f (formula), r (related terms),
        definition (level-selected), level, term_id, related_terms.

    Raises:
        HTTPException 404: If the term is not found in the glossary.
    """
    if not term_id:
        raise ValueError("term_id must be provided")

    index = _build_glossary_index()
    entry = index.get(term_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Glossary term not found: {term_id}",
        )

    level_normalized = (level or "beginner").strip().lower()
    field = _LEVEL_FIELD.get(level_normalized, "b")
    definition = entry.get(field) or entry.get("b") or entry.get("i") or ""

    response: dict[str, Any] = dict(entry)
    response["term_id"] = entry["key"]
    response["level"] = level_normalized
    response["definition"] = definition
    response["related_terms"] = entry.get("r", [])
    return response
