"""Producer provenance for cross-engine parity baselines (#12013).

A parity baseline is only evidence for engine X if engine X produced it. The
producing engine is declared by the *generator's* receipt (a JSON sidecar next
to the fixture), never by the label a consumer attaches. A comparison whose
baseline was produced by a different engine than the one it claims is rejected
instead of being reported as cross-engine agreement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

COMMITTED_PROVENANCE_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cross_engine_parity"
    / "simscape_canonical_baseline.provenance.json"
)


class BaselineProvenanceError(ValueError):
    """Baseline provenance is missing, malformed, or not from the claimed engine."""


@dataclass(frozen=True)
class BaselineProvenance:
    """Generator-recorded origin of a baseline artefact."""

    producing_engine: str
    receipt: str
    status: str


def load_provenance(path: Path) -> BaselineProvenance:
    """Load a provenance receipt. Fails closed if absent or incomplete."""
    if not path.is_file():
        msg = f"no provenance receipt at {path}; baseline origin is unverified"
        raise BaselineProvenanceError(msg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"provenance receipt {path} is not valid JSON: {exc}"
        raise BaselineProvenanceError(msg) from exc
    engine = raw.get("producing_engine") if isinstance(raw, dict) else None
    if not isinstance(engine, str) or not engine.strip():
        msg = f"provenance receipt {path} lacks a producing_engine"
        raise BaselineProvenanceError(msg)
    return BaselineProvenance(
        producing_engine=engine.strip().lower(),
        receipt=str(raw.get("receipt", "")),
        status=str(raw.get("status", "")),
    )


def require_producing_engine(
    provenance: BaselineProvenance, claimed_engine: str
) -> None:
    """Raise unless the baseline was produced by ``claimed_engine``."""
    claimed = claimed_engine.strip().lower()
    if not claimed:
        msg = "claimed_engine must be non-empty"
        raise ValueError(msg)
    if provenance.producing_engine != claimed:
        msg = (
            f"baseline was produced by {provenance.producing_engine!r} but is "
            f"claimed as {claimed!r}; comparing against it would be "
            "self-agreement, not cross-engine parity (#12013)"
        )
        raise BaselineProvenanceError(msg)
    logger.info("Baseline provenance verified: %s", provenance.producing_engine)
