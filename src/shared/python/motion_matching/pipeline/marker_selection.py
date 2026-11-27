"""Which markers a pipeline run fits and where its leg calibration starts (#15296).

A document may carry attachment seeds for the eight leg markers (a subject
preset's ``marker_seeds``) and name capture labels a fit leaves out
(``subject.dropped_markers``). The committed documents carry neither, so for
them the selection is what the pipeline always used: every attached marker
plus the built-in ``LEG_SEEDS``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.shared.python.motion_matching.pipeline.constants import LEG_SEEDS

Seeds = dict[str, tuple[str, tuple[float, float, float]]]

#: ``document``: the document's leg seeds when it carries them, the built-in
#: ``LEG_SEEDS`` otherwise. ``builtin``: always ``LEG_SEEDS``.
LEG_SEED_SOURCES = ("document", "builtin")
DEFAULT_LEG_SEED_SOURCE = LEG_SEED_SOURCES[0]
LOG = logging.getLogger(__name__)


def add_marker_selection_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--leg-seeds`` to the pipeline command line."""
    parser.add_argument(
        "--leg-seeds",
        choices=LEG_SEED_SOURCES,
        default=DEFAULT_LEG_SEED_SOURCE,
        help="where the leg marker calibration starts: 'document' uses the "
        "spec's own leg attachment seeds when it carries all eight (a subject "
        "preset) and the built-in seeds otherwise; 'builtin' ignores them",
    )


@dataclass(frozen=True)
class MarkerSelection:
    """Markers of one run: ``attachments`` (document seeds), ``labels`` (fitted
    capture labels), ``leg_seeds`` (``None``: built-in) and ``dropped``."""

    attachments: Seeds
    labels: tuple[str, ...]
    leg_seeds: Seeds | None
    dropped: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        """Receipt block; empty for the default selection so receipts keep."""
        out: dict[str, Any] = {}
        if self.leg_seeds is not None:
            out["leg_seeds"] = DEFAULT_LEG_SEED_SOURCE
        if self.dropped:
            out["dropped_markers"] = list(self.dropped)
        return out


def _dropped(document: Mapping[str, Any], capture_labels: Sequence[str]) -> list[str]:
    dropped = document.get("subject", {}).get("dropped_markers")
    if dropped is None:  # omitted: nothing is dropped
        return []
    # Only a list is a label list: ``or []`` would let 0 / False / {} through, and
    # a dict or tuple would pass the per-item check by iterating its keys (#15342).
    if not isinstance(dropped, list) or not all(isinstance(x, str) for x in dropped):
        raise ValueError("subject.dropped_markers must be a list of labels")
    if unknown := sorted(set(dropped) - set(capture_labels)):
        raise ValueError(f"dropped markers are not in the capture: {unknown}")
    if needed := sorted(set(dropped) & set(LEG_SEEDS)):
        raise ValueError(f"the leg calibration needs the dropped markers {needed}")
    return sorted(set(dropped))


def select_markers(
    document: Mapping[str, Any],
    capture_labels: Sequence[str],
    leg_seed_source: str = DEFAULT_LEG_SEED_SOURCE,
) -> MarkerSelection:
    """The marker selection of ``document`` for a capture with ``capture_labels``.

    Postconditions: no dropped label is in ``attachments`` or ``labels``; with
    no leg seeds and no dropped labels in the document the result equals the
    pre-#15296 selection. Raises ``ValueError`` for an unknown source, dropped
    labels that are not a list of capture labels or that the leg calibration
    needs, and a document seeding only some of the eight leg markers.
    """
    if leg_seed_source not in LEG_SEED_SOURCES:
        raise ValueError(
            f"leg seed source must be one of {LEG_SEED_SOURCES}, "
            f"got {leg_seed_source!r}"
        )
    dropped = _dropped(document, capture_labels)
    attachments: Seeds = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in document["marker_attachments"].items()
        if a["offset_m"] is not None and label not in dropped
    }
    carried = {label: attachments[label] for label in LEG_SEEDS if label in attachments}
    if carried and len(carried) < len(LEG_SEEDS):
        missing = sorted(set(LEG_SEEDS) - set(carried))
        raise ValueError(f"document seeds some leg markers but not {missing}")
    use_document = bool(carried) and leg_seed_source == DEFAULT_LEG_SEED_SOURCE
    if dropped or use_document:
        LOG.info("leg seeds from the document: %s; dropped %s", use_document, dropped)
    return MarkerSelection(
        attachments,
        tuple({**attachments, **LEG_SEEDS}),
        carried if use_document else None,
        tuple(dropped),
    )


def base_leg_seeds(lane: Any) -> Seeds:
    """Leg seeds ``lane`` calibrates from, before any forefoot squaring."""
    return dict(getattr(lane, "document_leg_seeds", None) or LEG_SEEDS)
