"""Cited spinal coupling-law data and loader (SPN-3, issue #12743).

Reads ``src/config/spine_coupling_laws.json`` and builds
:class:`~src.shared.python.spine.coupling.LinearCouplingLaw` objects. The JSON
stores only source numbers copied from
``docs/research/segmental_spine/data_extraction.md`` (plus labelled
assumptions); per-level matrices are derived here, so every literature number
in the file traces to an extraction block.

Distribution types:

* ``regional_split``: published regional percentages per axis, spread
  uniformly within each region (an assumption recorded in the JSON). The
  diagonal gain at a level is ``percent / 100 / n_levels_in_region``.
* ``per_level_ratio``: per-level segment angles measured at a reference trunk
  rotation, scaled to a gain per degree of that rotation.
* ``equal_share``: every level carries ``1 / n_levels`` of each axis.

Entries with ``status == "pending"`` carry no numbers and cannot be built.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.shared.python.spine.coupling import CouplingLaw, LinearCouplingLaw
from src.shared.python.spine.levels import SpinalLevel, SpinalRegion

__all__ = ["available_laws", "coupling_law", "load_coupling_laws"]

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "spine_coupling_laws.json"
)
_AXES = ("flexion_extension", "lateral_bending", "axial_rotation")
_REQUIRED = ("law_id", "version", "kind", "status", "independent", "citation")
_STATUSES = ("available", "pending")


def _validate_entry(entry: object) -> dict:
    if not isinstance(entry, dict):
        raise ValueError("each law entry must be a JSON object")
    missing = [k for k in _REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"law entry {entry.get('law_id')!r} lacks keys {missing}")
    if entry["status"] not in _STATUSES:
        raise ValueError(f"law {entry['law_id']!r}: bad status {entry['status']!r}")
    cite = entry["citation"]
    if not isinstance(cite, dict) or not str(cite.get("source", "")).strip():
        raise ValueError(f"law {entry['law_id']!r}: citation.source is required")
    if not str(cite.get("table", "")).strip():
        raise ValueError(f"law {entry['law_id']!r}: citation.table is required")
    if not isinstance(entry.get("verified_from_primary"), bool):
        raise ValueError(f"law {entry['law_id']!r}: verified_from_primary must be bool")
    if entry["status"] == "pending":
        if not str(entry.get("pending_reason", "")).strip():
            raise ValueError(f"pending law {entry['law_id']!r} needs pending_reason")
        if "distribution" in entry or "matrices" in entry:
            raise ValueError(f"pending law {entry['law_id']!r} must carry no numbers")
    elif "distribution" not in entry:
        raise ValueError(f"law {entry['law_id']!r}: distribution is required")
    return entry


def load_coupling_laws(path: Path | str | None = None) -> Mapping[str, dict]:
    """Return ``law_id -> entry metadata`` (read-only mapping of copies).

    Preconditions: ``path`` is ``None`` (repository default) or a readable
    JSON file with a ``laws`` list; ids are unique; each entry has the
    required keys, a citation with ``source`` and ``table``, and a boolean
    ``verified_from_primary``.

    Raises:
        ValueError: on malformed content or duplicate ids.
        OSError: if the file cannot be read.
    """
    target = DEFAULT_PATH if path is None else Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    laws = raw.get("laws") if isinstance(raw, dict) else None
    if not isinstance(laws, list):
        raise ValueError(f"{target}: expected an object with a 'laws' list")
    out: dict[str, dict] = {}
    for item in laws:
        entry = _validate_entry(item)
        if entry["law_id"] in out:
            raise ValueError(f"duplicate law_id {entry['law_id']!r} in {target}")
        out[entry["law_id"]] = copy.deepcopy(entry)
    return MappingProxyType(out)


def available_laws(include_pending: bool = False) -> tuple[str, ...]:
    """Sorted ids of buildable laws; pending ids too when requested."""
    entries = load_coupling_laws()
    return tuple(
        sorted(
            k
            for k, v in entries.items()
            if include_pending or v["status"] == "available"
        )
    )


def _regional_matrices(dist: dict) -> dict[SpinalLevel, np.ndarray]:
    splits = dist["regional_percent"]
    if dist.get("within_region") != "uniform":
        raise ValueError("regional_split supports within_region == 'uniform' only")
    regions = {"lumbar": SpinalRegion.LUMBAR, "thoracic": SpinalRegion.THORACIC}
    counts = {
        name: sum(1 for lv in SpinalLevel if lv.region is region)
        for name, region in regions.items()
    }
    out = {}
    for lv in SpinalLevel:
        name = "lumbar" if lv.region is SpinalRegion.LUMBAR else "thoracic"
        gains = [splits[axis][name] / 100.0 / counts[name] for axis in _AXES]
        out[lv] = np.diag(gains)
    return out


def _ratio_matrices(dist: dict) -> dict[SpinalLevel, np.ndarray]:
    ref = math.radians(float(dist["reference_trunk_rotation_deg"]))
    out = {}
    sign = dist["isb_sign"]
    keys = ("coupled_flexion", "coupled_lateral_bend", "axial_rotation")
    for row in dist["rows"]:
        values = tuple(sign[key] * row[key][0] for key in keys)
        out[SpinalLevel[row["level"]]] = (
            np.array([[math.radians(v)] for v in values]) / ref
        )
    return out


def _equal_matrices(_dist: dict) -> dict[SpinalLevel, np.ndarray]:
    n = len(SpinalLevel)
    return {lv: np.eye(3) / n for lv in SpinalLevel}


_BUILDERS = {
    "regional_split": _regional_matrices,
    "per_level_ratio": _ratio_matrices,
    "equal_share": _equal_matrices,
}


def _citation_text(entry: dict) -> str:
    cite = entry["citation"]
    parts = [cite["source"]]
    if cite.get("doi"):
        parts.append(f"doi:{cite['doi']}")
    parts.append(cite["table"])
    if cite.get("page"):
        parts.append(f"p. {cite['page']}")
    return "; ".join(parts)


def coupling_law(law_id: str) -> CouplingLaw:
    """Build the :class:`LinearCouplingLaw` for ``law_id``.

    Postconditions: the returned law has read-only matrices, a non-empty
    citation and the independent coordinates named in the JSON.

    Raises:
        TypeError: if ``law_id`` is not a ``str``.
        ValueError: for an unknown id, a pending law (no verified numbers
            shipped) or an unsupported ``kind``/distribution type.
    """
    if not isinstance(law_id, str):
        raise TypeError(f"law_id must be a str, got {type(law_id).__name__}")
    entries = load_coupling_laws()
    if law_id not in entries:
        raise ValueError(f"unknown coupling law {law_id!r}; known: {sorted(entries)}")
    entry = entries[law_id]
    if entry["status"] == "pending":
        raise ValueError(
            f"coupling law {law_id!r} is pending and cannot be built: "
            f"{entry['pending_reason']}"
        )
    if entry["kind"] != "linear":
        raise ValueError(f"law {law_id!r}: unsupported kind {entry['kind']!r}")
    dist = entry["distribution"]
    builder = _BUILDERS.get(dist.get("type"))
    if builder is None:
        raise ValueError(f"law {law_id!r}: unknown distribution {dist.get('type')!r}")
    return LinearCouplingLaw(
        law_id=law_id,
        version=entry["version"],
        matrices=builder(dist),
        independent=tuple(entry["independent"]),
        citation=_citation_text(entry),
    )
