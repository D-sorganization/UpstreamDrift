"""Exact ordered source identities and suffix-safe publication filenames."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    if not clean:
        raise ValueError("figure label has no safe filename characters")
    return f"{clean[:160]}_{hashlib.sha256(value.encode()).hexdigest()[:10]}"


def _validate_id(val: Any, name: str) -> str:
    if val is None:
        raise ValueError(f"{name} is missing")
    if not isinstance(val, str):
        raise TypeError(f"{name} must be a string, got {type(val).__name__}")
    if not val:
        raise ValueError(f"{name} cannot be empty")
    if val.strip() != val:
        raise ValueError(f"{name} contains surrounding whitespace: {val!r}")
    return val


def _source_parts(src: Mapping[str, Any]) -> tuple[Any, Any]:
    p = (
        src.get("source_protocol")
        if src.get("source_protocol") is not None
        else src.get("protocol_id")
    )
    c = src.get("alias_cell") if src.get("alias_cell") is not None else src.get("cell")
    return p, c


def pair_comparison_identity(
    record: Mapping[str, Any],
) -> tuple[str, str, str, str, str, float]:
    """Return deterministic protocol-qualified ordered pair identity."""
    if not isinstance(record, Mapping):
        raise TypeError(f"record must be a Mapping, got {type(record).__name__}")
    comparison = _validate_id(record.get("comparison_kind"), "comparison_kind")
    target_val = record.get("target_x_m")
    if target_val is None:
        raise ValueError("record lacks target_x_m")
    if isinstance(target_val, bool):
        raise TypeError("target_x_m cannot be a boolean")
    if not isinstance(target_val, (int, float)):
        raise TypeError(
            f"target_x_m must be a float or int, got {type(target_val).__name__}"
        )
    if not math.isfinite(target_val):
        raise ValueError(f"target_x_m must be a finite number: {target_val}")
    target_x_m = float(target_val)

    if "first" in record and "second" in record:
        first, second = record["first"], record["second"]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise TypeError("pair 'first' and 'second' must both be mappings")
        p0_r, c0_r = _source_parts(first)
        p1_r, c1_r = _source_parts(second)
    elif "sources" in record:
        sources = record["sources"]
        if not isinstance(sources, (list, tuple)) or len(sources) != 2:
            raise ValueError(
                f"comparison record sources must have exactly two elements: {record}"
            )
        s0, s1 = sources[0], sources[1]
        if not isinstance(s0, Mapping) or not isinstance(s1, Mapping):
            raise TypeError("sources elements must be mappings")
        p0_r, c0_r = _source_parts(s0)
        p1_r, c1_r = _source_parts(s1)
    elif "first_protocol" in record:
        p0_r = record.get("first_protocol")
        c0_r = (
            record.get("first_alias_cell")
            if record.get("first_alias_cell") is not None
            else record.get("first_cell")
        )
        p1_r = record.get("second_protocol")
        c1_r = (
            record.get("second_alias_cell")
            if record.get("second_alias_cell") is not None
            else record.get("second_cell")
        )
    else:
        raise ValueError(
            f"cannot extract ordered pair comparison identity from mapping: {record}"
        )

    return (
        comparison,
        _validate_id(p0_r, "first protocol"),
        _validate_id(c0_r, "first cell"),
        _validate_id(p1_r, "second protocol"),
        _validate_id(c1_r, "second cell"),
        target_x_m,
    )


def pair_comparison_slug(record: Mapping[str, Any]) -> str:
    """Return deterministic, human-readable filesystem slug for a comparison pair."""
    comparison, p0, c0, p1, c1, target_x_m = pair_comparison_identity(record)
    canonical = {
        "comparison_kind": comparison,
        "first": {"cell": c0, "protocol": p0},
        "second": {"cell": c1, "protocol": p1},
        "target_x_m": target_x_m,
    }
    canon_bytes = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(canon_bytes).hexdigest()[:10]
    toks = [
        re.sub(r"[^A-Za-z0-9_-]+", "_", x.replace(".", "p")).strip("_-")
        for x in (comparison, p0, c0, p1, c1)
    ]
    raw = f"{toks[0]}_{toks[1]}_{toks[2]}_vs_{toks[3]}_{toks[4]}_{target_x_m:.3f}m".replace(
        ".", "p"
    )
    prefix = re.sub(r"__+", "_", raw).strip("_-")[:160]
    return f"{prefix}_{digest}"


def validate_planned_figure_identities(
    pairs: Sequence[Mapping[str, Any]],
    cell_refs: Sequence[Mapping[str, Any]],
) -> None:
    """Preflight planned figure outputs to prevent duplicate stems before rendering."""
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise TypeError(f"pairs must be a Sequence, got {type(pairs).__name__}")
    if not isinstance(cell_refs, Sequence) or isinstance(cell_refs, (str, bytes)):
        raise TypeError(f"cell_refs must be a Sequence, got {type(cell_refs).__name__}")

    seen_cells: dict[str, tuple[str, str]] = {}
    seen_cell_ids: set[tuple[str, str]] = set()
    for ref in cell_refs:
        if not isinstance(ref, Mapping):
            raise TypeError(f"cell_ref must be a Mapping, got {type(ref).__name__}")
        cid = (
            _validate_id(ref.get("source_protocol"), "cell source_protocol"),
            _validate_id(ref.get("alias_cell", ref.get("cell")), "cell identifier"),
        )
        if cid in seen_cell_ids:
            raise ValueError(
                f"repeated cell identity in planned outputs: {cid[0]} / {cid[1]}"
            )
        seen_cell_ids.add(cid)
        cstem = _slug(cid[1])
        if cstem in seen_cells:
            raise ValueError(
                f"collision in planned cell stems: {cstem!r} produced by both {seen_cells[cstem]} and {cid}"
            )
        seen_cells[cstem] = cid

    seen_pairs: dict[str, tuple[str, str, str, str, str, float]] = {}
    seen_pair_ids: set[tuple[str, str, str, str, str, float]] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise TypeError(f"pair must be a Mapping, got {type(pair).__name__}")
        pid = pair_comparison_identity(pair)
        if pid in seen_pair_ids:
            raise ValueError(
                f"repeated pair comparison identity in planned outputs: {pid}"
            )
        seen_pair_ids.add(pid)
        pstem = pair_comparison_slug(pair)
        if pstem in seen_pairs:
            raise ValueError(
                f"collision in planned comparison stems: {pstem!r} produced by both {seen_pairs[pstem]} and {pid}"
            )
        seen_pairs[pstem] = pid
