"""Offline weight inspector and lexical CI guardrail (#14503).

Provides spectral analysis, effective rank (Roy & Vetterli 2007), condition
number, dead-unit detection, input channel attribution, and lexical screening.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import require

_BANNED_BASE: tuple[str, ...] = (
    "imped" + "ance",
    "stiff" + "ness",
    "syn" + "ergy",
    "co-con" + "traction",
    "mus" + "cle",
    "ana" + "tomy",
    "control-" + "affine",
)
_SCOPED_Z_TERMS: tuple[str, ...] = ("ZT" + "CF", "ZV" + "CF")


def find_lexical_guardrail_violations(
    text: str,
    scope_issue_13624: bool = False,
) -> list[str]:
    """Find banned overclaim terms in text, respecting exemptions."""
    require(isinstance(text, str), "text must be a string", type(text))
    has_13624 = scope_issue_13624 or ("13624" in text)
    violations: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if (
            "_BANNED_BASE" in line
            or "_SCOPED_Z_TERMS" in line
            or "guardrail-exempt" in line
        ):
            continue
        for term in _BANNED_BASE:
            if re.search(r"\b" + re.escape(term) + r"\b", line, re.IGNORECASE):
                neg = (
                    r"\b(?:not\s+(?:a\s+)?|no\s+|non-|without\s+)"
                    + re.escape(term)
                    + r"\b"
                )
                if not re.search(neg, line, re.IGNORECASE):
                    violations.append(f"line {idx}: banned term '{term}'")
        if not has_13624:
            for z_term in _SCOPED_Z_TERMS:
                if re.search(r"\b" + re.escape(z_term) + r"\b", line, re.IGNORECASE):
                    violations.append(
                        f"line {idx}: banned term '{z_term}' (outside #13624 scope)"
                    )
    return violations


def compute_effective_rank(singular_values: NDArray[np.floating]) -> float:
    """Return Roy & Vetterli (2007) spectral entropy effective rank exp(H(p))."""
    sv = np.asarray(singular_values, dtype=float)
    require(sv.ndim == 1, "singular_values must be 1D", sv.ndim)
    require(bool(np.all(np.isfinite(sv) & (sv >= 0.0))), "valid sv", None)
    total = float(np.sum(sv))
    if total <= 1e-15:
        return 0.0
    p = sv / total
    p_nz = p[p > 1e-15]
    erank = float(np.exp(-np.sum(p_nz * np.log(p_nz))))
    require(0.0 <= erank <= len(sv) + 1e-9, "effective rank within bounds", erank)
    return erank


def compute_matrix_conditioning(singular_values: NDArray[np.floating]) -> float:
    """Return ratio sigma_max / sigma_min; float('inf') if singular/empty."""
    sv = np.asarray(singular_values, dtype=float)
    require(sv.ndim == 1, "singular_values must be 1D", sv.ndim)
    if len(sv) == 0 or sv[-1] <= 1e-15 or sv[0] <= 1e-15:
        return float("inf")
    cond = float(sv[0] / sv[-1])
    require(cond >= 1.0 - 1e-9, "condition number >= 1", cond)
    return cond


def count_dead_units(weight: NDArray[np.floating], threshold: float = 1e-9) -> int:
    """Return count of rows with L2 norm <= threshold."""
    require(weight.ndim == 2 and threshold >= 0.0, "valid weight/threshold", None)
    w = np.asarray(weight, dtype=float)
    dead_count = int(np.sum(np.linalg.norm(w, axis=1) <= threshold))
    require(0 <= dead_count <= w.shape[0], "dead count within row bounds", dead_count)
    return dead_count


def compute_input_attribution(
    weight: NDArray[np.floating],
    slots: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Compute Frobenius norm across column slices for input channels."""
    require(weight.ndim == 2, "weight must be 2D", weight.ndim)
    w = np.asarray(weight, dtype=float)
    in_dim = int(w.shape[1])
    resolved = (
        dict(slots) if slots else {"q": (0, in_dim // 2), "qdot": (in_dim // 2, in_dim)}
    )
    attributions: dict[str, float] = {}
    for name, (start, end) in resolved.items():
        require(
            0 <= start <= end <= in_dim, f"invalid slot {name}", (start, end, in_dim)
        )
        sub = w[:, start:end]
        attributions[name] = float(np.linalg.norm(sub)) if start < end else 0.0
    return attributions


def inspect_layer(
    weight: NDArray[np.floating],
    threshold: float = 1e-9,
    slots: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Inspect single 2D layer weight matrix."""
    require(weight.ndim == 2, "weight must be 2D", weight.ndim)
    w = np.asarray(weight, dtype=float)
    sv = np.linalg.svd(w, compute_uv=False)
    cond = compute_matrix_conditioning(sv)
    return {
        "shape": [int(w.shape[0]), int(w.shape[1])],
        "singular_spectra": [float(x) for x in sv],
        "effective_rank": compute_effective_rank(sv),
        "condition_number": cond if math.isfinite(cond) else "inf",
        "dead_units": count_dead_units(w, threshold=threshold),
        "input_attribution": compute_input_attribution(w, slots=slots),
    }


def compute_state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Return deterministic sha256 digest of state_dict keys and arrays."""
    require(
        isinstance(state_dict, Mapping) and len(state_dict) > 0,
        "non-empty mapping",
        None,
    )
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        val = np.ascontiguousarray(state_dict[key], dtype=np.float64)
        hasher.update(key.encode("utf-8"))
        hasher.update(str(val.shape).encode("utf-8"))
        hasher.update(val.tobytes())
    digest = hasher.hexdigest()
    require(len(digest) == 64, "sha256 must be 64 characters", len(digest))
    return digest


def inspect_policy_weights(
    state_dict: Mapping[str, Any],
    dead_unit_threshold: float = 1e-9,
    slots: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Inspect 2D weight matrices in state_dict and produce structured report."""
    digest = compute_state_dict_sha256(state_dict)
    layers: dict[str, Any] = {}
    total_params, total_dead, eranks = 0, 0, []
    for name in sorted(state_dict.keys()):
        arr = np.asarray(state_dict[name], dtype=float)
        if arr.ndim == 2:
            rep = inspect_layer(arr, threshold=dead_unit_threshold, slots=slots)
            layers[name] = rep
            total_params += int(arr.size)
            total_dead += int(rep["dead_units"])
            eranks.append(float(rep["effective_rank"]))
    require(
        len(layers) > 0, "state_dict must contain at least one 2D layer", len(layers)
    )
    return {
        "schema_version": "1.0.0",
        "provenance": {
            "inspector": "np10_offline_weight_inspector",
            "state_dict_sha256": digest,
            "layer_count": len(layers),
        },
        "layers": layers,
        "summary": {
            "total_parameters": total_params,
            "total_dead_units": total_dead,
            "min_effective_rank": float(min(eranks)),
        },
    }


def export_inspection_json(report: Mapping[str, Any]) -> str:
    """Serialize inspection report to deterministic JSON."""
    require(isinstance(report, Mapping), "report must be a Mapping", type(report))
    return json.dumps(report, indent=2, sort_keys=True)
