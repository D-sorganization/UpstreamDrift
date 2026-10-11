"""Read-only loader/view for the LIFT-1 cross-engine baseline receipt (LIFT-8, #11748).

Wraps the committed ``docs/development/lifting/pack_parity_baseline.json``
receipt (schema ``lift-pack-parity-baseline/v1``, produced by
``src/shared/python/lifting/pack_audit``) into a JSON-safe, UI-ready shape for
the PyQt and web lift-viewing surfaces LIFT-8 will add on top of this module.

This module is **engine-free**: it only imports the pure, receipt-in/data-out
``pack_audit`` helpers (``pack_audit.analysis``, ``pack_audit.names`` and the
``SCHEMA`` constant from ``pack_audit.baseline``), never MuJoCo, Drake,
OpenSim or Pinocchio.

"Unavailable is never zero": every scalar measurement is returned as
``{"value": float | None, "reason": str | None}`` so a missing or
non-finite receipt value can never be silently read as a real zero.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.shared.python.data_io.path_utils import get_docs_dir
from src.shared.python.lifting.pack_audit import analysis as _an
from src.shared.python.lifting.pack_audit.baseline import SCHEMA
from src.shared.python.lifting.pack_audit.names import LIFTS

DEFAULT_BASELINE_PATH: Path = (
    get_docs_dir() / "development" / "lifting" / "pack_parity_baseline.json"
)

# Position-metric keys compared, pairwise, between engines for a given pose
# (see pack_audit.baseline._pair_metrics).
_PAIR_METRICS: tuple[str, ...] = (
    "segments_max_m",
    "hands_max_m",
    "feet_max_m",
    "bar_centre_max_m",
    "com_max_m",
    "lifter_com_max_m",
)

# Which of the five lifts a gap *key* can apply to. ``pack_audit.gaps``
# derives each gap from measured rows that are not always lift-labelled in
# their evidence text (e.g. "bench_mass" only ever measures bench_press, but
# its evidence strings list bare masses with no lift suffix). This mirrors
# that derivation's own lift scoping so a per-lift view never shows a gap
# under a lift it was never computed for. This is deliberately coarse (it
# does not try to recover the exact per-(engine, lift) pairs a cross-cutting
# gap was triggered by from its evidence text, which is unreliable); a key
# absent here is treated as applying to every lift that has the gap's engine,
# the fail-open default for an unrecognised key.
_GAP_LIFT_SCOPE: dict[str, tuple[str, ...]] = {
    # pack_audit.gaps._grip() builds its rows from
    # pack_audit.analysis.grip_rows(), which skips "squat" outright.
    "grip_attachment": tuple(lift for lift in LIFTS if lift != "squat"),
    "grip_width": tuple(lift for lift in LIFTS if lift != "squat"),
    # pack_audit.gaps._start() filters to pack_audit.analysis.FLOOR_PULLS.
    "start_pose": _an.FLOOR_PULLS,
    # pack_audit.gaps._mass_contact() computes "bench_mass" from bench_press
    # mass rows only.
    "bench_mass": ("bench_press",),
    # pack_audit.gaps._inertia() reads only receipt["results"]["deadlift"].
    "inertia": ("deadlift",),
}


def _require_receipt(receipt: Any) -> None:
    """Fail fast with a clear message when *receipt* is not a receipt dict."""
    if not isinstance(receipt, dict) or "results" not in receipt:
        raise TypeError(
            "receipt must be a baseline receipt dict with a 'results' key; "
            f"got {type(receipt).__name__}"
        )


def load_baseline(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the committed LIFT-1 cross-engine baseline receipt.

    Args:
        path: Receipt JSON path. Defaults to :data:`DEFAULT_BASELINE_PATH`.

    Returns:
        The parsed receipt dict.

    Postcondition:
        ``result["schema"] == pack_audit.baseline.SCHEMA``.

    Raises:
        FileNotFoundError: If no file exists at *path*.
        ValueError: If the file is not valid JSON, or is valid JSON whose
            ``schema`` field does not match the loader's expected schema.
    """
    receipt_path = Path(path) if path is not None else DEFAULT_BASELINE_PATH
    try:
        text = receipt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"lift baseline receipt not found at {receipt_path}; run "
            "scripts/lifting/run_pack_parity_baseline.py to generate it"
        ) from exc
    try:
        receipt = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"lift baseline receipt at {receipt_path} is not valid JSON: {exc}"
        ) from exc
    schema = receipt.get("schema") if isinstance(receipt, dict) else None
    if schema != SCHEMA:
        raise ValueError(
            f"lift baseline receipt at {receipt_path} has schema {schema!r}; "
            f"expected {SCHEMA!r}"
        )
    return receipt


def available_lifts(receipt: dict[str, Any]) -> list[str]:
    """Lifts present in *receipt*, in canonical LIFT-1 order.

    Raises:
        TypeError: If *receipt* is not a baseline receipt dict.
    """
    _require_receipt(receipt)
    results = receipt["results"]
    return [lift for lift in LIFTS if lift in results]


def _numeric_field(value: Any, *, reason: str | None = None) -> dict[str, Any]:
    """Wrap a scalar receipt value so missing/non-finite is never a bare 0.

    Returns:
        ``{"value": float, "reason": None}`` for a present, finite number,
        or ``{"value": None, "reason": <why>}`` otherwise.
    """
    if value is None:
        return {"value": None, "reason": reason or "not recorded in the receipt"}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return {"value": None, "reason": "not numeric in the receipt"}
    if not math.isfinite(number):
        return {"value": None, "reason": "non-finite value in the receipt"}
    return {"value": number, "reason": None}


def _pair_metric_view(value: Any, tolerance_m: float) -> dict[str, Any]:
    """A cross-engine pair metric, flagged against the position tolerance."""
    field = _numeric_field(value)
    if field["value"] is None:
        status = "unavailable"
    elif field["value"] <= tolerance_m:
        status = "pass"
    else:
        status = "fail"
    return {**field, "status": status, "tolerance_m": tolerance_m}


def _phase_view(phase: dict[str, Any]) -> dict[str, Any]:
    hand_bar = phase.get("summary", {}).get("hand_bar", {})
    return {
        "name": phase.get("name"),
        "fraction": _numeric_field(phase.get("fraction")),
        "n_targets": phase.get("n_targets"),
        "hand_bar_axis_distance_m": {
            side: _numeric_field((hand_bar.get(side) or {}).get("axis_distance_m"))
            for side in ("l", "r")
        },
    }


def _start_contact_view(start_contact: dict[str, Any]) -> dict[str, Any]:
    reason = start_contact.get("reason")
    return {
        "value_n": _numeric_field(start_contact.get("value_n"), reason=reason),
        "non_ground_normal_force_n": _numeric_field(
            start_contact.get("non_ground_normal_force_n"), reason=reason
        ),
        "n_ground_contacts": start_contact.get("n_ground_contacts"),
        "n_non_ground_contacts": start_contact.get("n_non_ground_contacts"),
        "reason": reason,
    }


def _engine_view(receipt: dict[str, Any], lift: str, engine: str) -> dict[str, Any]:
    res = receipt["results"][lift][engine]
    pack = receipt.get("packs", {}).get(engine, {})
    structure = res.get("structure", {})
    start = res.get("start", {})
    smoke = res.get("smoke", {})
    return {
        "engine": engine,
        "pack": {
            "repo": pack.get("repo"),
            "commit": pack.get("commit"),
            "licence": pack.get("licence"),
        },
        "structure": {
            "n_bodies": structure.get("n_bodies"),
            "nq": structure.get("nq"),
            "nv": structure.get("nv"),
        },
        "total_mass_kg": _numeric_field(start.get("total_mass_kg")),
        "bar_above_sole_m": _numeric_field(start.get("bar_above_sole_m")),
        "hand_mid_above_sole_m": _numeric_field(start.get("hand_mid_above_sole_m")),
        "smoke": {
            "loaded": smoke.get("loaded"),
            "stepped": smoke.get("stepped"),
            "max_abs_qvel": _numeric_field(smoke.get("max_abs_qvel")),
        },
        "start_contact": _start_contact_view(res.get("start_contact", {})),
        "phases": [_phase_view(phase) for phase in res.get("phases", [])],
    }


def _comparisons_view(
    receipt: dict[str, Any], lift: str, tolerance_m: float
) -> dict[str, Any]:
    comp = receipt.get("comparisons", {}).get(lift)
    if comp is None:
        return {
            "poses": {},
            "reason": "fewer than two engines available for this lift",
        }
    poses = {
        pose: {
            pair: {
                metric: _pair_metric_view(metrics.get(metric), tolerance_m)
                for metric in _PAIR_METRICS
            }
            for pair, metrics in pairs.items()
        }
        for pose, pairs in comp.get("poses", {}).items()
    }
    return {"poses": poses, "reason": None}


def _gap_applies_to_lift(
    gap: dict[str, Any], lift: str, available_engines: set[str]
) -> bool:
    """Whether a ``receipt["gaps"]`` entry should be shown under *lift*."""
    if not available_engines.intersection(gap.get("engines", ())):
        return False
    scope = _GAP_LIFT_SCOPE.get(gap.get("key", ""), LIFTS)
    return lift in scope


def lift_view(receipt: dict[str, Any], lift: str) -> dict[str, Any]:
    """A JSON-safe, UI-ready summary of *lift* from *receipt*.

    Args:
        receipt: A loaded baseline receipt (see :func:`load_baseline`).
        lift: One of the five canonical lift names (see
            :func:`available_lifts`).

    Returns:
        A dict with:

        - ``lift``: the requested lift name.
        - ``engines``: one summary per engine with a result for *lift*
          (pack commit/licence, structure counts, total mass, bar/hand-mid
          height above sole at start, smoke-test result, start contact
          summary, and the phase list with hand-to-bar-axis distances).
        - ``comparisons``: cross-engine pairwise position metrics per pose,
          each flagged ``"pass"``/``"fail"``/``"unavailable"`` against
          ``tolerances.position_m``.
        - ``gaps``: the entries of ``receipt["gaps"]`` that apply to *lift*.

    Raises:
        TypeError: If *receipt* is not a baseline receipt dict.
        ValueError: If *lift* is not one of the lifts present in *receipt*.
    """
    _require_receipt(receipt)
    lifts = available_lifts(receipt)
    if lift not in lifts:
        raise ValueError(f"unknown lift {lift!r}; valid lifts are {lifts}")
    engines = _an.engines_of(receipt, lift)
    tolerance_m = receipt["tolerances"]["position_m"]
    return {
        "lift": lift,
        "engines": [_engine_view(receipt, lift, engine) for engine in engines],
        "comparisons": _comparisons_view(receipt, lift, tolerance_m),
        "gaps": [
            gap
            for gap in receipt.get("gaps", [])
            if _gap_applies_to_lift(gap, lift, set(engines))
        ],
    }
