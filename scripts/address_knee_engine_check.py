"""Cross-engine address knee flexion from the committed receipts (R4, #12684).

Compares each engine's committed ``address.knee_flexion_deg.knees.<side>.model_deg``
(``address_knee.model_knee_flexion_deg`` on that engine's own FK) with the MuJoCo
reference for the same capture, candidate, base spec and foot mode, and reports every
disagreement above ``TOLERANCE_DEG`` per engine and side. Receipts only: no engine SDK,
no regeneration. An engine without a usable receipt is reported ``unavailable`` with
the reason, never as 0. A malformed or non-comparable receipt raises ``ValueError``
(fail closed). Exit codes: 0 agree, 1 disagreement, 2 malformed.
Run: ``python3 -m scripts.address_knee_engine_check``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/ground_support"
#: Largest accepted |engine - mujoco| knee flexion (deg): FK of the same address pose.
TOLERANCE_DEG = 0.1
REFERENCE = "mujoco"
ENGINES: tuple[str, ...] = ("mujoco", "drake", "pinocchio", "opensim", "myosuite")
SIDES: tuple[str, ...] = ("left", "right")
#: Receipt keys that must match the reference for two receipts to be comparable.
PIN_KEYS: tuple[str, ...] = (
    "capture_sha256",
    "candidate_sha256",
    "base_spec_sha256",
    "foot_progression_mode",
)
MAX_FLEXION_DEG = 180.0


@dataclass(frozen=True)
class Case:
    """One capture: ``receipts`` maps each engine to a receipt folder or ``None``."""

    label: str
    receipts: dict[str, str | None]


#: Engines missing from a case have no committed receipt and are reported unavailable.
CASES: tuple[Case, ...] = (
    Case(
        "driver",
        {
            "mujoco": "anthro_driver_seeds",
            "drake": "anthro_driver_drake",
            "pinocchio": "anthro_driver_pinocchio",
            "opensim": "anthro_driver_opensim",
        },
    ),
    Case("iron", {"mujoco": "anthro_iron_seeds_zmp"}),
)


def _load(folder: str, root: Path, engine: str) -> Mapping[str, Any]:
    path = root / folder / "receipt.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{engine}: cannot read receipt {folder}/receipt.json: {exc}"
        ) from exc
    if not isinstance(doc, Mapping):
        raise ValueError(f"{engine}: receipt {folder} must be a JSON object")
    return doc


def _flexion(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= MAX_FLEXION_DEG:
        raise ValueError(
            f"{where} must be finite in [0, {MAX_FLEXION_DEG}] deg, got {value}"
        )
    return value


def read_model_knees(receipt: Mapping[str, Any], engine: str) -> dict[str, float]:
    """Per-side model knee flexion (deg) from ``address.knee_flexion_deg``.

    Raises ``ValueError`` when the receipt is not an object, is made with another
    engine, has no enabled knee block, or lacks a finite flexion in [0, 180] deg.
    """
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{engine}: receipt must be a JSON object")
    made_with = receipt.get("engine") or receipt.get("backend")
    if made_with != engine:
        raise ValueError(f"{engine}: receipt was made with engine {made_with!r}")
    address = receipt.get("address")
    block = address.get("knee_flexion_deg") if isinstance(address, Mapping) else None
    if not isinstance(block, Mapping) or block.get("enabled") is not True:
        raise ValueError(
            f"{engine}: receipt has no enabled address.knee_flexion_deg block"
        )
    knees = block.get("knees")
    if not isinstance(knees, Mapping):
        raise ValueError(f"{engine}: address.knee_flexion_deg.knees missing")
    out: dict[str, float] = {}
    for side in SIDES:
        entry = knees.get(side)
        if not isinstance(entry, Mapping) or "model_deg" not in entry:
            raise ValueError(f"{engine}: knees.{side}.model_deg missing")
        out[side] = _flexion(entry["model_deg"], f"{engine}: knees.{side}.model_deg")
    return out


def unavailable(reason: str) -> dict[str, Any]:
    """An engine entry without a value; ``model_deg`` is ``None``, never 0."""
    if not reason:
        raise ValueError("an unavailable engine needs a reason")
    return {"available": False, "reason": reason, "model_deg": None, "delta_deg": None}


def _check_comparable(
    reference: Mapping[str, Any], other: Mapping[str, Any], engine: str
) -> None:
    for key in PIN_KEYS:
        if key not in reference or key not in other:
            raise ValueError(f"{engine}: receipts must both carry {key}")
        if reference[key] != other[key]:
            raise ValueError(
                f"{engine}: not comparable with the reference, {key} differs"
            )
    ref_mode = reference["address"]["knee_flexion_deg"].get("mode")
    if other["address"]["knee_flexion_deg"].get("mode") != ref_mode:
        raise ValueError(
            f"{engine}: not comparable with the reference, knee mode differs"
        )


def _without_address(
    doc: Mapping[str, Any], engine: str, folder: str
) -> dict[str, Any]:
    reason = doc.get("reason") or doc.get("note") or "receipt carries no address block"
    return unavailable(
        f"{engine} receipt {folder} ({doc.get('status', 'no status')}): {reason}"
    )


def _engine_entry(
    engine: str,
    folder: str | None,
    label: str,
    reference: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    if folder is None:
        return unavailable(f"no {engine} receipt is committed for {label}")
    doc = _load(folder, root, engine)
    if "address" not in doc:
        return _without_address(doc, engine, folder)
    knees = read_model_knees(doc, engine)
    _check_comparable(reference, doc, engine)
    ref = read_model_knees(reference, REFERENCE)
    delta = {s: knees[s] - ref[s] for s in SIDES}
    return {"available": True, "reason": None, "model_deg": knees, "delta_deg": delta}


def evaluate_case(case: Case, root: Path = EVIDENCE) -> dict[str, Any]:
    """Compare every engine of ``case`` with its MuJoCo reference receipt."""
    ref_folder = case.receipts.get(REFERENCE)
    if ref_folder is None:
        raise ValueError(f"case {case.label}: needs a {REFERENCE} reference receipt")
    reference = _load(ref_folder, root, REFERENCE)
    ref_knees = read_model_knees(reference, REFERENCE)
    engines: dict[str, Any] = {
        REFERENCE: {
            "available": True,
            "reason": None,
            "model_deg": ref_knees,
            "delta_deg": dict.fromkeys(SIDES, 0.0),
        }
    }
    disagreements: list[dict[str, Any]] = []
    for engine in ENGINES:
        if engine == REFERENCE:
            continue
        entry = _engine_entry(
            engine, case.receipts.get(engine), case.label, reference, root
        )
        engines[engine] = entry
        for side in SIDES if entry["available"] else ():
            delta = entry["delta_deg"][side]
            if abs(delta) > TOLERANCE_DEG:
                disagreements.append(
                    {
                        "engine": engine,
                        "side": side,
                        "reference_deg": ref_knees[side],
                        "engine_deg": entry["model_deg"][side],
                        "delta_deg": delta,
                        "tolerance_deg": TOLERANCE_DEG,
                    }
                )
    return {
        "label": case.label,
        "engines": engines,
        "disagreements": disagreements,
        "ok": not disagreements,
    }


def build_report(
    root: Path = EVIDENCE, cases: Sequence[Case] = CASES
) -> dict[str, Any]:
    """Evaluate every case; ``ok`` is true when no engine disagrees."""
    results = {c.label: evaluate_case(c, root) for c in cases}
    return {
        "tolerance_deg": TOLERANCE_DEG,
        "reference": REFERENCE,
        "cases": results,
        "ok": all(r["ok"] for r in results.values()),
    }


def format_report(report: Mapping[str, Any]) -> str:
    """Lines per engine (delta or unavailable reason) and a DISAGREE line per failure."""
    lines = []
    for label, case in report["cases"].items():
        for engine, entry in case["engines"].items():
            d = entry["delta_deg"]
            note = f"unavailable, {entry['reason']}" if d is None else f"delta {d}"
            lines.append(f"{label} {engine}: {note}")
        lines += [
            f"DISAGREE {label} {b['engine']} {b['side']}: {b['engine_deg']:.4f} vs "
            f"{b['reference_deg']:.4f} deg (tolerance {b['tolerance_deg']})"
            for b in case["disagreements"]
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; see the module docstring for the exit codes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.evidence_dir, CASES)
    except ValueError as exc:
        print(f"MALFORMED: {exc}")
        return 2
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
