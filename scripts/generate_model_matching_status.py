"""Generate the model-matching status table from docs/development/model_matching/model_matching_status.json.

MATCH-1 (#14443): one row per physics engine with the best current fit, its
approach and parameters, marker and joint-angle RMS, date, verdict and blocker.
Values not re-checked against a receipt are marked UNVERIFIED in the rendered
table; metrics that were never measured render as "not measured".

Usage:
    python scripts/generate_model_matching_status.py [--write] [--check]
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.contracts import postcondition, precondition  # noqa: E402

FENCE_START = "<!-- generated:model-matching-status -->"
FENCE_END = "<!-- end-generated:model-matching-status -->"
DEFAULT_LEDGER_PATH = (
    REPO_ROOT / "docs" / "development" / "model_matching" / "model_matching_status.json"
)
DEFAULT_STATUS_PATH = (
    REPO_ROOT / "docs" / "development" / "model_matching" / "STATUS.md"
)

REQUIRED_ENGINES = ("mujoco", "opensim", "pinocchio", "drake", "myosuite")
REQUIRED_FIELDS = (
    "engine",
    "model",
    "approach",
    "key_parameters",
    "marker_rms_mm",
    "joint_angle_rms_deg",
    "verified",
    "evidence",
    "date",
    "verdict",
    "blocker",
)
_METRIC_FIELDS = ("marker_rms_mm", "joint_angle_rms_deg")
_COLUMNS = (
    "Engine",
    "Model",
    "Approach",
    "Key Parameters",
    "Marker RMS (mm)",
    "Joint-Angle RMS (deg)",
    "Date",
    "Verdict",
    "Blocker",
    "Evidence",
)


def _validate_entry(entry: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"engine entry missing fields: {', '.join(missing)}")
    if not isinstance(entry["verified"], bool):
        raise ValueError(f"{entry['engine']}: verified must be a boolean")
    try:
        date.fromisoformat(str(entry["date"]))
    except ValueError as exc:
        raise ValueError(f"{entry['engine']}: date must be ISO YYYY-MM-DD") from exc
    for field in _METRIC_FIELDS:
        value = entry[field]
        if value is not None and (isinstance(value, bool) or value < 0):
            raise ValueError(f"{entry['engine']}: {field} must be null or >= 0")


def validate_ledger(ledger: dict[str, Any]) -> None:
    """Raise ValueError unless the ledger is complete and well formed."""
    if not isinstance(ledger.get("engines"), list) or not ledger["engines"]:
        raise ValueError("ledger must contain a non-empty 'engines' list")
    if ledger.get("public_ready") is not False:
        raise ValueError("public_ready must be false until a fit is accepted")
    for entry in ledger["engines"]:
        _validate_entry(entry)


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _metric(value: float | None, verified: bool) -> str:
    if value is None:
        return "not measured"
    text = f"{value:.1f}"
    return text if verified else f"{text} (UNVERIFIED)"


def _row(entry: dict[str, Any]) -> str:
    verified = entry["verified"]
    cells = (
        entry["engine"],
        entry["model"],
        entry["approach"],
        entry["key_parameters"],
        _metric(entry["marker_rms_mm"], verified),
        _metric(entry["joint_angle_rms_deg"], verified),
        entry["date"],
        entry["verdict"],
        entry["blocker"],
        entry["evidence"] if verified else f"{entry['evidence']} (UNVERIFIED)",
    )
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


@precondition(lambda ledger: isinstance(ledger, dict), "ledger must be a dict")
@postcondition(lambda result: result.endswith("\n"), "table must end with newline")
def render_model_matching_status(ledger: dict[str, Any]) -> str:
    """Render the ledger as a Markdown table (one row per engine)."""
    validate_ledger(ledger)
    lines = [
        f"As of {ledger['as_of']}. Public ready: no.",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    lines.extend(_row(entry) for entry in ledger["engines"])
    return "\n".join(lines) + "\n"


def _replace_fenced(text: str, table: str) -> str:
    start = text.index(FENCE_START) + len(FENCE_START)
    end = text.index(FENCE_END)
    return f"{text[:start]}\n{table}{text[end:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    table = render_model_matching_status(ledger)
    current = args.status.read_text(encoding="utf-8")
    updated = _replace_fenced(current, table)
    if args.write:
        args.status.write_text(updated, encoding="utf-8")
        return 0
    if args.check:
        if updated != current:
            print(f"{args.status} is stale; run with --write")  # noqa: T201
            return 1
        return 0
    print(table, end="")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
