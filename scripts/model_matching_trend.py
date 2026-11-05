"""Append-only model-matching metrics trend and daily-report delta (MATCH-4 #14446).

Each scaling check appends one dated JSON line to
docs/development/model_matching/model_matching_trend.jsonl from the status
ledger; the daily report embeds the delta between the last two records.

Usage:
    python scripts/model_matching_trend.py append [--date YYYY-MM-DD]
    python scripts/model_matching_trend.py delta
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

from scripts.generate_model_matching_status import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    _metric,
    validate_ledger,
)

DEFAULT_TREND_PATH = DEFAULT_LEDGER_PATH.with_name("model_matching_trend.jsonl")
TRACKED_FIELDS = ("marker_rms_mm", "joint_angle_rms_deg", "verified")


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"record date must be ISO YYYY-MM-DD, got {text!r}") from exc


def snapshot(ledger: dict[str, Any], record_date: str) -> dict[str, Any]:
    """Return the dated trend record for ``ledger`` (metrics and verification only)."""
    _iso_date(record_date)
    if not isinstance(ledger.get("engines"), list) or not ledger["engines"]:
        raise ValueError("ledger must contain a non-empty 'engines' list")
    engines = {
        entry["engine"]: {field: entry[field] for field in TRACKED_FIELDS}
        for entry in ledger["engines"]
    }
    return {"date": record_date, "engines": engines}


def read_records(trend_path: Path) -> list[dict[str, Any]]:
    """Return every record in the trend file, oldest first."""
    if not trend_path.exists():
        return []
    lines = trend_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def append_record(trend_path: Path, record: dict[str, Any]) -> None:
    """Append ``record``; its date must be ISO and later than the last record's.

    Postcondition: earlier lines are never rewritten.
    """
    new_date = _iso_date(record["date"])
    records = read_records(trend_path)
    if records and new_date <= _iso_date(records[-1]["date"]):
        raise ValueError(
            f"record date {record['date']} must be after {records[-1]['date']}"
        )
    with trend_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _delta(before: dict[str, Any], after: dict[str, Any]) -> str:
    old, new = before.get("marker_rms_mm"), after.get("marker_rms_mm")
    if old is None or new is None:
        return "n/a"
    return f"{new - old:+.1f}"


def _cell(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "absent"
    return _metric(metrics["marker_rms_mm"], metrics["verified"])


def render_delta(trend_path: Path) -> str:
    """Return the daily-report snippet comparing the last two trend records."""
    records = read_records(trend_path)
    if not records:
        raise ValueError(f"trend file is empty: {trend_path}")
    if len(records) == 1:
        return f"No earlier record to compare with {records[0]['date']}.\n"
    before, after = records[-2], records[-1]
    lines = [
        f"Marker RMS (mm), {before['date']} to {after['date']}:",
        "",
        "| Engine | Previous | Current | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for engine in sorted(set(before["engines"]) | set(after["engines"])):
        old = before["engines"].get(engine, {})
        new = after["engines"].get(engine, {})
        lines.append(f"| {engine} | {_cell(old)} | {_cell(new)} | {_delta(old, new)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Append a record from the ledger, or print the latest delta snippet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("append", "delta"))
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--trend", type=Path, default=DEFAULT_TREND_PATH)
    args = parser.parse_args(argv)
    if args.command == "append":
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        validate_ledger(ledger)
        append_record(args.trend, snapshot(ledger, args.date))
        return 0
    sys.stdout.write(render_delta(args.trend))
    return 0


if __name__ == "__main__":
    sys.exit(main())
