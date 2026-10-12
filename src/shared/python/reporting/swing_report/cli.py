"""Command line for swing report packages (#12327, epic #11419).

    python -m src.shared.python.reporting.swing_report export \\
        (--report REPORT.json | --results RESULTS.json) --out DIR [--timestamp-utc TS]
    python -m src.shared.python.reporting.swing_report verify DIR

Exit codes: 0 success, 1 contract/package violation, 2 usage or unreadable input.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import sys
from typing import Any, TextIO

from src.shared.python.data_io.provenance import ProvenanceInfo

from .adapters_base import EngineResult, QuantityData
from .builder import build_swing_report
from .contract import Capture, SwingReport
from .exporter import export_package, load_package
from .validator import validate_swing_report

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_VIOLATION, EXIT_USAGE = 0, 1, 2


class InputError(ValueError):
    """Raised when an input file is unreadable or not in the expected shape."""


def _emit(stream: TextIO, message: str) -> None:
    """Write one line to ``stream`` (CLI output; src/ forbids print)."""
    stream.write(message + chr(10))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


def _quantity_data(raw: dict[str, Any]) -> QuantityData:
    return QuantityData(
        units=raw["units"],
        frame=raw["frame"],
        time_s=tuple(raw["time_s"]),
        values=tuple(raw["values"]),
        channel=raw.get("channel"),
    )


def _engine_result(raw: dict[str, Any]) -> EngineResult:
    return EngineResult(
        engine=raw["engine"],
        run_id=raw["run_id"],
        engine_version=raw.get("engine_version"),
        model_file_hash=raw.get("model_file_hash"),
        run_class=raw["run_class"],
        quantities={k: _quantity_data(v) for k, v in raw.get("quantities", {}).items()},
        force_provenance=raw.get("force_provenance"),
        error=raw.get("error"),
    )


def _provenance(timestamp_utc: str | None) -> ProvenanceInfo:
    if timestamp_utc is None:
        return ProvenanceInfo.capture()
    return ProvenanceInfo(timestamp_utc=timestamp_utc, timestamp_local=timestamp_utc)


def report_from_results(path: Path, timestamp_utc: str | None) -> SwingReport:
    """Build a report from a results JSON file (see module docs for the shape)."""
    doc = _read_json(path)
    try:
        capture = Capture(**doc["capture"])
        results = [_engine_result(r) for r in doc["results"]]
        scope = doc["scope"]
    except (KeyError, TypeError) as exc:
        raise InputError(f"{path} is not a results file: {exc!r}") from exc
    return build_swing_report(
        results, capture, scope, provenance=_provenance(timestamp_utc)
    )


def report_from_document(path: Path) -> SwingReport:
    """Validate a contract document and return it."""
    return validate_swing_report(_read_json(path))


def _run_export(args: argparse.Namespace) -> int:
    if args.report is not None:
        report = report_from_document(args.report)
    else:
        report = report_from_results(args.results, args.timestamp_utc)
    digests = export_package(report, args.out)
    _emit(sys.stdout, f"OK: exported {len(digests)} files to {args.out}")
    return EXIT_OK


def _run_verify(args: argparse.Namespace) -> int:
    report = load_package(args.package)
    _emit(sys.stdout, f"OK: {args.package} verified ({report.report_id})")
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swing_report", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write a report package")
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path, help="contract document (JSON)")
    source.add_argument("--results", type=Path, help="engine results (JSON)")
    export.add_argument("--out", type=Path, required=True, help="new package dir")
    export.add_argument("--timestamp-utc", help="fixed timestamp for --results")
    export.set_defaults(run=_run_export)
    verify = sub.add_parser("verify", help="verify and re-validate a package")
    verify.add_argument("package", type=Path)
    verify.set_defaults(run=_run_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = _parser().parse_args(argv)
    try:
        return int(args.run(args))
    except InputError as exc:
        _emit(sys.stderr, f"error: {exc}")
        return EXIT_USAGE
    except ValueError as exc:  # contract, package and precondition violations
        _emit(sys.stderr, f"{exc}")
        return EXIT_VIOLATION
