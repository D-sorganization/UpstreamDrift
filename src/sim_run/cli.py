"""Headless simulation JSON batch CLI (Epic #14085 / #14090).

Provides the `ud-sim run` CLI entrypoint:
- Takes simulation case JSON via positional file, --case, or stdin (-).
- Dispatches execution through SimulationRunStore and SimulationWorkerPool.
- Emits RunRecord and ResultEnvelope to results_root/record.json and stdout.
- Returns 0 on success, nonzero on failure or invalid inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

from src.sim_run.models import RunRecord, RunRequest
from src.sim_run.store import RunStoreError, SimulationRunStore
from src.sim_run.worker import SimulationWorkerPool

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return top-level argument parser for ud-sim."""
    parser = argparse.ArgumentParser(
        prog="ud-sim",
        description="Headless simulation batch runner and run service CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    run_parser = subparsers.add_parser(
        "run", help="Execute a headless simulation from case JSON"
    )
    run_parser.add_argument(
        "case",
        nargs="?",
        default=None,
        help="Path to case JSON file or '-' to read from stdin",
    )
    run_parser.add_argument(
        "--case",
        dest="case_opt",
        default=None,
        help="Optional named argument for case JSON file",
    )
    run_parser.add_argument(
        "--results-root",
        default="./sim_results",
        help="Root directory for storing run records and artifacts (default: ./sim_results)",
    )
    run_parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run ID (default: auto-generated)",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Worker timeout in seconds (overrides request timeout)",
    )
    run_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum concurrent worker subprocesses (default: 2)",
    )
    run_parser.add_argument(
        "--format",
        choices=["json", "summary"],
        default="json",
        help="Output format on stdout (default: json)",
    )

    return parser


def load_case_payload(case_arg: str | None, case_opt: str | None) -> dict[str, Any]:
    """Resolve and load raw case JSON payload from file or stdin."""
    target = case_arg if case_arg is not None else case_opt

    if target == "-" or (target is None and not sys.stdin.isatty()):
        raw = sys.stdin.read()
        if not raw.strip():
            raise ValueError("Empty case JSON provided on stdin")
        return json.loads(raw)

    if target is not None:
        p = Path(target)
        if not p.is_file():
            raise FileNotFoundError(f"Case file not found: {target}")
        return json.loads(p.read_text(encoding="utf-8"))

    raise ValueError("No input case specified. Provide a path or '-' for stdin.")


def execute_cli_run(
    request: RunRequest,
    results_root: str | Path,
    run_id: str | None = None,
    timeout: float | None = None,
    max_concurrency: int = 2,
) -> RunRecord:
    """Execute a headless simulation run through store and worker pool."""
    store = SimulationRunStore(results_root=results_root)
    record = store.create_run(request=request, run_id=run_id)
    pool = SimulationWorkerPool(
        store=store,
        max_concurrency=max_concurrency,
        default_timeout_seconds=timeout or 300.0,
    )
    return pool.execute_run(record.run_id, timeout_seconds=timeout)


def run_command(args: argparse.Namespace) -> int:
    """Dispatch the 'run' CLI command."""
    try:
        payload = load_case_payload(args.case, args.case_opt)
        request = RunRequest.model_validate(payload)
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        sys.stderr.write(f"ud-sim run error: {exc}\n")
        return 2

    try:
        completed = execute_cli_run(
            request=request,
            results_root=args.results_root,
            run_id=args.run_id,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
        )
    except (RuntimeError, ValueError, OSError, RunStoreError) as exc:
        sys.stderr.write(f"ud-sim run execution error: {exc}\n")
        return 1

    if args.format == "json":
        sys.stdout.write(completed.model_dump_json(indent=2) + "\n")
    else:
        sys.stdout.write(
            f"Run ID: {completed.run_id}\n"
            f"Status: {completed.status}\n"
            f"Results Root: {args.results_root}\n"
        )

    return 0 if completed.status == "succeeded" else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for ud-sim."""
    parser = build_parser()
    parsed_args = parser.parse_args(argv)

    if parsed_args.command == "run":
        return run_command(parsed_args)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
