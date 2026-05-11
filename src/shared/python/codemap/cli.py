"""CLI entry points for the code-map indexer.

Usage
-----
    python -m shared.python.codemap rebuild [--repo <path>]
    python -m shared.python.codemap search <query> [--kind <kind>] [--limit N]
    python -m shared.python.codemap who-calls <qualified_name>
    python -m shared.python.codemap export --jsonl [--output <file>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _out(msg: str) -> None:
    """Write a message to stdout (CLI output — not application logging)."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _err(msg: str) -> None:
    """Write a message to stderr (CLI error output)."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _cmd_rebuild(args: argparse.Namespace) -> int:
    from .indexer import rebuild

    repo = Path(args.repo).resolve() if args.repo else Path.cwd()
    _out(f"Rebuilding code-map index for: {repo}")
    stats = rebuild(repo)
    _out(
        f"\u2713 {stats.files_processed} files, {stats.symbols_indexed} symbols "
        f"in {stats.duration_s:.2f}s"
    )
    if stats.errors:
        _err(f"  {stats.errors} files had errors (see logs)")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from .api import search

    db = Path(args.db) if args.db else None
    results = search(args.query, limit=args.limit, kind=args.kind, db_path=db)
    if not results:
        _out("No results found.")
        return 1
    for r in results:
        _out(f"  [{r.kind}] {r.qualified_name}  {r.path}:{r.line_start}")
        if r.signature:
            _out(f"           {r.signature}")
    return 0


def _cmd_who_calls(args: argparse.Namespace) -> int:
    from .api import who_calls

    db = Path(args.db) if args.db else None
    results = who_calls(args.name, limit=args.limit, db_path=db)
    if not results:
        _out(f"No callers found for: {args.name}")
        return 1
    for r in results:
        _out(f"  {r.qualified_name}  {r.path}:{r.line_start}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .db import open_db

    db_path = Path(args.db) if args.db else Path(".codemap/index.db")
    if not db_path.exists():
        _err(f"Index not found at {db_path} \u2014 run 'rebuild' first.")
        return 1

    conn = open_db(db_path)
    rows = conn.execute(
        "SELECT kind,qualified_name,path,line_start,line_end,"
        "signature,docstring,imports,calls_out,blake3_hash FROM symbols"
    ).fetchall()
    conn.close()

    from .db import SymbolRow

    import contextlib

    with contextlib.ExitStack() as stack:
        out = (
            stack.enter_context(open(args.output, "w", encoding="utf-8"))
            if args.output
            else sys.stdout
        )
        for r in rows:
            out.write(json.dumps(asdict(SymbolRow(*r))) + "\n")

    _err(f"Exported {len(rows)} symbols.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codemap",
        description="Code-map indexer: tree-sitter + SQLite FTS5 symbol index",
    )
    parser.add_argument("--db", help="Override path to index.db", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    # rebuild
    p_rebuild = sub.add_parser("rebuild", help="Cold rebuild the index")
    p_rebuild.add_argument("--repo", help="Repo root (default: cwd)", default=None)
    p_rebuild.set_defaults(func=_cmd_rebuild)

    # search
    p_search = sub.add_parser("search", help="Full-text search the index")
    p_search.add_argument("query", help="FTS5 query string")
    p_search.add_argument("--kind", help="Filter by symbol kind", default=None)
    p_search.add_argument("--limit", type=int, default=20)
    p_search.set_defaults(func=_cmd_search)

    # who-calls
    p_who = sub.add_parser("who-calls", help="Find callers of a symbol")
    p_who.add_argument("name", help="Qualified name of the callee")
    p_who.add_argument("--limit", type=int, default=20)
    p_who.set_defaults(func=_cmd_who_calls)

    # export
    p_export = sub.add_parser("export", help="Export index as JSONL")
    p_export.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    p_export.add_argument(
        "--output", help="Output file (default: stdout)", default=None
    )
    p_export.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
