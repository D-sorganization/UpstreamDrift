"""Command line of the swing report package (#15724, SRP-5); see README.md.

Selection is the catalog planner's: ``--config`` is a ``PackageRequest``
file, ``--package`` its starting set, ``--add``/``--drop`` extend it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
import sys

from src.tools.swing_report_package.bundle_phases import bundle_phases
from src.tools.swing_report_package.catalog import (
    Catalog,
    PackageRequest,
    load_catalog,
    package_request,
    plan_package,
)
from src.tools.swing_report_package.runner import (
    DiskSpaceError,
    load_runner_settings,
    plan_manifest,
    run_package,
    subprocess_render,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (defaults from ``runner_settings.json``)."""
    defaults = load_runner_settings()["cli_defaults"]
    p = argparse.ArgumentParser(
        prog="python3 -m src.tools.swing_report_package",
        description="Render a swing report package: videos, grids, phase sheets.",
    )
    p.add_argument("--bundle", type=Path, help="kinetic-match swing bundle (.npz)")
    p.add_argument("--out", type=Path, help="package folder (manifest.json)")
    p.add_argument("--package", default=None, help="'default' or a named package")
    p.add_argument("--add", action="append", default=[], metavar="ITEM")
    p.add_argument("--drop", action="append", default=[], metavar="ITEM")
    p.add_argument("--config", type=Path, help="custom package request (JSON)")
    p.add_argument("--swing", default=defaults["swing"])
    p.add_argument("--club", default=defaults["club"])
    p.add_argument("--impact-time", type=float, default=None, help="impact (s)")
    p.add_argument("--list", action="store_true", help="print the catalog")
    p.add_argument("--dry-run", action="store_true", help="print the plan only")
    return p


def _request(args: argparse.Namespace, catalog: Catalog) -> PackageRequest:
    """The package request of the flags; ``ValueError``/``OSError`` if bad."""
    base = PackageRequest.from_file(args.config) if args.config else PackageRequest()
    if args.package is not None:
        start = package_request(catalog, args.package)
        base = replace(
            base, default=start.default, include=(*start.include, *base.include)
        )
    add, drop = (*base.include, *args.add), (*base.exclude, *args.drop)
    return replace(base, include=add, exclude=drop)


def _catalog_text(catalog: Catalog) -> str:
    lines = ["items:"]
    for name, item in catalog.items.items():
        lines.append(f"  {name:22s} [{item['kind']}] {item['title']}")
    lines.append("packages:")
    packages = {"default": catalog.default_package, **catalog.packages}
    for name, items in packages.items():
        lines.append(f"  {name}: " + ", ".join(items))
    speeds = ", ".join(f"{s:g}" for s in catalog.speeds)
    return "\n".join([*lines, f"views: {', '.join(catalog.views)}; speeds: {speeds}\n"])


def _run(args: argparse.Namespace, jobs: list) -> int:
    common = {"impact_time_s": args.impact_time}
    manifest = run_package(
        jobs,
        args.out,
        swing=args.swing,
        render=subprocess_render(args.bundle, args.swing, args.club, **common),
        phases_of=bundle_phases(args.bundle, args.swing, args.club, **common),
    )
    for entry in manifest["jobs"]:
        why = f"  ({entry['error']})" if entry.get("error") else ""
        sys.stdout.write(f"{entry['status']:12s} {entry['key']}{why}\n")
    return int(any(e["status"] == "failed" for e in manifest["jobs"]))


def main(argv: Sequence[str] | None = None) -> int:
    """List, dry-run or run a package; see the module docstring."""
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog = load_catalog()
    if args.list:
        sys.stdout.write(_catalog_text(catalog))
        return 0
    if args.bundle is None or args.out is None:
        parser.error("--bundle and --out are required (or use --list)")
    if not args.bundle.is_file():
        parser.error(f"no bundle file {args.bundle}")
    try:
        jobs = plan_package(catalog, _request(args, catalog))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        sys.stdout.write(json.dumps(plan_manifest(jobs, swing=args.swing)) + "\n")
        return 0
    try:
        return _run(args, jobs)
    except DiskSpaceError as exc:
        sys.stderr.write(f"swing_report_package: {exc}\n")
        return 2
