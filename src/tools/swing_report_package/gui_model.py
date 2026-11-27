"""Pure logic of the swing report package window (#15724); no Qt. Choices
become command line flags, previewed with the CLI's own parser and planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from src.tools.swing_report_package import cli
from src.tools.swing_report_package.catalog import (
    Catalog,
    package_request,
    plan_package,
)
from src.tools.swing_report_package.runner import plan_manifest

GUI_SETTINGS_PATH = Path(__file__).with_name("gui_settings.json")
_REQUIRED = ("module", "window", "defaults", "bundle_filter", "labels")


@dataclass(frozen=True)
class ReportChoice:
    """What the window holds: files, package, ticked items, swing and club."""

    bundle: str
    out: str
    package: str
    checked: tuple[str, ...]
    swing: str
    club: str


def load_gui_settings(path: Path | None = None) -> dict[str, Any]:
    """The window settings; ``ValueError`` when a required key is missing."""
    doc = json.loads((path or GUI_SETTINGS_PATH).read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED if k not in doc]
    if missing:
        raise ValueError(f"gui settings miss {missing}")
    return dict(doc)


def item_rows(catalog: Catalog) -> list[tuple[str, str, str]]:
    """``(name, kind, title)`` of every catalog item, in catalog order."""
    return [(n, str(i["kind"]), str(i["title"])) for n, i in catalog.items.items()]


def package_names(catalog: Catalog) -> tuple[str, ...]:
    """``default`` then the catalog's named packages."""
    return ("default", *catalog.packages)


def starting_items(catalog: Catalog, package: str) -> tuple[str, ...]:
    """Items ``package`` plans; ``ValueError`` for an unknown package."""
    jobs = plan_package(catalog, package_request(catalog, package))
    return tuple(dict.fromkeys(job.item for job in jobs))


def selection_args(catalog: Catalog, package: str, checked: Sequence[str]) -> list[str]:
    """``--package`` plus the ``--add``/``--drop`` flags giving ``checked``."""
    start = starting_items(catalog, package)
    extra = [n for n in catalog.items if n in checked and n not in start]
    add = [a for n in extra for a in ("--add", n)]
    drop = [a for n in start if n not in checked for a in ("--drop", n)]
    return ["--package", package, *add, *drop]


def cli_args(catalog: Catalog, choice: ReportChoice) -> list[str]:
    """The command line flags of ``choice``; ``ValueError`` without files."""
    if not choice.bundle.strip():
        raise ValueError("choose a swing bundle file")
    if not choice.out.strip():
        raise ValueError("choose an output folder")
    head = ["--bundle", choice.bundle, "--out", choice.out, "--swing", choice.swing]
    head += ["--club", choice.club]
    return head + selection_args(catalog, choice.package, choice.checked)


def plan_preview(catalog: Catalog, choice: ReportChoice) -> dict[str, Any]:
    """The dry-run manifest of ``choice``; needs no bundle, renders nothing."""
    flags = selection_args(catalog, choice.package, choice.checked)
    request = cli.request_from_args(cli.build_parser().parse_args(flags), catalog)
    return plan_manifest(plan_package(catalog, request), swing=choice.swing)


def preview_lines(manifest: Mapping[str, Any]) -> list[str]:
    """A count line, then one ``status  key  title`` line per job."""
    lines = [f"{len(manifest['jobs'])} jobs"]
    for job in manifest["jobs"]:
        why = f"  ({job['error']})" if job.get("error") else ""
        lines.append(f"{job['status']:12s} {job['key']:40s} {job['title']}{why}")
    return lines


def module_command(
    catalog: Catalog, choice: ReportChoice, settings: Mapping[str, Any]
) -> tuple[str, list[str]]:
    """``(program, arguments)`` running the package module on ``choice``."""
    return sys.executable, ["-m", str(settings["module"]), *cli_args(catalog, choice)]
