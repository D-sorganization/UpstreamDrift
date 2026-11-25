"""Swing report package catalog and planner (#15723, SRP-4).

The catalog (``package_catalog.json``) lists every item a package can hold:
native-export videos with their options, case grids and phase sheets built
from them. A :class:`PackageRequest` (flags or a request file) starts from
the default package or from nothing, includes, excludes and overrides; the
planner turns it into ordered :class:`Job` s, renders before the composites
that need them. Planning renders nothing, so a dry run is free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any

__all__ = [
    "CATALOG_PATH",
    "KINDS",
    "Catalog",
    "Job",
    "PackageRequest",
    "load_catalog",
    "plan_package",
]

CATALOG_PATH = Path(__file__).with_name("package_catalog.json")
KINDS = ("video", "grid", "phase_sheet")
_COMPOSITES = ("grid", "phase_sheet")


@dataclass(frozen=True)
class Catalog:
    """The checked catalog: shared render options, items and defaults."""

    render_defaults: Mapping[str, Any]
    items: Mapping[str, Mapping[str, Any]]
    default_package: tuple[str, ...]
    views: tuple[str, ...]
    speeds: tuple[float, ...]


def load_catalog(path: Path | None = None) -> Catalog:
    """The package catalog, checked.

    Raises:
        ValueError: an unknown kind, a composite needing an unknown or
            non-video item, or a default naming an unknown item.
    """
    doc = json.loads((path or CATALOG_PATH).read_text(encoding="utf-8"))
    items = doc["items"]
    for name, item in items.items():
        if item.get("kind") not in KINDS or not item.get("title"):
            raise ValueError(
                f"catalog item {name!r} needs a title and a kind in {KINDS}"
            )
        for need in item.get("needs", ()):
            if items.get(need, {}).get("kind") != "video":
                raise ValueError(
                    f"catalog item {name!r} needs {need!r}, not a video item"
                )
    unknown = [n for n in doc["default_package"] if n not in items]
    if unknown:
        raise ValueError(f"default package names unknown items {unknown}")
    return Catalog(
        doc["render_defaults"],
        items,
        tuple(doc["default_package"]),
        tuple(doc["views"]),
        tuple(float(s) for s in doc["speeds"]),
    )


@dataclass(frozen=True)
class PackageRequest:
    """What the user asks for; every field is optional."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    #: start from the catalog's default package (else from nothing)
    default: bool = True
    views: tuple[str, ...] | None = None
    speeds: tuple[float, ...] | None = None
    render_overrides: Mapping[str, Any] = field(default_factory=dict)
    item_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> PackageRequest:
        """A request from a JSON file with this class's field names.

        Raises:
            ValueError: an unknown key (a typo would silently do nothing).
        """
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(doc) - known)
        if unknown:
            raise ValueError(
                f"unknown package request key(s) {unknown}; use {sorted(known)}"
            )
        for key in ("include", "exclude", "views"):
            if key in doc:
                doc[key] = tuple(doc[key])
        if "speeds" in doc:
            doc["speeds"] = tuple(float(s) for s in doc["speeds"])
        return cls(**doc)


@dataclass(frozen=True)
class Job:
    """One unit of package work: an item at a view and a speed."""

    item: str
    kind: str
    title: str
    view: str
    speed: float
    args: Mapping[str, Any] = field(default_factory=dict)
    needs: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.item}@{self.view}@{self.speed:g}x"

    def exporter_argv(self) -> list[str]:
        """``native_viewer_export`` options of a video job (true flags bare)."""
        argv = ["--views", self.view, "--speeds", f"{self.speed:g}"]
        for name, value in self.args.items():
            flag = "--" + name.replace("_", "-")
            if value is True:
                argv.append(flag)
            elif value is not False and value is not None:
                argv.extend([flag, str(value)])
        return argv


def _selected(catalog: Catalog, request: PackageRequest) -> list[str]:
    asked = [*(catalog.default_package if request.default else ()), *request.include]
    unknown = sorted({n for n in (*asked, *request.exclude) if n not in catalog.items})
    if unknown:
        raise ValueError(f"unknown package items {unknown}")
    bad = sorted(set(request.item_overrides) - set(catalog.items))
    if bad:
        raise ValueError(f"item override for unknown items {bad}")
    chosen = [n for n in asked if n not in request.exclude]
    for name in list(chosen):  # a composite pulls in what it is built from
        chosen.extend(catalog.items[name].get("needs", ()))
    return [n for n in catalog.items if n in chosen]  # catalog order, once each


def plan_package(catalog: Catalog, request: PackageRequest) -> list[Job]:
    """The ordered jobs of ``request``: renders first, then composites.

    Postconditions: every job's ``needs`` are keys of earlier jobs.

    Raises:
        ValueError: an unknown item in the request or its overrides.
    """
    names = _selected(catalog, request)
    views = request.views or catalog.views
    speeds = request.speeds or catalog.speeds
    jobs: list[Job] = []
    for name in sorted(names, key=lambda n: catalog.items[n]["kind"] in _COMPOSITES):
        item = catalog.items[name]
        args = {
            **catalog.render_defaults,
            **request.render_overrides,
            **item.get("args", {}),
            **request.item_overrides.get(name, {}),
        }
        for view in views:
            for speed in speeds:
                needs = tuple(f"{n}@{view}@{speed:g}x" for n in item.get("needs", ()))
                job = Job(name, item["kind"], item["title"], view, speed, args, needs)
                jobs.append(job)
    return jobs
