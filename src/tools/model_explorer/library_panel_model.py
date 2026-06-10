"""Headless model for the model explorer's unified library panel."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

InfoResolver = Callable[[str, str], Mapping[str, Any] | None]

CATEGORY_ORDER: tuple[str, ...] = (
    "human",
    "golf_clubs",
    "pendulum",
    "robotic",
    "component",
    "discovered",
    "sibling",
    "robot_descriptions",
    "imported",
    "embedded",
)

CATEGORY_LABELS: Mapping[str, str] = {
    "human": "Human Models",
    "golf_clubs": "Golf Clubs",
    "pendulum": "Pendulum Models",
    "robotic": "Robotic Manipulators",
    "component": "Components",
    "discovered": "Repository Models",
    "sibling": "Sibling Repositories",
    "robot_descriptions": "Community Models",
    "imported": "Imported Models",
    "embedded": "Embedded Models",
}

_BADGE_ALIASES: Mapping[str, str] = {
    "embedded": "MJCF",
    "mjcf": "MJCF",
    "xml": "MJCF",
    "mujoco": "MJCF",
    "urdf": "URDF",
    "local": "URDF",
    "local_submodule": "URDF",
    "remote": "URDF",
    "downloadable": "URDF",
    "sdf": "SDF",
    "osim": "OSIM",
    "opensim": "OSIM",
    "opensim-osim": "OSIM",
    "component": "COMPONENT",
}

_CATEGORY_DEFAULT_BADGES: Mapping[str, str] = {
    "human": "URDF",
    "golf_clubs": "URDF",
    "component": "COMPONENT",
    "embedded": "MJCF",
}


@dataclass(frozen=True)
class LibraryModelEntry:
    """One selectable model row in the unified library panel."""

    category: str
    category_label: str
    key: str
    name: str
    format_badge: str
    source_label: str
    description: str
    info: Mapping[str, Any]
    search_text: str


@dataclass(frozen=True)
class LibraryCategoryGroup:
    """A category group plus its visible model rows."""

    category: str
    label: str
    entries: tuple[LibraryModelEntry, ...]


@dataclass(frozen=True)
class LibraryPanelModel:
    """Searchable, category-preserving library panel data."""

    entries: tuple[LibraryModelEntry, ...]

    @classmethod
    def from_library(cls, library: Any) -> LibraryPanelModel:
        """Build panel rows from ``ModelLibrary`` without importing Qt."""
        if library is None:
            raise ValueError("library must be provided")
        return cls.from_listing(library.list_available_models(), library.get_model_info)

    @classmethod
    def from_listing(
        cls,
        listing: Mapping[str, Any],
        info_resolver: InfoResolver,
    ) -> LibraryPanelModel:
        """Build panel rows from a ``ModelLibrary.list_available_models`` snapshot."""
        if listing is None:
            raise ValueError("listing must be provided")
        if info_resolver is None:
            raise ValueError("info_resolver must be provided")

        rows: list[LibraryModelEntry] = []
        for category in _ordered_categories(listing):
            rows.extend(
                _entries_for_category(category, listing[category], info_resolver)
            )
        return cls(tuple(rows))

    def filter_entries(self, query: str) -> tuple[LibraryModelEntry, ...]:
        """Return entries matching all search tokens."""
        if query is None:
            raise ValueError("query must be provided")
        tokens = tuple(token for token in query.lower().split() if token)
        if not tokens:
            return self.entries
        return tuple(
            entry
            for entry in self.entries
            if all(token in entry.search_text for token in tokens)
        )

    def grouped_entries(self, query: str = "") -> tuple[LibraryCategoryGroup, ...]:
        """Return filtered entries grouped in stable category order."""
        visible = self.filter_entries(query)
        groups: list[LibraryCategoryGroup] = []
        for category in CATEGORY_ORDER + tuple(
            sorted({entry.category for entry in visible} - set(CATEGORY_ORDER))
        ):
            entries = tuple(entry for entry in visible if entry.category == category)
            if entries:
                groups.append(
                    LibraryCategoryGroup(
                        category=category,
                        label=CATEGORY_LABELS.get(
                            category, category.replace("_", " ").title()
                        ),
                        entries=entries,
                    )
                )
        return tuple(groups)


def format_badge_for_model(category: str, model_info: Mapping[str, Any]) -> str:
    """Return the compact format badge shown beside a library model."""
    if not category:
        raise ValueError("category must be provided")
    if model_info is None:
        raise ValueError("model_info must be provided")

    explicit = _first_text(model_info, "format", "source_format", "type")
    if explicit:
        badge = _BADGE_ALIASES.get(explicit.lower())
        if badge:
            return badge
        return explicit.upper()
    return _CATEGORY_DEFAULT_BADGES.get(category, "MODEL")


def _ordered_categories(listing: Mapping[str, Any]) -> tuple[str, ...]:
    known = tuple(category for category in CATEGORY_ORDER if category in listing)
    unknown = tuple(sorted(set(listing) - set(CATEGORY_ORDER)))
    return known + unknown


def _entries_for_category(
    category: str,
    items: Any,
    info_resolver: InfoResolver,
) -> list[LibraryModelEntry]:
    rows: list[LibraryModelEntry] = []
    for key, info in _iter_model_infos(category, items, info_resolver):
        if info is None:
            continue
        rows.append(_make_entry(category, key, info))
    return rows


def _iter_model_infos(
    category: str,
    items: Any,
    info_resolver: InfoResolver,
) -> Sequence[tuple[str, Mapping[str, Any] | None]]:
    if isinstance(items, Mapping):
        return tuple((str(key), _as_mapping(value)) for key, value in items.items())
    if isinstance(items, list | tuple):
        rows: list[tuple[str, Mapping[str, Any] | None]] = []
        for item in items:
            if isinstance(item, str):
                rows.append((item, info_resolver(category, item)))
            elif isinstance(item, Mapping):
                key = str(
                    item.get("config_key")
                    or item.get("key")
                    or item.get("name")
                    or item.get("path")
                    or ""
                )
                if key:
                    rows.append((key, item))
        return tuple(rows)
    return ()


def _make_entry(
    category: str,
    key: str,
    info: Mapping[str, Any],
) -> LibraryModelEntry:
    category_label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())
    name = _first_text(info, "name") or key
    badge = format_badge_for_model(category, info)
    source = _source_label(info)
    description = _first_text(info, "description") or ""
    search_text = " ".join(
        value.lower()
        for value in (
            category,
            category_label,
            key,
            name,
            badge,
            source,
            description,
            _first_text(info, "repo"),
            _first_text(info, "package"),
            _first_text(info, "path"),
        )
        if value
    )
    return LibraryModelEntry(
        category=category,
        category_label=category_label,
        key=key,
        name=name,
        format_badge=badge,
        source_label=source,
        description=description,
        info=info,
        search_text=search_text,
    )


def _source_label(info: Mapping[str, Any]) -> str:
    return (
        _first_text(info, "repo")
        or _first_text(info, "package")
        or _first_text(info, "path")
        or _first_text(info, "description")
        or ""
    )


def _first_text(info: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = info.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None
