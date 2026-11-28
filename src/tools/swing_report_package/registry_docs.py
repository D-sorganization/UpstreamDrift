"""Registry-generated SOPs, semantic glossary, and list/describe CLI (#15734, SRP-7).

Reads directly from :func:`src.tools.swing_report_package.catalog.load_catalog`
with zero hand-maintained item lists, generating deterministic Markdown/JSON
documentation, runnable CLI/Python SOP snippets, and a biomechanical glossary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import sys
from typing import Any

from src.tools.swing_report_package.catalog import Catalog, load_catalog

__all__ = [
    "GlossaryTerm",
    "ItemDescription",
    "RegistryDocumentationBundle",
    "build_registry_documentation_bundle",
    "describe_registry_item",
    "generate_registry_docs_json",
    "generate_registry_docs_markdown",
    "list_registry_items",
    "main",
]


def _require_non_empty(obj: object, *attrs: str) -> None:
    """Validate that required string fields on ``obj`` are non-empty."""
    for attr in attrs:
        if not str(getattr(obj, attr)).strip():
            raise ValueError(f"{type(obj).__name__}.{attr} must be non-empty")


@dataclass(frozen=True)
class GlossaryTerm:
    """One biomechanical or rendering term in the semantic glossary."""

    term: str
    category: str
    definition: str
    units: str = ""
    sign_convention: str = ""
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate required glossary fields."""
        _require_non_empty(self, "term", "category", "definition")

    def to_dict(self) -> dict[str, Any]:
        """Convert this glossary term to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ItemDescription:
    """Complete catalog metadata, runnable SOPs, and caveats for one item."""

    item_id: str
    kind: str
    title: str
    needs: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    views: tuple[str, ...] = ()
    speeds: tuple[float, ...] = ()
    effective_args: Mapping[str, Any] = field(default_factory=dict)
    cli_sop: str = ""
    python_sop: str = ""
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate required item identity fields."""
        _require_non_empty(self, "item_id", "kind", "title")

    def to_dict(self) -> dict[str, Any]:
        """Convert this item description to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RegistryDocumentationBundle:
    """Self-contained documentation snapshot built from the package catalog."""

    schema_version: int
    views: tuple[str, ...]
    speeds: tuple[float, ...]
    packages: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    items: tuple[ItemDescription, ...] = ()
    glossary: tuple[GlossaryTerm, ...] = ()
    render_defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate schema version positivity."""
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert the documentation bundle to a JSON-serializable dictionary."""
        return asdict(self)


_GLOSSARY_ROWS: tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...] = (
    (
        "Golf World Frame (+Z Up) & Camera Views",
        "coordinate_frame",
        "Right-handed +Z up world frame aligned to ball-to-target X-Y; face_on faces chest, down_the_line views from behind along target line.",
        "m",
        "+Z upward; level phases occur at Z = 0 crossings; right-hand rule for torques.",
        (
            "Phase detection fails closed if Z = 0 is not crossed; default size is 1920x1080.",
        ),
    ),
    (
        "ZTCF & ZVCF Counterfactual Split (Drift vs. Control)",
        "decomposition",
        "ZTCF (tau = 0, #0072B2) isolates gravity, momentum drift, and passive forces; ZVCF (#D55E00) isolates applied joint torques alone.",
        "N, N*m",
        "Tip-to-tail superposition equals Total (ZTCF + ZVCF = Total).",
        (
            "Assumes planted feet for ground reactions and a dynamics bundle with efforts.",
        ),
    ),
    (
        "Midhands Equivalent Force and Couple",
        "sign_convention",
        "Reduction of both hands' grip wrenches to midpoint force + couple.",
        "N, N*m",
        "Right-hand-rule axial couple around the midpoint grip axis.",
        ("Omits palm-on-thumb squeeze unless --palm-thumb-force is set.",),
    ),
    (
        "Segment Axial Shading (Tension / Compression)",
        "sign_convention",
        "Color-coded internal axial section load along each body segment.",
        "N",
        "Tension > 0 (#009E73 green); compression < 0 (#CC79A7 purple).",
        ("Clamps symmetrically at shading_limit_n (default 400 N).",),
    ),
    (
        "Playback Speed & Composite Alignment",
        "physical_caveat",
        "Slow-motion speed factor (0.25x) and composite timestamp sync.",
        "x (speed factor), s (swing time)",
        "Physical time_s in manifest.json is invariant to speed.",
        ("Composites fail closed if input clips differ in frame timestamps.",),
    ),
)
_SEMANTIC_GLOSSARY: tuple[GlossaryTerm, ...] = tuple(
    GlossaryTerm(*row) for row in _GLOSSARY_ROWS
)


def _packages_for(item_id: str, catalog: Catalog) -> tuple[str, ...]:
    """Return every package name containing ``item_id`` in canonical order."""
    packages = {"default": catalog.default_package, **catalog.packages}
    return tuple(name for name, members in packages.items() if item_id in members)


def _item_caveats(
    kind: str, needs: tuple[str, ...], args: Mapping[str, Any]
) -> tuple[str, ...]:
    """Build deterministic physical and operational caveats for a catalog item."""
    notes: list[str] = []
    part = str(args.get("force_part") or args.get("ztcf_zvcf_view") or "")
    if part in {"ztcf", "zvcf", "tip_to_tail", "total"}:
        notes.append("ZTCF/ZVCF splits assume planted feet and a dynamics bundle.")
    if args.get("ztcf_zvcf_hands") == "midhands":
        notes.append("Midhands omits grip squeeze unless --palm-thumb-force is set.")
    if args.get("segment_shading"):
        limit = args.get("shading_limit_n", 400)
        notes.append(f"Segment shading clamps tension/compression at +/-{limit} N.")
    if kind in {"grid", "phase_sheet"} and needs:
        notes.append(f"Composite needs matching timestamps for: {', '.join(needs)}.")
    return tuple(notes)


def list_registry_items(catalog: Catalog | None = None) -> tuple[str, ...]:
    """Return all catalog item IDs directly from ``package_catalog.json``."""
    return tuple((catalog or load_catalog()).items.keys())


def describe_registry_item(
    item_id: str, catalog: Catalog | None = None
) -> ItemDescription:
    """Return full metadata, runnable SOP snippets, and caveats for ``item_id``.

    Raises:
        ValueError: if ``item_id`` is empty.
        KeyError: if ``item_id`` is not present in the catalog.
    """
    if not item_id or not item_id.strip():
        raise ValueError("item_id must be a non-empty string")
    cat = catalog or load_catalog()
    if item_id not in cat.items:
        raise KeyError(f"unknown registry item {item_id!r}; known: {list(cat.items)}")
    raw = cat.items[item_id]
    kind, title = str(raw["kind"]), str(raw["title"])
    needs = tuple(str(n) for n in raw.get("needs", ()))
    args = raw.get("args", {})
    effective = {**cat.render_defaults, **args} if kind == "video" else dict(args)
    cli_sop = f"python -m src.tools.swing_report_package --bundle swing.npz --out package --add {item_id}"
    python_sop = (
        "from src.tools.swing_report_package.catalog import PackageRequest, load_catalog, plan_package\n"
        f"jobs = plan_package(load_catalog(), PackageRequest(include=({item_id!r},), default=False))"
    )
    return ItemDescription(
        item_id=item_id,
        kind=kind,
        title=title,
        needs=needs,
        packages=_packages_for(item_id, cat),
        views=cat.views,
        speeds=cat.speeds,
        effective_args=effective,
        cli_sop=cli_sop,
        python_sop=python_sop,
        caveats=_item_caveats(kind, needs, effective),
    )


def build_registry_documentation_bundle(
    catalog: Catalog | None = None,
) -> RegistryDocumentationBundle:
    """Build the complete registry documentation bundle from ``catalog``."""
    cat = catalog or load_catalog()
    return RegistryDocumentationBundle(
        schema_version=1,
        views=cat.views,
        speeds=cat.speeds,
        packages={"default": cat.default_package, **cat.packages},
        items=tuple(describe_registry_item(name, cat) for name in cat.items),
        glossary=_SEMANTIC_GLOSSARY,
        render_defaults=dict(cat.render_defaults),
    )


def _format_item_markdown(item: ItemDescription) -> list[str]:
    """Render one catalog item's SOP section as Markdown lines."""
    lines = [
        f"### `{item.item_id}` ({item.kind})",
        f"- **Title:** {item.title}",
        f"- **Packages:** {', '.join(item.packages) or 'custom only'}",
    ]
    if item.needs:
        lines.append(f"- **Needs:** {', '.join(item.needs)}")
    if item.effective_args:
        lines.append(
            f"- **Effective args:** `{json.dumps(item.effective_args, sort_keys=True)}`"
        )
    lines.extend(["```bash", item.cli_sop, "```", "```python", item.python_sop, "```"])
    lines.extend(f"- **Caveat:** {c}" for c in item.caveats)
    return [*lines, ""]


def generate_registry_docs_markdown(
    catalog: Catalog | None = None,
    bundle: RegistryDocumentationBundle | None = None,
) -> str:
    """Generate deterministic Markdown SOPs and glossary from the catalog."""
    doc = bundle or build_registry_documentation_bundle(catalog)
    speeds = ", ".join(f"{s:g}x" for s in doc.speeds)
    lines = [
        "# Swing Report Package Registry Reference & SOPs",
        f"- **Schema:** {doc.schema_version} | **Views:** {', '.join(doc.views)} | **Speeds:** {speeds}",
        "",
        "## Named Packages",
        *(f"- **`{k}`:** {', '.join(v)}" for k, v in doc.packages.items()),
        "",
        "## Catalog Items & SOPs",
    ]
    for item in doc.items:
        lines.extend(_format_item_markdown(item))
    lines.append("## Biomechanical Semantic Glossary")
    for term in doc.glossary:
        lines.extend([f"### {term.term} (`{term.category}`)", term.definition])
        if term.units:
            lines.append(f"- **Units:** {term.units}")
        if term.sign_convention:
            lines.append(f"- **Sign convention:** {term.sign_convention}")
        lines.extend(f"- **Caveat:** {c}" for c in term.caveats)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _dump_json(payload: Any, indent: int = 2) -> str:
    """Serialize ``payload`` deterministically with trailing newline."""
    return json.dumps(payload, indent=indent, sort_keys=True) + "\n"


def generate_registry_docs_json(
    catalog: Catalog | None = None,
    bundle: RegistryDocumentationBundle | None = None,
    *,
    indent: int = 2,
) -> str:
    """Generate deterministic JSON registry documentation from the catalog."""
    doc = bundle or build_registry_documentation_bundle(catalog)
    return _dump_json(doc.to_dict(), indent=indent)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``list``, ``describe``, and ``docs``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.tools.swing_report_package.registry_docs",
        description="Inspect swing report package items, SOPs, and glossary.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd, fmt in (("list", "json"), ("docs", "markdown")):
        sub.add_parser(cmd).add_argument(
            "--format", choices=("json", "markdown"), default=fmt
        )
    p_desc = sub.add_parser("describe")
    p_desc.add_argument("item_id")
    p_desc.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``list``, ``describe``, and ``docs``."""
    args = _build_parser().parse_args(argv)
    catalog = load_catalog()
    if args.command == "list":
        items = list_registry_items(catalog)
        if args.format == "markdown":
            lines = ["# Swing Report Package Items", *(f"- `{i}`" for i in items)]
            sys.stdout.write("\n".join(lines) + "\n")
            return 0
        pkgs = {"default": list(catalog.default_package)}
        pkgs.update({k: list(v) for k, v in catalog.packages.items()})
        data: dict[str, Any] = {
            "items": list(items),
            "packages": pkgs,
            "views": list(catalog.views),
            "speeds": list(catalog.speeds),
        }
        sys.stdout.write(_dump_json(data))
        return 0
    if args.command == "describe":
        try:
            desc = describe_registry_item(args.item_id, catalog)
        except (KeyError, ValueError) as exc:
            sys.stderr.write(f"registry_docs: {exc}\n")
            return 2
        md = "\n".join(_format_item_markdown(desc))
        sys.stdout.write(
            md if args.format == "markdown" else _dump_json(desc.to_dict())
        )
        return 0
    doc = (
        generate_registry_docs_json(catalog)
        if args.format == "json"
        else generate_registry_docs_markdown(catalog)
    )
    sys.stdout.write(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
