"""Discovery sources for the Sidekick feature catalog."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from .feature_types import FeatureEntry, FeatureKind

DiscoverySource: TypeAlias = Callable[[], list[FeatureEntry]]


def discover_sources() -> tuple[DiscoverySource, ...]:
    """Return ordered feature discovery sources."""
    return (
        _discover_calculators,
        _discover_process_calculators,
        _discover_theme,
        _discover_workflows,
        _discover_subtabs,
    )


def _discover_subtabs() -> list[FeatureEntry]:
    parsed = _parse_help_content()
    if not parsed:
        return []

    out: list[FeatureEntry] = []
    for tab_id, meta in parsed.items():
        title = meta.get("title") or tab_id.replace("_", " ").title()
        summary = meta.get("summary") or f"{title} subtab."
        raw_module = meta.get("source") or "sidekick.ui.tools_sidebar"
        module = _closest_importable(raw_module, fallback="sidekick.ui")
        out.append(
            FeatureEntry(
                feature_id=f"subtab.{tab_id}",
                kind=FeatureKind.SUBTAB.value,
                title=title,
                summary=summary,
                module=module,
                help_anchors=(),
            )
        )
    return out


def _closest_importable(dotted: str, *, fallback: str) -> str:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if _is_importable(candidate):
            return candidate
    if _is_importable(fallback):
        return fallback
    return "sidekick"


def _is_importable(dotted: str) -> bool:
    try:
        importlib.import_module(dotted)
    except Exception:  # noqa: BLE001 - import-time errors are unbounded.
        return False
    return True


def _discover_calculators() -> list[FeatureEntry]:
    return list(
        _walk_package(
            "sidekick.calculators",
            FeatureKind.CALCULATOR.value,
        )
    )


def _discover_process_calculators() -> list[FeatureEntry]:
    return list(
        _walk_package(
            "sidekick.process_calculators",
            FeatureKind.PROCESS_CALCULATOR.value,
        )
    )


def _discover_theme() -> list[FeatureEntry]:
    return [
        FeatureEntry(
            feature_id="theme.sidekick_tokens",
            kind=FeatureKind.THEME.value,
            title="Sidekick design tokens",
            summary=(
                "The canonical color, spacing, and font tokens used by every "
                "Sidekick surface. Both PyQt and React shells consume these."
            ),
            module="sidekick.theme",
            help_anchors=("sidekick/README.md",),
        ),
    ]


def _discover_workflows() -> list[FeatureEntry]:
    try:
        wd = importlib.import_module("shared.python.ai.workflow_definitions")
    except ImportError:
        return []
    registry = getattr(wd, "WORKFLOWS", None) or getattr(wd, "ALL_WORKFLOWS", None)
    if not registry:
        return []
    out: list[FeatureEntry] = []
    for name, workflow in _iter_named(registry):
        title = getattr(workflow, "name", name)
        summary = getattr(workflow, "description", None) or f"Workflow {title}."
        out.append(
            FeatureEntry(
                feature_id=f"workflow.{name}",
                kind=FeatureKind.WORKFLOW.value,
                title=str(title),
                summary=str(summary),
                module="shared.python.ai.workflow_definitions",
                help_anchors=(),
            )
        )
    return out


def _parse_help_content() -> dict[str, dict[str, str]]:
    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    target = (
        repo_root
        / "src"
        / "shared"
        / "python"
        / "sidekick"
        / "ui"
        / "tools_sidebar"
        / "help_content.py"
    )
    if not target.exists():
        return {}
    try:
        tree = ast.parse(target.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    for node in ast.walk(tree):
        target_id: str | None = None
        value: object | None = None
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_id = node.targets[0].id
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_id = node.target.id
            value = node.value
        if target_id != "DEFAULT_SIDEBAR_TAB_HELP":
            continue
        if not isinstance(value, ast.Dict):
            return {}
        return _extract_help_dict(value)
    return {}


def _extract_help_dict(node: ast.Dict) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        if key_node is None or not isinstance(key_node, ast.Constant):
            continue
        if not isinstance(key_node.value, str):
            continue
        out[key_node.value] = _extract_call_meta(value_node)
    return out


def _extract_call_meta(node: object) -> dict[str, str]:
    if not isinstance(node, ast.Call):
        return {}

    meta: dict[str, str] = {}
    if node.args:
        if isinstance(node.args[0], ast.Constant) and isinstance(
            node.args[0].value, str
        ):
            meta["title"] = node.args[0].value
        if (
            len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            meta["summary"] = node.args[1].value
    for kw in node.keywords:
        if (
            kw.arg in {"title", "summary", "source"}
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            meta[kw.arg] = kw.value.value
    return meta


def _walk_package(package_name: str, kind: str) -> Iterator[FeatureEntry]:
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return

    pkg_path = getattr(package, "__path__", None)
    if pkg_path is None:
        return

    for module_info in pkgutil.iter_modules(pkg_path):
        if module_info.name.startswith("_"):
            continue
        full = f"{package_name}.{module_info.name}"
        try:
            mod = importlib.import_module(full)
        except Exception:  # noqa: BLE001 - skip any broken sibling.
            continue
        summary = (
            _module_summary(mod) or f"{module_info.name.replace('_', ' ').title()}"
        )
        title = _module_title(mod) or module_info.name.replace("_", " ").title()
        yield FeatureEntry(
            feature_id=f"{kind}.{module_info.name}",
            kind=kind,
            title=title,
            summary=summary,
            module=full,
            help_anchors=(),
        )


def _module_summary(mod: object) -> str:
    doc = getattr(mod, "__doc__", None)
    if not isinstance(doc, str):
        return ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _module_title(mod: object) -> str:
    summary = _module_summary(mod)
    if not summary:
        return ""
    for sep in (" — ", " - ", ". ", ":"):
        head, _, _ = summary.partition(sep)
        if head and head != summary:
            return head.strip()
    return summary


def _iter_named(registry: object) -> Iterator[tuple[str, object]]:
    if isinstance(registry, Mapping):
        yield from registry.items()
        return
    if isinstance(registry, Sequence):
        for item in registry:
            name = getattr(item, "name", None) or getattr(item, "id", None)
            if name:
                yield str(name), item
