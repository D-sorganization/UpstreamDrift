"""Every launcher manifest entry must point at a module that actually exists.

`tool_manifest.yaml` advertised "Rate of Closure Impact Explorer" →
`rate_of_closure.ui.pyqt6.main_window:RateOfClosureMainWindow` for three weeks
while no such module existed in any Tools checkout the launcher searches: the
tool had never landed on Tools `main`, and `vendor/ud-tools` was pinned at a
commit with zero `rate_of_closure` files. Nothing failed, because nothing
checked. Clicking the entry was the only way to find out.

These tests resolve each declared module to a file on disk rather than
importing it. Importing would drag in PyQt6 and every tool's runtime
dependencies, which is both slow and a different failure mode — a missing
third-party package is not the same defect as a manifest pointing at nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.gui_launcher.manifest_loader import load_manifest

pytestmark = pytest.mark.unit

_UD_ROOT = Path(__file__).resolve().parents[2]


def _tools_source_roots() -> list[Path]:
    """Return the Tools source trees the launcher can resolve modules from.

    Mirrors `tools_repo_bridge.load_tools_repo`'s discovery order: a sibling
    `Tools/` checkout first, then the vendored `vendor/ud-tools`. UpstreamDrift's
    own `src/` is deliberately excluded — its presence must not make this sweep
    look satisfiable when no Tools checkout is available at all.
    """
    candidates = [
        _UD_ROOT.parent / "Tools" / "src",
        _UD_ROOT / "vendor" / "ud-tools" / "src",
    ]
    return [path for path in candidates if path.is_dir()]


def _module_index(roots: tuple[Path, ...]) -> frozenset[str]:
    """Index every Python module path under *roots*, as posix path tails.

    Tools does not use one flat import root: `rate_of_closure` sits at
    `src/rate_of_closure/`, while `signal_processing_studio` sits at
    `src/signal_processing_studio/python/signal_processing_studio/`. Matching on
    the path tail resolves both without this test having to encode, and then
    drift from, each tool's private layout.
    """
    tails: set[str] = set()
    for root in roots:
        for path in root.rglob("*.py"):
            if "node_modules" in path.parts or ".venv" in path.parts:
                continue
            posix = path.as_posix()
            tails.add(posix)
            if path.name == "__init__.py":
                tails.add(path.parent.as_posix())
    return frozenset(tails)


_INDEX_CACHE: dict[tuple[Path, ...], frozenset[str]] = {}


def _module_exists(module: str, roots: list[Path]) -> bool:
    key = tuple(roots)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = _module_index(key)
    relative = "/".join(module.split("."))
    suffixes = (f"/{relative}.py", f"/{relative}")
    return any(entry.endswith(suffixes) for entry in _INDEX_CACHE[key])


def _pyqt6_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for tool in load_manifest():
        pyqt6 = tool.get("pyqt6")
        if not isinstance(pyqt6, dict):
            continue
        module = pyqt6.get("module")
        if isinstance(module, str) and module:
            entries.append((str(tool.get("tool_name", "<unnamed>")), module))
    return entries


def test_manifest_declares_at_least_one_pyqt6_entry() -> None:
    """Guard the guard: an empty manifest would make the sweep vacuous."""
    assert _pyqt6_entries(), "tool_manifest.yaml declares no pyqt6 modules"


@pytest.mark.parametrize(
    ("tool_name", "module"),
    _pyqt6_entries(),
    ids=[name for name, _ in _pyqt6_entries()],
)
def test_manifest_pyqt6_module_resolves_to_a_file(tool_name: str, module: str) -> None:
    """Each advertised PyQt6 entry point must exist in a reachable source root."""
    roots = _tools_source_roots()
    if not roots:
        pytest.skip("no source root available to resolve tool modules against")

    assert _module_exists(module, roots), (
        f"launcher manifest advertises {tool_name!r} at {module!r}, "
        f"but no such module exists under any of: "
        f"{', '.join(str(root) for root in roots)}. "
        "Either the tool has not landed yet or vendor/ud-tools is pinned to a "
        "commit that predates it — do not ship a launcher entry for a tool the "
        "user cannot open."
    )
