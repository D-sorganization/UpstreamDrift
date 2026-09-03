"""Generate the launcher artifacts from the single tile registry (issue #9412).

``src/config/models.yaml`` is the only hand-edited tile registry. This script
projects it into:

* ``src/config/launcher_manifest.json`` — the web/API catalog (every tile,
  ordered, with the honest ``web`` contract and the readiness dimensions);
* the ``tiles`` arrays of ``src/config/feature_parity.json`` — tile-to-feature
  bindings come from each tile's ``feature_id`` (feature-level facts such as
  status, paths and issues stay in the parity ledger);
* the "Launcher Tiles" table in ``README.md`` between the generated markers.

Usage:
    python -m scripts.registry.generate_registry_artifacts [--check] [--repo-root DIR]

``--check`` writes nothing and exits 1 when any committed artifact differs
from the projection (wired into the ``repo-structure-gates`` job of
``ci-standard.yml``; RM #1507).

Only the standard library and PyYAML are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.tile_registry import (  # noqa: E402
    REGISTRY_PATH,
    TileRecord,
    TileRegistry,
    TileRegistryError,
    load_tile_registry,
)

MANIFEST_RELATIVE = Path("src/config/launcher_manifest.json")
PARITY_RELATIVE = Path("src/config/feature_parity.json")
README_RELATIVE = Path("README.md")
README_BEGIN = "<!-- BEGIN GENERATED: launcher tiles (scripts/registry/generate_registry_artifacts.py) -->"
README_END = "<!-- END GENERATED: launcher tiles -->"

_MATURITY_LABELS = {
    "ready": "ready",
    "beta": "beta",
    "experimental": "experimental",
    "hidden": "hidden",
}


# ---------------------------------------------------------------------------
# Help-page coverage gate (issue #9413)
#
# Every ``ready`` / ``beta`` tile must declare a ``help:`` page that exists on
# disk, so a new tile cannot ship without user-facing help. The gate shipped
# on 2026-09-03 with a shrink-only ratchet of the 19 tiles that were already
# without a page; every one of them now has ``docs/help/<tile_id>.md``, so
# the ratchet is gone and the rule is unconditional.
# ---------------------------------------------------------------------------

#: Maturity levels whose tiles must carry a help page.
HELP_REQUIRED_MATURITIES: frozenset[str] = frozenset({"ready", "beta"})


class ArtifactDriftError(RuntimeError):
    """Raised by ``--check`` when a committed artifact is stale."""


def _canonical_json(document: object) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def render_launcher_manifest(registry: TileRegistry) -> str:
    """The launcher_manifest.json text for ``registry``."""
    return _canonical_json(registry.web_manifest_dict())


def render_feature_parity(registry: TileRegistry, committed: str) -> str:
    """Rewrite the ``tiles`` arrays of the committed parity ledger.

    Feature-level facts (title, status, paths, issue, reason, notes) are the
    ledger's own; only the tile bindings are single-sourced from the registry.
    """
    document = json.loads(committed)
    features = document.get("features")
    if not isinstance(features, dict):
        raise ArtifactDriftError("feature_parity.json must hold a 'features' object")
    bindings = registry.feature_bindings()
    unknown = sorted(set(bindings) - set(features))
    if unknown:
        raise TileRegistryError(
            "models.yaml binds tiles to feature ids missing from "
            f"feature_parity.json: {unknown}"
        )
    for feature_id, entry in features.items():
        entry["tiles"] = bindings.get(feature_id, [])
    return _canonical_json(document)


def _help_cell(tile: TileRecord) -> str:
    if not tile.help:
        return "—"
    return f"[help]({tile.help})"


def render_readme_table(registry: TileRegistry) -> str:
    """Markdown table of visible tiles (between the README markers)."""
    visible = sorted(
        (tile for tile in registry.tiles if tile.visible),
        key=lambda t: (t.order, t.id),
    )
    counts = registry.maturity_counts()
    lines = [
        README_BEGIN,
        "",
        f"{len(visible)} visible tiles from `src/config/models.yaml` "
        f"(maturity: {counts['ready']} ready, {counts['beta']} beta, "
        f"{counts['experimental']} experimental, {counts['hidden']} hidden). "
        "Regenerate with `python -m scripts.registry.generate_registry_artifacts`.",
        "",
        "| Tile | Category | Maturity | Surfaces | Web | Help |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for tile in visible:
        web = tile.web["mode"]
        if web == "route":
            web = f"route `{tile.web['route']}`"
        lines.append(
            f"| {tile.name} (`{tile.id}`) | {tile.category} | "
            f"{_MATURITY_LABELS[tile.maturity]} | {', '.join(tile.surfaces)} | "
            f"{web} | {_help_cell(tile)} |"
        )
    lines += ["", README_END]
    return "\n".join(lines)


def render_readme(registry: TileRegistry, committed: str) -> str:
    """Replace the generated region of README.md (markers must exist)."""
    begin = committed.find(README_BEGIN)
    end = committed.find(README_END)
    if begin == -1 or end == -1 or end < begin:
        raise ArtifactDriftError(
            "README.md is missing the generated launcher-tiles markers "
            f"({README_BEGIN!r} ... {README_END!r})"
        )
    end += len(README_END)
    return committed[:begin] + render_readme_table(registry) + committed[end:]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def project_artifacts(
    repo_root: Path, *, registry_path: Path | None = None
) -> dict[Path, tuple[str, str]]:
    """Return ``{path: (committed_text, rendered_text)}`` for every artifact."""
    registry = load_tile_registry(
        registry_path or (repo_root / REGISTRY_PATH.relative_to(REPO_ROOT))
    )
    renderers: dict[Path, Callable[[str], str]] = {
        repo_root / MANIFEST_RELATIVE: lambda _committed: render_launcher_manifest(
            registry
        ),
        repo_root / PARITY_RELATIVE: lambda committed: render_feature_parity(
            registry, committed
        ),
        repo_root / README_RELATIVE: lambda committed: render_readme(
            registry, committed
        ),
    }
    out: dict[Path, tuple[str, str]] = {}
    for path, renderer in renderers.items():
        committed = _read(path) if path.is_file() else ""
        out[path] = (committed, renderer(committed))
    return out


def help_violations(
    repo_root: Path, *, registry: TileRegistry | None = None
) -> list[str]:
    """Return human-readable help-coverage violations for the registry.

    Two rules are enforced:

    1. A tile that declares ``help:`` must point at a file that exists.
    2. A ``ready`` / ``beta`` tile must declare ``help:``.

    Args:
        repo_root: Repository root that owns ``docs/`` and ``src/config/``.
        registry: Pre-loaded registry; loaded from ``repo_root`` when omitted.

    Returns:
        A list of violation messages. Empty means the gate passes.
    """
    reg = registry or load_tile_registry(
        repo_root / REGISTRY_PATH.relative_to(REPO_ROOT)
    )
    problems: list[str] = []

    for tile in sorted(reg.tiles, key=lambda t: t.id):
        if tile.help:
            if not (repo_root / tile.help).is_file():
                problems.append(
                    f"tile {tile.id!r} declares help {tile.help!r} but that "
                    "file does not exist"
                )
        elif tile.maturity in HELP_REQUIRED_MATURITIES:
            problems.append(
                f"tile {tile.id!r} is {tile.maturity} but declares no help "
                "page; add `help: docs/help/<tile_id>.md` and write the page "
                "(purpose, inputs, outputs, method, limitations)"
            )

    return problems


def check(repo_root: Path) -> list[Path]:
    """Return the artifacts whose committed text differs from the projection."""
    return [
        path
        for path, (committed, rendered) in project_artifacts(repo_root).items()
        if committed != rendered
    ]


def generate(repo_root: Path) -> list[Path]:
    """Write every stale artifact; return the paths that changed."""
    changed: list[Path] = []
    for path, (committed, rendered) in project_artifacts(repo_root).items():
        if committed != rendered:
            _write(path, rendered)
            changed.append(path)
    return changed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 (without writing) when a committed artifact is stale",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.check:
            if problems := help_violations(repo_root):
                sys.stderr.write("registry help coverage failed:\n")
                for problem in problems:
                    sys.stderr.write(f"  - {problem}\n")
                return 1
            stale = check(repo_root)
            if stale:
                names = ", ".join(str(p.relative_to(repo_root)) for p in stale)
                sys.stderr.write(
                    "registry artifacts are stale relative to src/config/models.yaml: "
                    f"{names}\nRun: python -m scripts.registry.generate_registry_artifacts\n"
                )
                return 1
            sys.stdout.write("registry artifacts are up to date\n")
            return 0
        changed = generate(repo_root)
        if changed:
            for path in changed:
                sys.stdout.write(f"wrote {path.relative_to(repo_root)}\n")
        else:
            sys.stdout.write("registry artifacts already up to date\n")
        return 0
    except (TileRegistryError, ArtifactDriftError, FileNotFoundError) as exc:
        sys.stderr.write(f"registry artifact generation refused: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
