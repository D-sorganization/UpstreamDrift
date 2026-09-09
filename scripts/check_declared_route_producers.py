#!/usr/bin/env python3
"""Fail if a launcher tile declares a web route that no pipeline produces.

Issue #9484: the ``rate_of_closure`` tile advertises ``/tools/impact-explorer``
and ``src/api/local_server.py`` mounts
``vendor/ud-tools/src/rate_of_closure/web/dist`` when it exists, but nothing
built that bundle, so a clean checkout served the honest fallback instead of
the app. This gate enforces the general contract: every launcher-manifest tile
declaring ``web.mode: route`` must have a producer among the GitHub Actions
workflows — either the host UI build (React-served routes) or a workflow step
that builds the route's vendored Tools bundle from the pinned vendor tree with
the exact base-path authority the route's page documents.

Run from the repo root: ``python scripts/check_declared_route_producers.py``
Exit codes: 0 = every declared route has a producer, 1 = at least one is
unproduced or the manifest is malformed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_RELPATH = Path("src/config/launcher_manifest.json")
WORKFLOWS_RELPATH = Path(".github/workflows")

# Producer of every React-served route: the host UI bundle build
# (``frontend-tests`` Build Check in ci-standard.yml, ``build`` in release.yml).
UI_BUILD_DIR = "ui"
UI_BUILD_COMMAND = "npm run build"

# Route-level vendored Tools bundles served by the API under their own mount.
# Each entry pairs the declared launcher route with the bundle directory the
# API mounts and the build command the route's fallback page documents:
# - build-command authority (Tools): vendor/ud-tools/.github/workflows/
#   rate-of-closure-web-distribution.yml (``npm ci`` then a ``vite build``).
# - base-path authority (UpstreamDrift): the /tools/impact-explorer fallback
#   in ui/src/pages/ImpactExplorer.tsx mandates --base=/impact-explorer-app/.


@dataclass(frozen=True)
class VendoredBundle:
    """A vendored bundle a declared route embeds, and how CI must build it."""

    route: str
    bundle_dir: str
    build_command: str


VENDORED_BUNDLES: tuple[VendoredBundle, ...] = (
    VendoredBundle(
        route="/tools/impact-explorer",
        bundle_dir="vendor/ud-tools/src/rate_of_closure/web",
        build_command="npm run build -- --base=/impact-explorer-app/",
    ),
)


class ManifestError(ValueError):
    """Raised when the launcher manifest cannot be audited."""


@dataclass(frozen=True)
class RouteDeclaration:
    """A launcher tile that advertises a web route."""

    tile_id: str
    route: str


@dataclass(frozen=True)
class WorkflowStep:
    """A workflow step that runs a shell command from a working directory."""

    workflow: str
    job: str
    working_directory: str
    command: str


def load_route_declarations(manifest_path: Path) -> list[RouteDeclaration]:
    """Return every tile declaring ``web.mode: route`` from the manifest.

    Args:
        manifest_path: Path to ``launcher_manifest.json``.

    Returns:
        One :class:`RouteDeclaration` per route-declaring tile.

    Raises:
        ManifestError: If the manifest is not a JSON object with a ``tiles``
            list, or a route-declaring tile has no usable route value.
    """
    import json

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(
            f"cannot read launcher manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):
        raise ManifestError(
            f"{manifest_path} is not a launcher manifest (no tiles list)"
        )

    declarations: list[RouteDeclaration] = []
    for tile in manifest["tiles"]:
        if not isinstance(tile, dict):
            raise ManifestError(f"{manifest_path} contains a non-object tile entry")
        web = tile.get("web") or {}
        if not isinstance(web, dict) or web.get("mode") != "route":
            continue
        tile_id = tile.get("id", "<unnamed>")
        route = web.get("route")
        if not isinstance(route, str) or not route.startswith("/"):
            raise ManifestError(
                f"tile {tile_id!r} declares web.mode 'route' without a usable "
                "route path (expected a string starting with '/')"
            )
        declarations.append(RouteDeclaration(tile_id=tile_id, route=route))
    return declarations


def _normalize_working_directory(raw: object) -> str:
    """Return a comparable relative working-directory path."""
    if not isinstance(raw, str) or not raw:
        return "."
    normalized = raw.replace("\\", "/").removeprefix("./").strip("/")
    return normalized or "."


def _workflow_steps(path: Path) -> list[WorkflowStep]:
    """Return every shell-command step of one workflow file."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
        return []

    steps: list[WorkflowStep] = []
    for job_name, job in document["jobs"].items():
        if not isinstance(job, dict):
            continue
        defaults = job.get("defaults") or {}
        job_dir = _normalize_working_directory(
            (defaults.get("run") or {}).get("working-directory")
        )
        for step in job.get("steps") or []:
            if not isinstance(step, dict) or not isinstance(step.get("run"), str):
                continue
            step_dir = step.get("working-directory")
            directory = (
                _normalize_working_directory(step_dir)
                if step_dir is not None
                else job_dir
            )
            steps.append(
                WorkflowStep(
                    workflow=path.name,
                    job=str(job_name),
                    working_directory=directory,
                    command=step["run"],
                )
            )
    return steps


def collect_producer_steps(workflows_dir: Path) -> list[WorkflowStep]:
    """Collect every shell-command step across the repository's workflows.

    Args:
        workflows_dir: Directory containing the GitHub Actions workflows.

    Returns:
        Steps from all ``.yml``/``.yaml`` workflows, in deterministic order.
    """
    steps: list[WorkflowStep] = []
    for path in sorted(workflows_dir.glob("*.y*ml")):
        steps.extend(_workflow_steps(path))
    return steps


def _produces_vendored_bundle(step: WorkflowStep, bundle: VendoredBundle) -> bool:
    """Return True if the step builds the vendored bundle with its authority."""
    return (
        step.working_directory == bundle.bundle_dir
        and bundle.build_command in step.command
    )


def _produces_ui_bundle(step: WorkflowStep) -> bool:
    """Return True if the step is the host UI build that serves React routes."""
    return step.working_directory == UI_BUILD_DIR and UI_BUILD_COMMAND in step.command


def unproduced_routes(
    declarations: list[RouteDeclaration], steps: list[WorkflowStep]
) -> list[str]:
    """Return a failure message for every declared route lacking a producer.

    A route is produced when a workflow step builds its serving bundle: the
    mapped vendored Tools bundle (for routes embedding one) or, for plain
    React routes, the host UI build.
    """
    vendored = {bundle.route: bundle for bundle in VENDORED_BUNDLES}
    problems: list[str] = []
    for declaration in declarations:
        bundle = vendored.get(declaration.route)
        if bundle is not None:
            produced = any(_produces_vendored_bundle(step, bundle) for step in steps)
            expected = (
                f"a workflow step in {bundle.bundle_dir} running "
                f"`{bundle.build_command}`"
            )
        else:
            produced = any(_produces_ui_bundle(step) for step in steps)
            expected = f"a workflow step in {UI_BUILD_DIR} running `{UI_BUILD_COMMAND}`"
        if not produced:
            problems.append(
                f"route {declaration.route} (tile {declaration.tile_id}) declares "
                f"web.mode 'route' but no pipeline produces it: expected {expected}"
            )
    return problems


def main() -> int:
    """Audit every declared web route against the workflow producers."""
    try:
        declarations = load_route_declarations(ROOT / MANIFEST_RELPATH)
    except ManifestError as exc:
        print(f"FAIL: {exc}")
        return 1

    steps = collect_producer_steps(ROOT / WORKFLOWS_RELPATH)
    problems = unproduced_routes(declarations, steps)
    print(f"Auditing {len(declarations)} declared web route(s) for producers")
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        print(f"{len(problems)} declared route(s) have no producing pipeline")
        return 1
    print("All declared web routes have producing pipelines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
