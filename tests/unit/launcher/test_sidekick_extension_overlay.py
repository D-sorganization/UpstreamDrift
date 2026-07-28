"""Contracts for manifest-gated UpstreamDrift Sidekick extensions."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess  # nosec B404 - fixed interpreter and local import probe
import sys
from types import ModuleType

import pytest

from src.launchers.sidekick_extension_overlay import (
    install_manifest_gated_sidekick_extensions,
    validate_parent_sidekick_runtime,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _write_module(root: Path, relative: str, body: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body or f'ORIGIN = "{root.name}"\n', encoding="utf-8")
    return path


def _write_package_tree(root: Path, package: str) -> None:
    current = root
    for part in package.split("."):
        current /= part
        current.mkdir(parents=True, exist_ok=True)
        (current / "__init__.py").write_text("", encoding="utf-8")


def _write_manifest(path: Path, entries: dict[str, str]) -> None:
    lines = ["schema_version: 1", "paths:"]
    for relative, owner in entries.items():
        lines.extend(
            [
                f"  {relative}:",
                f"    owner: {owner}",
                "    rationale: Test ownership decision.",
                "    tracking_issue: 5623",
                '    review_date: "2026-12-31"',
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_sidekick_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name in {
            "shared",
            "sidekick",
            "src.shared",
        } or name.startswith(
            (
                "shared.python",
                "sidekick.",
                "src.shared.python.sidekick",
            )
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_exact_approved_extension_loads_after_canonical_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    local = tmp_path / "local"
    manifest = tmp_path / "ownership.yaml"
    _write_package_tree(parent, "sidekick.lab.bio")
    _write_package_tree(parent, "shared.python.sidekick.lab.bio")
    _write_module(
        parent,
        "shared/python/sidekick/lab/bio/c3d_reader.py",
        'ORIGIN = "Tools"\n',
    )
    approved = _write_module(
        local,
        "sidekick/lab/bio/force_plate_stitching.py",
        'ORIGIN = "UpstreamDrift"\n',
    )
    _write_manifest(
        manifest,
        {"sidekick/lab/bio/force_plate_stitching.py": "UpstreamDrift"},
    )
    monkeypatch.syspath_prepend(str(parent))
    _clear_sidekick_modules(monkeypatch)

    finder = install_manifest_gated_sidekick_extensions(
        local_python_root=local,
        parent_python_root=parent,
        manifest_path=manifest,
    )
    extension = importlib.import_module("sidekick.lab.bio.force_plate_stitching")
    canonical = importlib.import_module("shared.python.sidekick.lab.bio.c3d_reader")

    assert Path(extension.__file__).resolve() == approved.resolve()
    assert extension.ORIGIN == "UpstreamDrift"
    assert canonical.ORIGIN == "Tools"
    assert str(local / "sidekick") not in list(
        importlib.import_module("sidekick").__path__
    )
    finder.uninstall()
    assert "sidekick.lab.bio.force_plate_stitching" not in sys.modules
    assert "shared.python.sidekick.lab.bio.force_plate_stitching" not in sys.modules
    assert "src.shared.python.sidekick.lab.bio.force_plate_stitching" not in sys.modules


def test_local_only_package_supports_exact_approved_relative_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    local = tmp_path / "local"
    manifest = tmp_path / "ownership.yaml"
    _write_package_tree(parent, "sidekick.process_calculators")
    _write_package_tree(parent, "shared.python.sidekick.process_calculators")
    _write_module(
        local,
        "sidekick/process_calculators/syngas_compression/__init__.py",
        "from .worker import VALUE\n",
    )
    _write_module(
        local,
        "sidekick/process_calculators/syngas_compression/worker.py",
        "VALUE = 42\n",
    )
    _write_manifest(
        manifest,
        {
            "sidekick/process_calculators/syngas_compression/__init__.py": (
                "UpstreamDrift"
            ),
            "sidekick/process_calculators/syngas_compression/worker.py": (
                "UpstreamDrift"
            ),
        },
    )
    monkeypatch.syspath_prepend(str(parent))
    _clear_sidekick_modules(monkeypatch)

    finder = install_manifest_gated_sidekick_extensions(
        local_python_root=local,
        parent_python_root=parent,
        manifest_path=manifest,
    )
    package = importlib.import_module("sidekick.process_calculators.syngas_compression")

    assert package.VALUE == 42
    assert list(package.__path__) == []
    finder.uninstall()


def test_unclassified_local_extension_fails_closed(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    local = tmp_path / "local"
    manifest = tmp_path / "ownership.yaml"
    _write_package_tree(parent, "sidekick.lab.bio")
    _write_module(local, "sidekick/lab/bio/unclassified.py")
    _write_manifest(manifest, {})

    with pytest.raises(RuntimeError, match="ownership inventory mismatch"):
        install_manifest_gated_sidekick_extensions(
            local_python_root=local,
            parent_python_root=parent,
            manifest_path=manifest,
        )


def test_unresolved_ownership_fails_closed(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    local = tmp_path / "local"
    manifest = tmp_path / "ownership.yaml"
    _write_package_tree(parent, "sidekick.lab.bio")
    _write_module(local, "sidekick/lab/bio/undecided.py")
    _write_manifest(
        manifest,
        {"sidekick/lab/bio/undecided.py": "Unresolved"},
    )

    with pytest.raises(RuntimeError, match="must resolve every entry"):
        install_manifest_gated_sidekick_extensions(
            local_python_root=local,
            parent_python_root=parent,
            manifest_path=manifest,
        )


def test_manifest_entry_with_parent_counterpart_fails_closed(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    local = tmp_path / "local"
    manifest = tmp_path / "ownership.yaml"
    relative = "sidekick/lab/bio/c3d_reader.py"
    _write_package_tree(parent, "sidekick.lab.bio")
    _write_module(parent, relative)
    _write_module(local, relative)
    _write_manifest(manifest, {relative: "UpstreamDrift"})

    with pytest.raises(RuntimeError, match="ownership inventory mismatch"):
        install_manifest_gated_sidekick_extensions(
            local_python_root=local,
            parent_python_root=parent,
            manifest_path=manifest,
        )


def test_preloaded_downstream_extension_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    local = tmp_path / "local"
    manifest = tmp_path / "ownership.yaml"
    relative = "sidekick/lab/bio/force_plate_stitching.py"
    _write_package_tree(parent, "sidekick.lab.bio")
    source = _write_module(local, relative)
    _write_manifest(manifest, {relative: "UpstreamDrift"})
    module_name = "sidekick.lab.bio.force_plate_stitching"
    preloaded = ModuleType(module_name)
    preloaded.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module_name, preloaded)

    with pytest.raises(RuntimeError, match="already loaded"):
        install_manifest_gated_sidekick_extensions(
            local_python_root=local,
            parent_python_root=parent,
            manifest_path=manifest,
        )


def test_missing_canonical_parent_runtime_fails_before_startup(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()

    with pytest.raises(RuntimeError, match="canonical Sidekick runtime is incomplete"):
        validate_parent_sidekick_runtime(parent)


def test_real_source_overlay_preserves_parent_and_downstream_authority() -> None:
    """The selected parent and retained Upstream extensions must coexist."""
    repo_root = Path(__file__).resolve().parents[3]
    tools_root = Path(
        os.environ.get("TOOLS_REPO_PATH", repo_root / "vendor/ud-tools")
    ).resolve()
    script = """
from pathlib import Path
from types import SimpleNamespace
import os

import launch_upstream_drift
from src.launchers.launcher_sidekick_sidebar import SidekickSidebarManager

SidekickSidebarManager(SimpleNamespace())._install_sidekick_import_paths()
import shared.python.sidekick.lab.bio.force_plate_stitching as shared_extension
import sidekick.lab.bio.force_plate_stitching as direct_extension
import sidekick.persistence.schema as schema
import sidekick.standalone.runner as runner
import src.api.routes.chat_ws
import src.shared.python.engine_core as engine_core
import src.shared.python.motion_matching as motion_matching
import src.shared.python.sidekick.lab.bio.force_plate_stitching as src_extension

tools_root = Path(os.environ["TOOLS_REPO_PATH"]).resolve()
repo_root = Path.cwd().resolve()
assert direct_extension is shared_extension is src_extension
assert Path(runner.__file__).resolve().is_relative_to(tools_root)
assert Path(schema.__file__).resolve().is_relative_to(tools_root)
assert Path(direct_extension.__file__).resolve().is_relative_to(repo_root)
assert Path(engine_core.__file__).resolve().is_relative_to(repo_root)
assert Path(motion_matching.__file__).resolve().is_relative_to(repo_root)
"""
    env = os.environ.copy()
    env["TOOLS_REPO_PATH"] = str(tools_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(  # nosec B603 - fixed interpreter and inline probe
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
