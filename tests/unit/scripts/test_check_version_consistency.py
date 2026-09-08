"""Tests for the release version consistency guard."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]

pytestmark = pytest.mark.unit
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_version_consistency.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_version_consistency", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_project(root: Path, *, version: str) -> None:
    (root / "src" / "api").mkdir(parents=True)
    (root / "ui").mkdir()
    (root / "rust_core" / "upstream-physics").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "upstream-drift"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "src" / "api" / "_version.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (root / "ui" / "package.json").write_text(
        f'{{"name": "upstream-drift-ui", "version": "{version}"}}\n',
        encoding="utf-8",
    )
    (root / "Cargo.toml").write_text(
        f'[workspace.package]\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "rust_core" / "upstream-physics" / "pyproject.toml").write_text(
        f'[project]\nname = "upstream-physics"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    (root / "ui" / "src-tauri").mkdir(parents=True)
    (root / "ui" / "src-tauri" / "tauri.conf.json").write_text(
        f'{{"productName": "UpstreamDrift", "version": "{version}"}}\n',
        encoding="utf-8",
    )
    (root / "ui" / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "upstream-drift-ui",
                "version": version,
                "lockfileVersion": 3,
                "packages": {"": {"name": "upstream-drift-ui", "version": version}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_sbom_baseline(root, version=version)
    major, minor, _patch = version.split(".")
    _write_security_md(root, supported=f"{major}.{minor}")


def _write_sbom_baseline(root: Path, *, version: str) -> None:
    (root / "scripts" / "config").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "config" / "sbom_baseline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project": "upstream-drift",
                "version": version,
                "tiers": {
                    "core": {"install_spec": f"upstream-drift=={version}"},
                    "full": {"install_spec": f"upstream-drift[all]=={version}"},
                },
                "expected_artifacts": [
                    f"upstream-drift-{version}.cyclonedx.core.json",
                    f"upstream-drift-{version}.spdx.full.json",
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_security_md(root: Path, *, supported: str) -> None:
    (root / "SECURITY.md").write_text(
        "# Security Policy\n\n## Supported Versions\n\n"
        "| Version | Supported          |\n"
        "| ------- | ------------------ |\n"
        f"| {supported}.x   | :white_check_mark: |\n"
        f"| < {supported}   | :x:                |\n",
        encoding="utf-8",
    )


def test_check_versions_passes_when_all_surfaces_match_latest_tag(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.0")

    report = module.check_versions(
        tmp_path,
        tag_reader=lambda _root: ("v1.9.0", "v2.1.0", "pre-rust-migration"),
    )

    assert report.ok is True
    assert report.canonical_version == "2.1.0"
    assert report.latest_tag == "v2.1.0"
    assert report.errors == ()


def test_check_versions_reports_surface_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.0")
    (tmp_path / "ui" / "package.json").write_text(
        '{"name": "upstream-drift-ui", "version": "2.0.0"}\n',
        encoding="utf-8",
    )

    report = module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))

    assert report.ok is False
    assert any("ui/package.json" in error for error in report.errors)


def test_check_versions_reports_tauri_conf_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.1")
    (tmp_path / "ui" / "src-tauri" / "tauri.conf.json").write_text(
        '{"productName": "UpstreamDrift", "version": "2.1.0"}\n',
        encoding="utf-8",
    )

    report = module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))

    assert report.ok is False
    assert any("ui/src-tauri/tauri.conf.json" in error for error in report.errors)


def test_check_versions_reports_version_file_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.1")
    (tmp_path / "VERSION").write_text("2.1.0\n", encoding="utf-8")

    report = module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))

    assert report.ok is False
    assert any(error.startswith("VERSION version") for error in report.errors)


def test_check_versions_reports_sbom_baseline_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.1")
    _write_sbom_baseline(tmp_path, version="2.1.0")

    report = module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))

    assert report.ok is False
    sbom_errors = [e for e in report.errors if "sbom_baseline.json" in e]
    # top-level version + 2 install_spec pins + 2 expected artifact names
    assert len(sbom_errors) == 5


def test_check_versions_reports_security_md_series_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.1")
    _write_security_md(tmp_path, supported="1.0")

    report = module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))

    assert report.ok is False
    assert any("SECURITY.md supported versions" in error for error in report.errors)
    assert any("'2.1.x'" in error for error in report.errors)


def test_check_versions_reports_security_md_without_supported_row(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.1")
    (tmp_path / "SECURITY.md").write_text("# Security Policy\n", encoding="utf-8")

    report = module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))

    assert report.ok is False
    assert any("at least one supported" in error for error in report.errors)


def test_check_versions_reports_missing_semver_tag(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.0")

    report = module.check_versions(
        tmp_path,
        tag_reader=lambda _root: ("pre-rust-migration", "refactor-start-v1"),
    )

    assert report.ok is False
    assert any("No SemVer release tags" in error for error in report.errors)


def test_check_versions_rejects_version_behind_latest_tag(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.0.9")

    report = module.check_versions(
        tmp_path,
        tag_reader=lambda _root: ("v2.0.9", "v2.1.0"),
    )

    assert report.ok is False
    assert any("behind latest release tag" in error for error in report.errors)


def test_check_versions_validates_repo_root_type() -> None:
    module = _load_script_module()

    with pytest.raises(TypeError, match="repo_root"):
        module.check_versions("not-a-path")


def test_check_versions_reports_package_lock_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.1")
    (tmp_path / "ui" / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "upstream-drift-ui",
                "version": "2.1.0",
                "lockfileVersion": 3,
                "packages": {"": {"name": "upstream-drift-ui", "version": "2.1.0"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))

    assert report.ok is False
    assert any("ui/package-lock.json" in error for error in report.errors)


def test_check_versions_reports_package_lock_internal_mismatch(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.1")
    (tmp_path / "ui" / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "upstream-drift-ui",
                "version": "2.1.1",
                "lockfileVersion": 3,
                "packages": {"": {"name": "upstream-drift-ui", "version": "2.1.0"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match top-level version"):
        module.check_versions(tmp_path, tag_reader=lambda _root: ("v2.1.0",))


def test_set_versions_updates_all_surfaces_and_sbom(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.0")

    module.set_versions(tmp_path, "2.2.0")

    report = module.check_versions(
        tmp_path, tag_reader=lambda _root: ("v2.1.0", "v2.2.0")
    )
    assert report.ok is True
    assert report.canonical_version == "2.2.0"
    for surface in report.surfaces:
        assert surface.version == "2.2.0"

    # Verify SBOM baseline was updated
    sbom = json.loads(
        (tmp_path / "scripts" / "config" / "sbom_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert sbom["version"] == "2.2.0"
    assert sbom["tiers"]["core"]["install_spec"] == "upstream-drift==2.2.0"
    assert "2.2.0" in sbom["expected_artifacts"][0]

    # Verify SECURITY.md added 2.2.x series
    sec = (tmp_path / "SECURITY.md").read_text(encoding="utf-8")
    assert "| 2.2.x" in sec


def test_main_set_cli_updates_and_verifies(tmp_path: Path) -> None:
    module = _load_script_module()
    _write_project(tmp_path, version="2.1.0")

    module.set_versions(tmp_path, "2.1.5")
    assert (tmp_path / "VERSION").read_text(encoding="utf-8").strip() == "2.1.5"

    report = module.check_versions(
        tmp_path, tag_reader=lambda _root: ("v2.1.0", "v2.1.5")
    )
    assert report.ok is True
    assert report.canonical_version == "2.1.5"
