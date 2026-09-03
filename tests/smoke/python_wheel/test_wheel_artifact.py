"""Smoke tests for the built Python wheel artifact."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
import venv
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
_WHEEL_ENV = "UPSTREAM_DRIFT_WHEEL"
pytestmark = pytest.mark.smoke


def _wheel_artifact() -> Path:
    """Return the explicit wheel produced by this workflow's build job."""
    raw_path = os.environ.get(_WHEEL_ENV)
    if not raw_path:
        raise AssertionError(
            f"{_WHEEL_ENV} must name the wheel built for this smoke job"
        )
    wheel_path = Path(raw_path).expanduser().resolve()
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise AssertionError(f"{_WHEEL_ENV} is not a wheel artifact: {wheel_path}")
    return wheel_path


def test_wheel_artifact_requires_explicit_build_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke tests must not select a stale wheel by filename ordering."""
    monkeypatch.delenv(_WHEEL_ENV, raising=False)

    with pytest.raises(AssertionError, match=_WHEEL_ENV):
        _wheel_artifact()


def _venv_python(tmp_path: Path) -> Path:
    """Create a clean venv and return its Python executable."""
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    return venv_dir / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )


def _console_script(python_bin: Path) -> Path:
    """Return the platform-specific upstream-drift console script path."""
    if sys.platform == "win32":
        return python_bin.parent / "upstream-drift.exe"
    return python_bin.parent / "upstream-drift"


def _install_wheel(python_bin: Path, *, extra: str | None = None) -> None:
    """Install this workflow's explicit wheel into the target interpreter."""
    wheel_path = _wheel_artifact()
    requirement = str(wheel_path)
    if extra is not None:
        requirement = f"{requirement}[{extra}]"
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", requirement],
        check=True,
    )


def test_wheel_installs_in_clean_venv(tmp_path: Path) -> None:
    """Install the built wheel into a clean venv instead of importing the source tree."""
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin)
    subprocess.run(
        [
            str(python_bin),
            "-c",
            "from src.api._version import __version__; print(__version__)",
        ],
        check=True,
        # Run outside the repo so `src` resolves to the installed package
        # rather than the source tree sitting in the default cwd.
        cwd=str(tmp_path),
    )


def test_canonical_bunkershot_package_imports_from_installed_wheel(
    tmp_path: Path,
) -> None:
    """The canonical package must not depend on the repository's test path."""
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin)
    probe = """
import importlib.util
from bunkershot3d import WrenchTrace as public_trace
from bunkershot3d.postproc.wrench_trace import WrenchTrace as module_trace

assert public_trace is module_trace
assert importlib.util.find_spec("src.bunkershot3d") is None
"""
    subprocess.run(
        [str(python_bin), "-P", "-c", probe],
        check=True,
        cwd=str(tmp_path),
    )


def test_api_server_imports_from_core_only_install(tmp_path: Path) -> None:
    """A no-extras install must be able to import the API application (#8032).

    ``src.api._version`` is a leaf module and passes even when the server's
    dependency declarations are incomplete; importing ``local_server`` is what
    actually exercises the core dependency set.
    """
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin)
    subprocess.run(
        [str(python_bin), "-c", "import src.api.local_server"],
        check=True,
        cwd=str(tmp_path),
    )


def test_wheel_contains_ui_bundle() -> None:
    """The compiled frontend must ship inside the wheel (#8018, #9449).

    ``build_hooks.UIBuildHook._register_ui_bundle`` force-includes ``ui/dist``
    at the install root only when the bundle exists, and downgrades to a log
    warning when it does not. A release that never built the frontend would
    therefore publish a UI-less wheel silently, so the payload is asserted
    here rather than trusted.
    """
    with zipfile.ZipFile(_wheel_artifact()) as wheel:
        names = wheel.namelist()
    assert "ui/dist/index.html" in names, (
        "wheel is missing the compiled UI bundle; "
        f"top-level entries: {sorted({n.split('/')[0] for n in names})}"
    )
    bundle_payload = [
        name for name in names if name.startswith("ui/dist/") and not name.endswith("/")
    ]
    assert len(bundle_payload) > 1, (
        "wheel ships ui/dist/index.html with no bundled payload; "
        f"entries: {sorted(bundle_payload)}"
    )
    assert any(
        name.startswith("ui/dist/assets/") and name.endswith(".js")
        for name in bundle_payload
    ), (
        "wheel is missing the compiled UI JavaScript bundle under "
        f"ui/dist/assets/; entries: {sorted(bundle_payload)}"
    )


def _project_version() -> str:
    """Return the canonical project version from ``pyproject.toml``."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def test_wheel_version_matches_project_and_tag() -> None:
    """The wheel must carry the version of the tag that is being released.

    ``release.yml`` already fails the ``build`` job when the tag and
    ``pyproject.toml`` disagree; this asserts the same identity on the
    artifact that actually reaches PyPI and the GitHub release, so a stale
    downloaded ``dist/`` artifact cannot be published under the wrong tag.
    """
    expected = _project_version()
    wheel_path = _wheel_artifact()
    wheel_version = wheel_path.name.split("-")[1]
    assert wheel_version == expected, (
        f"wheel {wheel_path.name} does not carry pyproject version {expected}"
    )

    with zipfile.ZipFile(wheel_path) as wheel:
        dist_infos = {
            name.split("/", 1)[0]
            for name in wheel.namelist()
            if name.endswith(".dist-info/METADATA")
        }
    assert dist_infos == {f"upstream_drift-{expected}.dist-info"}, (
        f"unexpected dist-info for version {expected}: {sorted(dist_infos)}"
    )

    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    ref_type = os.environ.get("GITHUB_REF_TYPE", "")
    if ref_type == "tag" and ref_name.startswith("v"):
        assert ref_name == f"v{expected}", (
            f"tag {ref_name} does not match the built wheel version {expected}"
        )


def test_wheel_excludes_sidekick_tests() -> None:
    """Test suites must not ship inside the wheel (#8018)."""
    with zipfile.ZipFile(_wheel_artifact()) as wheel:
        names = wheel.namelist()
    shipped_tests = [
        name
        for name in names
        if name.startswith("sidekick/")
        and ("/tests/" in name or Path(name).name.startswith("test_"))
    ]
    assert not shipped_tests, f"wheel ships {len(shipped_tests)} sidekick test files"


def test_console_script_help_runs_from_installed_wheel(tmp_path: Path) -> None:
    """The public console script must be generated and able to render help."""
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin)
    console_script = _console_script(python_bin)

    assert console_script.is_file()
    subprocess.run(
        [str(console_script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_sidekick_uses_one_parent_owned_alias_graph(tmp_path: Path) -> None:
    """Direct and legacy Sidekick/Chat imports must share canonical identities."""
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin, extra="gui-tools")
    probe = """
import src
import chat.chat_dock_widget as direct_chat
import shared.python.chat.chat_dock_widget as canonical_chat
import src.shared.python.chat.chat_dock_widget as legacy_chat
import sidekick.ui.tools_sidebar.sidebar as direct_sidebar
import shared.python.sidekick.ui.tools_sidebar.sidebar as canonical_sidebar
import src.shared.python.sidekick.ui.tools_sidebar.sidebar as legacy_sidebar
import sidekick.standalone.runner
import shared.python.chat_contracts.conversation
import shared.python.notes
import shared.python.theme
import utils.logging_utils

assert src._PARENT_SHARED_ALIASES_INSTALLED is True
assert direct_chat is canonical_chat is legacy_chat
assert direct_sidebar is canonical_sidebar is legacy_sidebar
"""
    subprocess.run(
        [str(python_bin), "-c", probe],
        check=True,
        cwd=str(tmp_path),
    )
    subprocess.run(
        [str(python_bin), "-m", "sidekick", "--help"],
        check=True,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )


def _pinned_tools_blob(relative: str) -> bytes:
    """Read expected bytes directly from the exact superproject gitlink."""
    gitlink = subprocess.run(  # nosec B603 - fixed local Git command
        ["git", "ls-tree", "HEAD", "--", "vendor/ud-tools"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = gitlink.stdout.strip().split(maxsplit=3)
    assert len(fields) >= 3 and fields[:2] == ["160000", "commit"]
    blob = subprocess.run(  # nosec B603 - fixed local Git command
        ["git", "-C", "vendor/ud-tools", "show", f"{fields[2]}:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return blob.stdout


def test_critical_tools_modules_match_exact_pinned_blobs(tmp_path: Path) -> None:
    """Installed critical modules must originate byte-for-byte from Tools."""
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin, extra="gui-tools")
    module_paths = {
        "chat.chat_dock_widget": "src/shared/python/chat/chat_dock_widget.py",
        "sidekick.ui.tools_sidebar.sidebar": (
            "src/shared/python/sidekick/ui/tools_sidebar/sidebar.py"
        ),
    }
    probe = "\n".join(
        (
            """
import hashlib
import importlib
import json
from pathlib import Path

""",
            f"modules = {json.dumps(tuple(module_paths))}",
            """
print(json.dumps({
    name: {
        "origin": str(Path(module.__file__).resolve()),
        "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    }
    for name in modules
    for module in [importlib.import_module(name)]
}))
""",
        )
    )
    result = subprocess.run(  # nosec B603 - fixed venv interpreter and probe
        [str(python_bin), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    installed = json.loads(result.stdout)

    for module_name, relative in module_paths.items():
        expected_digest = hashlib.sha256(_pinned_tools_blob(relative)).hexdigest()
        origin = Path(installed[module_name]["origin"])
        assert "site-packages" in origin.parts
        assert installed[module_name]["sha256"] == expected_digest
