import os
import sys
from unittest.mock import MagicMock, call, patch


class DummyHookInterface:
    def __init__(self, root, config):
        self.root = root
        self.config = config


sys.modules["hatchling"] = MagicMock()
sys.modules["hatchling.builders"] = MagicMock()
sys.modules["hatchling.builders.hooks"] = MagicMock()
sys.modules["hatchling.builders.hooks.plugin"] = MagicMock()
sys.modules["hatchling.builders.hooks.plugin.interface"] = MagicMock()
sys.modules[
    "hatchling.builders.hooks.plugin.interface"
].BuildHookInterface = DummyHookInterface  # type: ignore

import subprocess  # noqa: E402

import build_hooks  # noqa: E402
import pytest  # noqa: E402
from scripts.packaging.pinned_tools_provenance import (  # noqa: E402
    compute_tools_source_sha256,
)

pytestmark = pytest.mark.unit

_PINNED_TOOLS_SHA = "a" * 40


class DummyConfig:
    def __init__(self, root, config=None):
        self.root = root
        self.config = config or {}


@pytest.fixture(autouse=True)
def _canonical_tools_package_roots(tmp_path) -> None:
    """Model the pinned Tools submodule present in a normal checkout."""
    tools_src = tmp_path / "vendor" / "ud-tools" / "src"
    canonical_python = tools_src / "shared" / "python"
    for package_name in ("sidekick", "chat", "chat_contracts"):
        (canonical_python / package_name).mkdir(parents=True)
    for package_name in ("sidekick", "chat"):
        (tools_src / package_name).mkdir(parents=True)
    (tools_src / "python" / "src" / "utils").mkdir(parents=True)
    (tools_src / "contracts.py").write_text("canonical", encoding="utf-8")
    _write_ownership_manifest(
        tmp_path, {"sidekick/fixture_extension.py": "UpstreamDrift"}
    )


def _write_package_file(root, relative: str, content: str = "canonical") -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_ownership_manifest(root, owners: dict[str, str]) -> None:
    """Create the constrained manifest shape consumed by the build hook."""
    path = root / "scripts" / "config" / "shared_python_ownership_exceptions.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["schema_version: 1", "paths:"]
    for relative, owner in owners.items():
        lines.extend(
            [
                f"  {relative}:",
                f"    owner: {owner}",
                "    rationale: test fixture",
                "    tracking_issue: 1",
                '    review_date: "2099-01-01"',
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _registered_packages(hook) -> dict[str, str]:
    build_data: dict = {}
    with patch(
        "build_hooks.subprocess.run",
        side_effect=_matching_tools_git_results(),
    ):
        hook._register_tools_packages("1.0.0", build_data)
    return build_data["force_include"]


def _initialize_with_pinned_tools(hook, version: str, build_data: dict) -> None:
    with patch.object(
        build_hooks.UIBuildHook,
        "_validate_pinned_tools_checkout",
        return_value=True,
    ):
        hook.initialize(version, build_data)


def _matching_tools_git_results() -> list[subprocess.CompletedProcess]:
    gitlink_args = ["git", "ls-tree", "HEAD", "--", "vendor/ud-tools"]
    checkout_args = ["git", "-C", "vendor/ud-tools", "rev-parse", "HEAD"]
    status_args = [
        "git",
        "-C",
        "vendor/ud-tools",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "src/shared",
        "src/sidekick",
        "src/chat",
        "src/python/src/utils",
        "src/contracts.py",
    ]
    return [
        subprocess.CompletedProcess(
            gitlink_args,
            0,
            stdout=(f"160000 commit {_PINNED_TOOLS_SHA}\tvendor/ud-tools\n"),
            stderr="",
        ),
        subprocess.CompletedProcess(
            checkout_args,
            0,
            stdout=f"{_PINNED_TOOLS_SHA}\n",
            stderr="",
        ),
        subprocess.CompletedProcess(status_args, 0, stdout="", stderr=""),
    ]


def test_canonical_chat_and_sidekick_packages_are_registered(tmp_path) -> None:
    tools_src = tmp_path / "vendor" / "ud-tools" / "src"
    canonical_python = tools_src / "shared" / "python"
    _write_package_file(canonical_python, "sidekick/__init__.py")
    _write_package_file(canonical_python, "chat/__init__.py")
    _write_package_file(tools_src, "sidekick/__init__.py", "sidekick shim")
    _write_package_file(tools_src, "chat/__init__.py", "chat shim")

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert registered[str(canonical_python / "sidekick" / "__init__.py")] == (
        "shared/python/sidekick/__init__.py"
    )
    assert registered[str(canonical_python / "chat" / "__init__.py")] == (
        "shared/python/chat/__init__.py"
    )
    assert registered[str(tools_src / "sidekick" / "__init__.py")] == (
        "sidekick/__init__.py"
    )
    assert registered[str(tools_src / "chat" / "__init__.py")] == "chat/__init__.py"


def test_canonical_chat_contracts_dependency_is_registered(tmp_path) -> None:
    canonical_python = tmp_path / "vendor" / "ud-tools" / "src" / "shared" / "python"
    _write_package_file(canonical_python, "chat_contracts/conversation.py")

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert (
        registered[str(canonical_python / "chat_contracts" / "conversation.py")]
        == "shared/python/chat_contracts/conversation.py"
    )


def test_canonical_tools_dependency_closure_is_registered(tmp_path) -> None:
    tools_src = tmp_path / "vendor" / "ud-tools" / "src"
    _write_package_file(
        tools_src,
        "shared/python/import_aliases.py",
        "canonical aliases",
    )
    _write_package_file(tools_src, "shared/python/theme/__init__.py")
    _write_package_file(tools_src, "python/src/utils/logging_utils.py")

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert (
        registered[str(tools_src / "shared" / "python" / "import_aliases.py")]
        == "shared/python/import_aliases.py"
    )
    assert (
        registered[str(tools_src / "shared" / "python" / "theme" / "__init__.py")]
        == "shared/python/theme/__init__.py"
    )
    assert (
        registered[str(tools_src / "python" / "src" / "utils" / "logging_utils.py")]
        == "utils/logging_utils.py"
    )
    assert registered[str(tools_src / "contracts.py")] == "contracts.py"


def test_local_canonical_counterpart_is_ignored(tmp_path) -> None:
    canonical_python = tmp_path / "vendor" / "ud-tools" / "src" / "shared" / "python"
    local_python = tmp_path / "src" / "shared" / "python"
    relative = "sidekick/agent/core.py"
    _write_package_file(canonical_python, relative, "canonical")
    _write_package_file(local_python, relative, "stale local drift")

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert registered[str(canonical_python / relative)] == (f"shared/python/{relative}")
    assert str(local_python / relative) not in registered


def test_local_only_extension_is_included(tmp_path) -> None:
    local_python = tmp_path / "src" / "shared" / "python"
    relative = "sidekick/upstream_extension.py"
    _write_package_file(local_python, relative, "Upstream-only extension")
    _write_ownership_manifest(tmp_path, {relative: "UpstreamDrift"})

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert registered[str(local_python / relative)] == (f"shared/python/{relative}")


def test_unresolved_local_extension_is_not_force_included(tmp_path) -> None:
    """Unresolved production paths cannot silently enter the parent package."""
    local_python = tmp_path / "src" / "shared" / "python"
    relative = "sidekick/pending_owner.py"
    _write_package_file(local_python, relative, "needs ownership")
    _write_ownership_manifest(tmp_path, {relative: "Unresolved"})

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert str(local_python / relative) not in registered


def test_local_non_python_artifact_is_not_force_included(tmp_path) -> None:
    """Only manifest-governed source modules may extend a Tools package."""
    local_python = tmp_path / "src" / "shared" / "python"
    relative = "sidekick/process_calculators/runtime_requirements.txt"
    _write_package_file(local_python, relative, "not package source")

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert str(local_python / relative) not in registered


def test_unclassified_local_extension_fails_closed(tmp_path) -> None:
    """A local-only production path needs an explicit manifest decision."""
    local_python = tmp_path / "src" / "shared" / "python"
    _write_package_file(local_python, "chat/unclassified_extension.py")

    with pytest.raises(RuntimeError, match="lacks ownership classification"):
        _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))


def test_tools_package_registration_excludes_test_artifacts(tmp_path) -> None:
    canonical_python = tmp_path / "vendor" / "ud-tools" / "src" / "shared" / "python"
    local_python = tmp_path / "src" / "shared" / "python"
    artifacts = (
        canonical_python / "sidekick" / "tests" / "test_hidden.py",
        canonical_python / "chat" / "test_protocol.py",
        local_python / "sidekick" / "__pycache__" / "extension.pyc",
        local_python / "chat" / "tests" / "test_local.py",
    )
    for artifact in artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("must not ship", encoding="utf-8")
    _write_package_file(canonical_python, "sidekick/runtime.py")

    registered = _registered_packages(build_hooks.UIBuildHook(str(tmp_path), {}))

    assert registered[str(canonical_python / "sidekick" / "runtime.py")] == (
        "shared/python/sidekick/runtime.py"
    )
    assert all(str(artifact) not in registered for artifact in artifacts)


def test_release_build_missing_canonical_vendor_fails_closed(tmp_path) -> None:
    missing_checkout = tmp_path / "missing-checkout"
    hook = build_hooks.UIBuildHook(str(missing_checkout), {})

    with pytest.raises(RuntimeError, match="Pinned Tools package roots") as error:
        hook._register_tools_packages("1.0.0", {})

    assert "sidekick" in str(error.value)
    assert "chat" in str(error.value)
    assert "shared" in str(error.value)
    assert "python/src/utils" in str(error.value)
    assert "contracts.py" in str(error.value)


def test_pinned_tools_validation_uses_fixed_git_commands(tmp_path) -> None:
    hook = build_hooks.UIBuildHook(str(tmp_path), {})

    with patch(
        "build_hooks.subprocess.run",
        side_effect=_matching_tools_git_results(),
    ) as run:
        hook._register_tools_packages("1.0.0", {})

    subprocess_contract = {
        "cwd": str(tmp_path),
        "check": True,
        "capture_output": True,
        "text": True,
    }
    assert run.call_args_list == [
        call(
            ["git", "ls-tree", "HEAD", "--", "vendor/ud-tools"],
            **subprocess_contract,
        ),
        call(
            ["git", "-C", "vendor/ud-tools", "rev-parse", "HEAD"],
            **subprocess_contract,
        ),
        call(
            [
                "git",
                "-C",
                "vendor/ud-tools",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "src/shared",
                "src/sidekick",
                "src/chat",
                "src/python/src/utils",
                "src/contracts.py",
            ],
            **subprocess_contract,
        ),
    ]


@pytest.mark.parametrize(
    "status_output",
    [
        " M src/shared/python/chat/chat_dock_widget.py\n",
        "?? src/sidekick/untracked_runtime.py\n",
    ],
)
def test_dirty_relevant_pinned_tools_sources_fail_closed(
    tmp_path,
    status_output: str,
) -> None:
    """Tracked and untracked source drift invalidate package provenance."""
    results = _matching_tools_git_results()
    results[2] = subprocess.CompletedProcess(
        results[2].args,
        0,
        stdout=status_output,
        stderr="",
    )
    hook = build_hooks.UIBuildHook(str(tmp_path), {})

    with (
        patch("build_hooks.subprocess.run", side_effect=results),
        pytest.raises(RuntimeError, match="Pinned Tools package sources are not clean"),
    ):
        hook._register_tools_packages("1.0.0", {})


@pytest.mark.parametrize("version", ["1.0.0", "editable"])
def test_explicit_tools_gitlink_mismatch_always_fails_closed(
    tmp_path,
    monkeypatch,
    version: str,
) -> None:
    monkeypatch.delenv("CI", raising=False)
    mismatched_results = _matching_tools_git_results()
    mismatched_results[1] = subprocess.CompletedProcess(
        mismatched_results[1].args,
        0,
        stdout=f"{'b' * 40}\n",
        stderr="",
    )
    hook = build_hooks.UIBuildHook(str(tmp_path), {})

    with (
        patch("build_hooks.subprocess.run", side_effect=mismatched_results),
        pytest.raises(RuntimeError, match="does not match superproject gitlink"),
    ):
        hook._register_tools_packages(version, {})


@pytest.mark.parametrize(
    ("version", "ci"),
    [("1.0.0", False), ("editable", True)],
)
def test_release_or_ci_missing_tools_git_metadata_fails_closed(
    tmp_path,
    monkeypatch,
    version: str,
    ci: bool,
) -> None:
    monkeypatch.setenv("CI", "true" if ci else "false")
    hook = build_hooks.UIBuildHook(str(tmp_path), {})

    with (
        patch(
            "build_hooks.subprocess.run",
            side_effect=FileNotFoundError("secret subprocess detail"),
        ),
        pytest.raises(RuntimeError, match="git metadata unavailable") as error,
    ):
        hook._register_tools_packages(version, {})

    assert "secret subprocess detail" not in str(error.value)


def test_release_build_accepts_content_bound_container_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    """An isolated Docker context may replace Git metadata, not verification."""
    tools_root = tmp_path / "vendor" / "ud-tools"
    monkeypatch.setenv("UPSTREAMDRIFT_TOOLS_GITLINK_SHA", _PINNED_TOOLS_SHA)
    monkeypatch.setenv(
        "UPSTREAMDRIFT_TOOLS_SOURCE_SHA256",
        compute_tools_source_sha256(tools_root),
    )
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    build_data: dict = {}

    with patch(
        "build_hooks.subprocess.run",
        side_effect=FileNotFoundError("Git metadata excluded from Docker context"),
    ):
        hook._register_tools_packages("1.0.0", build_data)

    assert build_data["force_include"]


def test_release_build_rejects_mismatched_container_source_digest(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("UPSTREAMDRIFT_TOOLS_GITLINK_SHA", _PINNED_TOOLS_SHA)
    monkeypatch.setenv("UPSTREAMDRIFT_TOOLS_SOURCE_SHA256", "b" * 64)
    hook = build_hooks.UIBuildHook(str(tmp_path), {})

    with (
        patch(
            "build_hooks.subprocess.run",
            side_effect=FileNotFoundError("Git metadata excluded from Docker context"),
        ),
        pytest.raises(RuntimeError, match="source digest does not match"),
    ):
        hook._register_tools_packages("1.0.0", {})


def test_editable_non_ci_missing_git_metadata_warns_and_skips(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.delenv("CI", raising=False)
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    build_data: dict = {}

    with patch(
        "build_hooks.subprocess.run",
        side_effect=FileNotFoundError("secret subprocess detail"),
    ):
        hook._register_tools_packages("editable", build_data)

    assert build_data == {}
    assert "git metadata unavailable" in caplog.text
    assert "secret subprocess detail" not in caplog.text


def test_ui_build_hook_ci_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CI", "true")
    (tmp_path / "ui" / "dist").mkdir(parents=True)
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    _initialize_with_pinned_tools(hook, "1.0.0", {})
    # Should skip, no error


def test_ui_build_hook_skip_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKIP_UI_BUILD", "1")
    (tmp_path / "ui" / "dist").mkdir(parents=True)
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    _initialize_with_pinned_tools(hook, "1.0.0", {})
    # Should skip, no error


def test_ui_build_hook_ci_env_without_bundle_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CI", "true")
    hook = build_hooks.UIBuildHook(str(tmp_path), {})

    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "UI bundle is missing" in str(exc.value)


def test_ui_build_hook_editable_ci_without_bundle_skips(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CI", "true")
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    _initialize_with_pinned_tools(hook, "editable", {})


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_builds(mock_run, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    _initialize_with_pinned_tools(hook, "1.0.0", {})

    assert mock_run.call_count == 2
    mock_run.assert_any_call(
        (
            ["npm", "ci", "--legacy-peer-deps"]
            if os.name != "nt"
            else ["npm.cmd", "ci", "--legacy-peer-deps"]
        ),
        cwd=str(tmp_path / "ui"),
        check=True,
        capture_output=True,
        text=True,
    )


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_fails(mock_run, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    mock_run.side_effect = subprocess.CalledProcessError(1, "npm")

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "UI build failed" in str(exc.value)


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_missing_npm(mock_run, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    mock_run.side_effect = FileNotFoundError("npm")

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "npm not found" in str(exc.value)


def test_ui_dir_property(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    assert hook._ui_dir == tmp_path / "ui"


def test_dist_dir_property(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    assert hook._dist_dir == tmp_path / "ui" / "dist"


def test_force_ui_build_false_by_default(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    assert hook._force_ui_build() is False


def test_force_ui_build_true_when_configured(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {"force_ui_build": True})
    assert hook._force_ui_build() is True


def test_npm_error_message_prefers_stderr():
    err = subprocess.CalledProcessError(
        1, "npm", stderr="stderr msg", output="stdout msg"
    )
    assert build_hooks.UIBuildHook._npm_error_message(err) == "stderr msg"


def test_npm_error_message_falls_back_to_stdout():
    err = subprocess.CalledProcessError(1, "npm", stderr="", output="stdout msg")
    assert build_hooks.UIBuildHook._npm_error_message(err) == "stdout msg"
