"""TDD tests for build_hooks.py and install.sh contracts (issue #2496).

Two bugs:
1. UIBuildHook.initialize() silently logs a warning when CI=1 and ui/dist is absent.
   Packaging then produces an artifact with a broken web UI. The correct behavior is
   to raise an error (or at minimum, fail loudly) when CI is set and ui/dist is absent.
2. install.sh uses `pipx install .` / `pip3 install .` (installs from current-dir `.`),
   but when piped via `curl | bash` there is no local checkout. The script must either
   clone the repo first or install from a remote URL.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


class TestUIBuildHookMissingDist:
    """UIBuildHook.initialize() must fail loudly when CI is set and ui/dist is absent.

    hatchling may not be installed in the local dev environment, so these tests
    use source-level inspection rather than instantiating the class.
    """

    def _get_initialize_source(self) -> str:
        return Path("build_hooks.py").read_text()

    def test_ci_block_raises_when_dist_missing(self) -> None:
        """The CI block in initialize() must raise when dist_dir is absent, not silently return."""
        source = self._get_initialize_source()

        assert '_env_flag("CI")' in source
        assert "if skip_requested and not force_ui_build:" in source
        assert "UI bundle is missing" in source
        assert "raise RuntimeError(msg)" in source

    def test_hatch_sdist_preserves_built_ui_artifacts(self) -> None:
        """The sdist-to-wheel path must carry the prebuilt UI bundle forward."""
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        hatch_build = pyproject["tool"]["hatch"]["build"]

        assert "ui/dist/**/*" in hatch_build.get("include", [])
        assert "ui/dist/**/*" in hatch_build.get("artifacts", [])


class TestInstallShNotLocalInstall:
    """install.sh must not run `pipx install .` or `pip3 install .` as a remote-install script."""

    def _install_sh_source(self) -> str:
        return Path("install.sh").read_text(encoding="utf-8")

    def test_install_sh_does_not_install_from_dot(self) -> None:
        """install.sh must not use `install .` (current-directory install fails without a checkout)."""
        source = self._install_sh_source()
        for line in source.splitlines():
            stripped = line.strip()
            # Skip commented lines
            if stripped.startswith("#"):
                continue
            # Check for bare `install .` at the end of the install command
            if (
                "pipx install ." in stripped
                or "pip3 install ." in stripped
                or "pip install ." in stripped
            ):
                raise AssertionError(
                    f"install.sh runs a local-checkout install command: {stripped!r}\n"
                    "This fails when the script is run via `curl | bash` without a checkout."
                )

    def test_install_sh_clones_or_uses_remote_url(self) -> None:
        """install.sh must either clone the repo or install from a git/pypi URL."""
        source = self._install_sh_source()
        # At least one of: git clone, pip install git+, pip install from a URL, gh repo clone
        has_remote_install = (
            "git clone" in source
            or "git+https" in source
            or "git+http" in source
            or "git+${REPO_URL}" in source
            or "pipx install git+" in source
            or "pip3 install git+" in source
            or "pip install git+" in source
        )
        assert has_remote_install, (
            "install.sh does not clone the repo or install from a remote URL. "
            "When run via `curl | bash`, there is no local checkout."
        )
