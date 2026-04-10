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
        lines = source.splitlines()

        # Find the CI check block and verify it contains a raise (not just a warning) for missing dist
        ci_start = None
        for i, line in enumerate(lines):
            if 'environ.get("CI")' in line or "environ.get('CI')" in line:
                ci_start = i
                break

        assert ci_start is not None, "Cannot find CI check in build_hooks.py"

        # Collect lines in the CI block (until we exit the indented block)
        ci_indent = len(lines[ci_start]) - len(lines[ci_start].lstrip())
        ci_block_lines = []
        for line in lines[ci_start:]:
            indent = len(line) - len(line.lstrip()) if line.strip() else ci_indent + 1
            if line.strip() and indent <= ci_indent and len(ci_block_lines) > 0:
                break
            ci_block_lines.append(line)

        ci_block = "\n".join(ci_block_lines)
        # The fix must add a raise (not just a warning) when dist is missing
        has_raise_for_missing_dist = "raise" in ci_block and (
            "dist" in ci_block.lower() or "ui" in ci_block.lower()
        )
        assert has_raise_for_missing_dist, (
            "CI block in initialize() does not raise when ui/dist is missing. "
            f"CI block content:\n{ci_block}"
        )


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
            or "pipx install git+" in source
            or "pip3 install git+" in source
            or "pip install git+" in source
        )
        assert has_remote_install, (
            "install.sh does not clone the repo or install from a remote URL. "
            "When run via `curl | bash`, there is no local checkout."
        )
