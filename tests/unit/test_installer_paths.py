"""TDD tests for correct asset path resolution in setup scripts (issue #2495).

Two bugs:
1. setup_golf_suite.py._find_source_image() searches repo_root / "launchers" / "assets" / ...
   but the actual assets are under repo_root / "src" / "launchers" / "assets" / ...
2. installer/windows/setup.py hard-codes project_root / "launchers" / ..., "api" / ...,
   "shared" / "urdf" / ..., etc. — all the real paths are under "src/".
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# setup_golf_suite.py
# ---------------------------------------------------------------------------


class TestSetupGolfSuiteSourceImagePaths:
    """_find_source_image() must search the correct src/launchers/assets/ location."""

    def test_find_source_image_searches_src_launchers_assets(self) -> None:
        """_find_source_image must include paths under src/launchers/assets, not bare launchers/assets."""
        import inspect

        import setup_golf_suite

        src = inspect.getsource(setup_golf_suite._find_source_image)
        assert "src" + "/" + "launchers" in src or '"src"' in src, (
            "_find_source_image does not reference src/launchers/assets — "
            "the assets were moved to src/launchers/assets but the function still searches launchers/assets"
        )

    def test_find_source_image_no_bare_launchers_path(self) -> None:
        """_find_source_image must not search the non-existent bare launchers/assets path."""
        import inspect

        import setup_golf_suite

        src = inspect.getsource(setup_golf_suite._find_source_image)
        # The bare "launchers" / "assets" path without "src/" prefix should not appear
        # We check that "launchers" does not appear without a preceding "src" context
        lines = src.splitlines()
        for line in lines:
            if '"launchers"' in line or "'launchers'" in line:
                assert (
                    '"src"' in line or "'src'" in line or "src" in line
                ), f"Line references bare launchers/ path without src/: {line.strip()}"

    def test_output_icon_path_under_src_launchers(self) -> None:
        """The output_icon path in main() must reference src/launchers/assets, not launchers/assets."""
        source = Path("setup_golf_suite.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Constant)
                and node.right.value == "launchers"
            ):
                # Find string literals in division chains (Path / "launchers")
                # Check the left side doesn't skip "src"
                source_segment = ast.unparse(node)
                # If "launchers" is directly after repo_root without "src", that's the bug
                assert (
                    "src" in source_segment
                ), f"Path construction found 'launchers' without 'src' prefix: {source_segment}"


# ---------------------------------------------------------------------------
# installer/windows/setup.py
# ---------------------------------------------------------------------------


class TestWindowsInstallerPaths:
    """installer/windows/setup.py must reference src/ paths for launchers, api, config, shared."""

    def _read_installer_source(self) -> str:
        return Path("installer/windows/setup.py").read_text()

    def test_launcher_script_path_includes_src(self) -> None:
        """The upstream_drift_launcher.py script path must include 'src/'."""
        source = self._read_installer_source()
        # Find lines that reference upstream_drift_launcher.py
        for line in source.splitlines():
            if "upstream_drift_launcher.py" in line:
                assert (
                    '"src"' in line or "'src'" in line or "src" in line
                ), f"upstream_drift_launcher.py path missing 'src': {line.strip()}"

    def test_api_server_path_includes_src(self) -> None:
        """The api/server.py script path must include 'src/'."""
        source = self._read_installer_source()
        for line in source.splitlines():
            if "server.py" in line and "api" in line:
                assert (
                    '"src"' in line or "'src'" in line or "src" in line
                ), f"api/server.py path missing 'src': {line.strip()}"

    def test_shared_urdf_path_includes_src(self) -> None:
        """The shared/urdf include_files path must reference src/shared/urdf."""
        source = self._read_installer_source()
        for line in source.splitlines():
            if '"urdf"' in line or "'urdf'" in line or "/urdf" in line:
                assert "src" in line, f"shared/urdf path missing 'src': {line.strip()}"

    def test_config_path_includes_src(self) -> None:
        """The config include_files path must reference src/config."""
        source = self._read_installer_source()
        for line in source.splitlines():
            if (
                ('"config"' in line or "'config'" in line)
                and "include"
                in source[
                    max(0, source.find(line) - 200) : source.find(line) + len(line)
                ]
                and "project_root" in line
            ):
                assert "src" in line, f"config path missing 'src': {line.strip()}"
