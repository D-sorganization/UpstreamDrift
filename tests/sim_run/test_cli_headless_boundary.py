"""``ud-sim`` CLI cold-start stays GUI/physics-free (issue #15830).

Audit of ``ud-sim --help`` import boundary established the following baseline:

* Cold-start wall time: ~489 ms (measured via ``python -X importtime``, 489 lines
  of output, zero GUI/physics entries).
* ``src.sim_run.__init__`` eagerly re-exports MCP symbols (``SimulationMcpServer``,
  ``SecurityPolicy``, ``run_stdio_loop``), accounting for ~136 ms (~28 %) of the
  cold-start cost.  This is intentional; ``ud-sim-mcp`` uses the same package.
* ``sim_run.vector_overlay`` imports matplotlib at module level, but it is
  **not reachable** from ``cli.py``'s import chain.  No action needed.

This test locks those findings in as a regression guard: any future change that
causes the CLI to drag in a GUI or physics library will make this test fail and
surface the regression immediately.

``python -X importtime -m src.sim_run.cli --help`` is used rather than a bare
``import sim_run.cli`` statement because ``cli.py`` uses ``from src.sim_run.*``
absolute imports internally.  Running via ``-m src.sim_run.cli`` avoids the
``src/__init__.py`` vendor-submodule guard that fires when ``python -c`` puts
the repo root on ``sys.path[0]`` and those absolute imports are resolved.

Subprocess pattern mirrors ``tests/bunkershot3d/test_package_import_stays_headless_9145.py``
so the check runs in a clean interpreter unaffected by other test imports.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: Heavyweight GUI/physics modules that the CLI import must never pull in.
_FORBIDDEN_PREFIXES = ("matplotlib", "PyQt", "pyvista", "vtk", "OpenGL")

#: Repo root -- this file lives at ``tests/sim_run/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _importtime_modules() -> tuple[set[str], str]:
    """Run ``ud-sim --help`` with importtime tracing; return (loaded_modules, raw_stderr).

    ``python -X importtime`` writes one line per imported module to stderr in
    the form::

        import time: self [us] | cumulative | <module name>

    Running via ``python -m src.sim_run.cli --help`` is the only reliable way
    to exercise the CLI import chain in this worktree: ``cli.py`` uses
    ``from src.sim_run.*`` absolute imports internally, so a bare
    ``import sim_run.cli`` would hit ``src/__init__.py``'s vendor-submodule
    guard via the ``''`` (CWD) entry that ``python -c`` adds to ``sys.path``.
    """
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "src.sim_run.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        cwd=str(_REPO_ROOT),
    )
    modules: set[str] = set()
    for line in result.stderr.splitlines():
        # importtime lines end with the module name after the last '|'
        if "|" in line:
            module_name = line.rsplit("|", 1)[-1].strip()
            if module_name:
                modules.add(module_name)
    return modules, result.stderr


class TestCliImportBoundary:
    """CLI cold-start must not cross the GUI/physics boundary."""

    def test_import_cli_does_not_load_gui_or_physics_modules(self) -> None:
        """Running ``ud-sim --help`` must not drag in matplotlib, PyQt, pyvista, vtk, or OpenGL.

        Baseline (issue #15830): 489-line importtime trace, zero forbidden entries.
        """
        loaded, raw = _importtime_modules()
        leaked = {m for m in loaded if any(m.startswith(p) for p in _FORBIDDEN_PREFIXES)}
        assert not leaked, (
            f"`ud-sim --help` importtime trace contains {sorted(leaked)}; "
            "the CLI import path must stay GUI/physics-free -- see #15830\n"
            f"Full importtime stderr:\n{raw}"
        )

    def test_vector_overlay_not_reachable_from_cli_import(self) -> None:
        """vector_overlay (real matplotlib import) must not appear in the import chain."""
        loaded, raw = _importtime_modules()
        assert not any("vector_overlay" in m for m in loaded), (
            "`sim_run.vector_overlay` appeared in `ud-sim --help` importtime trace; "
            "it has top-level matplotlib imports and must stay out of the CLI path -- see #15830\n"
            f"Full importtime stderr:\n{raw}"
        )


class TestCliHelpColdStart:
    """``ud-sim --help`` must exit 0 and produce output in a headless environment."""

    def test_help_exits_zero(self) -> None:
        """Running the CLI entry point with --help must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "src.sim_run.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"`python -m src.sim_run.cli --help` exited {result.returncode};\n"
            f"stderr: {result.stderr.strip()}"
        )

    def test_help_mentions_subcommands(self) -> None:
        """Help output must name the core subcommands."""
        result = subprocess.run(
            [sys.executable, "-m", "src.sim_run.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(_REPO_ROOT),
        )
        combined = result.stdout + result.stderr
        for cmd in ("run", "runs", "reproduce", "preset-run"):
            assert cmd in combined, (
                f"`ud-sim --help` did not mention subcommand '{cmd}' -- see #15830"
            )
