"""Guard: simulation output must never be written under ``src/``.

See UpstreamDrift issue #9220.

``OutputManager`` resolves its default base directory by walking up from its
own module looking for a project-root marker. The old heuristic accepted any
ancestor containing a directory named ``engines``, which matches ``src/``
(``src/engines`` exists). Every default-constructed ``OutputManager`` -- the
one ``SimulationService`` builds when no manager is injected -- therefore
rooted its output at ``src/output``, so a plain test run left untracked
``src/output/simulations/<engine>/simulation_*.json`` files behind. That
dirties ``git status --short``, which several agent workflows in this repo use
to decide whether work is in progress.

The fix is two-part and both halves are pinned here:

* the default resolves to the *repository* root's documented ``output/``
  directory, never to ``src/``; and
* ``UPSTREAM_DRIFT_OUTPUT_DIR`` overrides it, which ``tests/conftest.py`` sets
  at import time so the suite writes nowhere inside the checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.data_io._format_handlers import OutputFormat
from src.shared.python.data_io._path_utils import (
    OUTPUT_DIR_ENV_VAR,
    find_project_root,
    resolve_base_path,
)
from src.shared.python.data_io.output_manager import OutputManager

_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / ".git").exists()
)
_SRC_DIR = _PROJECT_ROOT / "src"


@pytest.mark.unit
def test_find_project_root_skips_src() -> None:
    """Root detection must reach the checkout root, not ``src/``."""
    assert find_project_root() == _PROJECT_ROOT
    assert find_project_root() != _SRC_DIR


@pytest.mark.unit
def test_default_output_base_is_not_inside_src(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default output base must resolve outside the source tree."""
    monkeypatch.delenv(OUTPUT_DIR_ENV_VAR, raising=False)

    resolved = resolve_base_path(None)

    assert not resolved.resolve().is_relative_to(_SRC_DIR), (
        f"Default output base {resolved} is inside src/; generated run "
        "artifacts would dirty the source tree."
    )
    assert resolved.resolve() == (_PROJECT_ROOT / "output").resolve()


@pytest.mark.unit
def test_output_dir_env_var_overrides_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``UPSTREAM_DRIFT_OUTPUT_DIR`` redirects the default base path."""
    target = tmp_path / "run-output"
    monkeypatch.setenv(OUTPUT_DIR_ENV_VAR, str(target))

    assert resolve_base_path(None).resolve() == target.resolve()
    assert OutputManager().base_path.resolve() == target.resolve()


@pytest.mark.unit
def test_simulation_save_writes_nothing_under_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A representative simulation save must not create files under ``src/``.

    This mirrors what ``SimulationService._persist_simulation_results`` does
    for a completed REST-API run: a default-constructed ``OutputManager`` plus
    a JSON save named ``simulation_<engine>``.
    """
    monkeypatch.setenv(OUTPUT_DIR_ENV_VAR, str(tmp_path / "run-output"))

    src_output_before = (_SRC_DIR / "output").exists()

    saved = OutputManager().save_simulation_results(
        {"time": [0.0, 0.001], "qpos": [[0.0], [0.1]]},
        filename="simulation_mujoco",
        format_type=OutputFormat.JSON,
        engine="mujoco",
    )

    assert saved.exists()
    assert not saved.resolve().is_relative_to(_SRC_DIR), (
        f"Simulation output was written to {saved}, inside the source tree."
    )
    assert (_SRC_DIR / "output").exists() == src_output_before, (
        "Saving simulation results created src/output/ in the checkout."
    )
