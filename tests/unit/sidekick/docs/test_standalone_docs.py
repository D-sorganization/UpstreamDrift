"""T9 TDD: standalone.md exists and its runnable examples are correct.

Validates acceptance criterion: docs include runnable examples that reflect
the actual sidekick CLI interface.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent.parent
STANDALONE_DOC = ROOT / "docs" / "sidekick" / "standalone.md"
FIXTURES_WGS = ROOT / "tests" / "fixtures" / "wgs.json"

# Ensure the subprocess can import sidekick from the source tree.
_PYTHONPATH_EXTRA = os.pathsep.join(
    [
        str(ROOT / "src" / "shared" / "python"),
        str(ROOT / "src"),
        str(ROOT),
    ]
)


def _env_with_pythonpath() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_PYTHONPATH_EXTRA}{os.pathsep}{existing}" if existing else _PYTHONPATH_EXTRA
    )
    return env


# ---------------------------------------------------------------------------
# T9-AC-2: docs file exists and has required sections
# ---------------------------------------------------------------------------


def test_standalone_doc_exists() -> None:
    assert STANDALONE_DOC.exists(), (
        f"docs/sidekick/standalone.md not found at {STANDALONE_DOC}"
    )


def test_standalone_doc_has_install_section() -> None:
    text = STANDALONE_DOC.read_text(encoding="utf-8")
    assert "pip install" in text, "standalone.md must include pip install instructions"


def test_standalone_doc_has_sidekick_run_example() -> None:
    text = STANDALONE_DOC.read_text(encoding="utf-8")
    assert "sidekick run" in text, "standalone.md must show a 'sidekick run' example"


def test_standalone_doc_has_two_layouts() -> None:
    text = STANDALONE_DOC.read_text(encoding="utf-8")
    assert "chat-first" in text, "standalone.md must document the chat-first layout"
    assert "calc-first" in text, "standalone.md must document the calc-first layout"


# ---------------------------------------------------------------------------
# T9-AC-2: runnable examples actually work (doctest-style subprocess check)
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_sidekick_help_exits_zero() -> None:
    """The 'sidekick --help' example in the docs actually works."""
    main_py = ROOT / "src" / "shared" / "python" / "sidekick" / "__main__.py"
    if not main_py.exists():
        pytest.skip(
            "sidekick/__main__.py not present in this branch (T6 not yet merged)"
        )

    result = subprocess.run(
        [sys.executable, "-m", "sidekick", "--help"],
        capture_output=True,
        cwd=str(ROOT),
        env=_env_with_pythonpath(),
    )
    assert result.returncode == 0, (
        f"'python -m sidekick --help' failed:\n{result.stderr.decode()}"
    )


@pytest.mark.headless_safe
def test_sidekick_run_wgs_exits_zero(tmp_path: Path) -> None:
    """The 'sidekick run --calculator wgs_reactor' example in the docs works."""
    main_py = ROOT / "src" / "shared" / "python" / "sidekick" / "__main__.py"
    if not main_py.exists():
        pytest.skip(
            "sidekick/__main__.py not present in this branch (T6 not yet merged)"
        )
    if not FIXTURES_WGS.exists():
        pytest.skip("wgs.json fixture not found — run from full repo")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sidekick",
            "run",
            "--calculator",
            "wgs_reactor",
            "--inputs",
            str(FIXTURES_WGS),
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        cwd=str(ROOT),
        env=_env_with_pythonpath(),
    )
    assert result.returncode == 0, f"sidekick run failed:\n{result.stderr.decode()}"
    assert (tmp_path / "out.json").exists()
