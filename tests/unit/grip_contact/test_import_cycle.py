"""Cold-process import order regression for the grip_contact package.

``model_appearance.grip_pose`` imports ``grip_contact.contact_topology``, which
runs the ``grip_contact`` package ``__init__``; the binding and allocation
modules it loads must not import ``motion_matching.grip_contract`` (which
imports ``grip_pose``) at module level.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]

FIRST_IMPORTS = (
    "src.shared.python.model_appearance.grip_pose",
    "src.shared.python.model_appearance",
    "src.shared.python.motion_matching.grip_contract",
    "src.shared.python.grip_contact",
)


@pytest.mark.parametrize("module", FIRST_IMPORTS)
def test_cold_import_has_no_cycle(module: str) -> None:
    """Each entry point imports first in a fresh interpreter without a cycle."""
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and argument list
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
