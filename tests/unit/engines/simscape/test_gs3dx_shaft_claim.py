"""The GS3DX design report must describe the shaft the committed model has.

R11 (#12691) asks for a flexible shaft in every engine and says to verify the
Simscape GS3DX "flexible shaft" claim first. In ``GS3DX_Human.slx`` the
``Flexible Beam Model`` subsystem exists but is commented out, and the active
shaft is the ``Rigid Shaft`` cylinder. Contract: the design report claims a
flexible shaft only while the beam subsystem is active, and says rigid
otherwise. Read straight from the model zip, so no MATLAB is needed.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_GS3DX = (
    Path(__file__).resolve().parents[4]
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx"
)
_SLX = _GS3DX / "models" / "GS3DX_Human.slx"
_REPORT = _GS3DX / "docs" / "DESIGN_REPORT.md"
_BLOCK = r'<Block BlockType="SubSystem" Name="{name}"[^>]*>(.*?)</Block>'


def _is_active(name: str) -> bool:
    """True when subsystem ``name`` exists in the Club system and is not commented."""
    pattern = re.compile(_BLOCK.format(name=re.escape(name)), re.DOTALL)
    with zipfile.ZipFile(_SLX) as archive:
        for entry in archive.namelist():
            if not entry.startswith("simulink/systems/system_"):
                continue
            match = pattern.search(archive.read(entry).decode("utf-8", "ignore"))
            if match:
                head = match.group(1)[:1500]
                return not re.search(r'<P Name="Commented">(on|through)</P>', head)
    raise AssertionError(f"{name!r} not found in {_SLX.name}")


def test_model_ships_the_rigid_shaft_active() -> None:
    assert _is_active("Rigid Shaft")


def test_design_report_claims_flexible_shaft_only_when_beam_is_active() -> None:
    text = _REPORT.read_text(encoding="utf-8").lower()
    if _is_active("Flexible Beam Model"):
        return
    assert "with a flexible shaft" not in text
    assert "rigid shaft" in text
