"""R11 (#12691) Simscape guard: the committed GS3DX club shaft is rigid.

Simscape cannot join ``CHAIN_ENGINES`` until the GS3DX "flexible shaft" claim is
reconciled in MATLAB R2025b.  That probe needs a licensed host, but the claim
can be checked statically: ``GS3DX_Human.slx`` is a zip of XML.  This gate is
engine-free and pins what the committed model contains, so a future change that
re-enables the beam (or removes the rigid body) forces a re-reconciliation and
cannot silently change the Simscape refusal's premise.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # noqa: S405 - committed model, not untrusted
import zipfile
from pathlib import Path

import pytest

from src.shared.python.motion_matching.club_shaft_chain import CHAIN_ENGINES

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
GS3DX = (
    ROOT
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx"
)
MODEL = GS3DX / "models/GS3DX_Human.slx"
DESIGN_REPORT = GS3DX / "docs/DESIGN_REPORT.md"
BEAM = "Flexible Beam Model"
RIGID = "Rigid Shaft"


def _block_xml(name: str) -> str:
    """Return the ``<Block ... Name=name>`` element text of the club system."""
    pattern = re.compile(
        rf'<Block BlockType="SubSystem" Name="{re.escape(name)}".*?</Block>',
        re.DOTALL,
    )
    with zipfile.ZipFile(MODEL) as archive:
        for entry in archive.namelist():
            if not re.fullmatch(r"simulink/systems/system_\d+\.xml", entry):
                continue
            text = archive.read(entry).decode("utf-8", errors="replace")
            found = pattern.search(text)
            if found:
                return found.group(0)
    raise AssertionError(f"{name!r} block not found in {MODEL.name}")


def _commented_state(block: str) -> str:
    """Return the block's own ``Commented`` value (``off`` when absent).

    Only direct ``<P>`` children count, so a nested block's property is not
    mistaken for the subsystem's.  Simulink values are ``on``, ``off`` and
    ``through``; both ``on`` and ``through`` bypass the subsystem.
    """
    element = ET.fromstring(block)  # noqa: S314 - committed model, not untrusted
    for prop in element.findall("P"):
        if prop.get("Name") == "Commented":
            return (prop.text or "").strip().lower()
    return "off"


def _is_commented(block: str) -> bool:
    """True when the block is bypassed (``on`` or ``through``)."""
    return _commented_state(block) != "off"


def test_flexible_beam_is_present_but_commented_out() -> None:
    assert _is_commented(_block_xml(BEAM))


def test_rigid_shaft_is_the_active_club_body() -> None:
    assert not _is_commented(_block_xml(RIGID))


@pytest.mark.parametrize("state", ["on", "through"])
def test_rigid_shaft_guard_rejects_any_bypass_state(state: str) -> None:
    block = _block_xml(RIGID)
    opening = re.match(r"<Block[^>]*>", block)
    assert opening is not None
    mutated = (
        opening.group(0) + f'<P Name="Commented">{state}</P>' + block[opening.end() :]
    )
    assert mutated != block
    assert _is_commented(mutated)


def test_simscape_stays_outside_the_chain_engines() -> None:
    assert "simscape" not in CHAIN_ENGINES


def test_design_report_does_not_claim_an_active_flexible_shaft() -> None:
    text = DESIGN_REPORT.read_text(encoding="utf-8")
    assert "with a flexible shaft" not in text
    assert "rigid shaft" in text.lower()
    assert "#12691" in text
