"""XXE / entity-expansion rejection tests for model-file parsing (#6927).

These adapters parse user-supplied URDF / OpenSim XML on the upload/convert
path. They must use ``defusedxml`` so external-entity references and entity
expansion are refused rather than resolved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# defusedxml raises subclasses of this for forbidden constructs.
from defusedxml.common import DefusedXmlException

from anthropometrics.engine_adapters._urdf_io import read_urdf_subject
from anthropometrics.engine_adapters.opensim import OpenSimAdapter

# A classic XXE payload: an external entity pointing at a local file. A safe
# parser must refuse to resolve it (defusedxml raises before the root tag is
# even inspected).
_XXE_DOC = """<?xml version="1.0"?>
<!DOCTYPE root [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>
"""


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_opensim_import_rejects_external_entity(tmp_path: Path) -> None:
    """OpenSim ``import_back`` must refuse external entities (XXE)."""
    malicious = _write(tmp_path / "evil.osim", _XXE_DOC)
    with pytest.raises(DefusedXmlException):
        OpenSimAdapter().import_back(malicious)


def test_urdf_reader_rejects_external_entity(tmp_path: Path) -> None:
    """URDF subject reader must refuse external entities (XXE)."""
    malicious = _write(tmp_path / "evil.urdf", _XXE_DOC)
    with pytest.raises(DefusedXmlException):
        read_urdf_subject(malicious)
