"""Pinocchio :class:`EngineAdapter` — exports and re-imports a URDF.

Pinocchio (https://github.com/stack-of-tasks/pinocchio) reads URDF
natively. The :class:`PinocchioAdapter` shares its serialisation
with :class:`DrakeAdapter` — both engines consume the same URDF
schema — but advertises a distinct ``engine_name`` for adapter
registry lookup.
"""

from __future__ import annotations

from pathlib import Path

from .._subject_anthropometrics import SubjectAnthropometrics
from ._urdf_io import read_urdf_subject, write_urdf_subject


class PinocchioAdapter:
    """Round-trip a :class:`SubjectAnthropometrics` through a Pinocchio URDF."""

    engine_name: str = "pinocchio"

    def export(
        self, anthropometrics: SubjectAnthropometrics, output_path: Path
    ) -> None:
        write_urdf_subject(anthropometrics, Path(output_path))

    def import_back(self, input_path: Path) -> SubjectAnthropometrics:
        return read_urdf_subject(Path(input_path))
