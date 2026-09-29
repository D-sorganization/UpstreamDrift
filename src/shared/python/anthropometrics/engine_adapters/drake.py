"""Drake :class:`EngineAdapter` — exports and re-imports a URDF.

Drake (https://drake.mit.edu) reads URDF natively via
``MultibodyPlant.AddModels``. The :class:`DrakeAdapter` therefore
emits the same URDF schema as the Pinocchio adapter and recovers
the canonical :class:`SubjectAnthropometrics` from it without
information loss.

The adapter does not depend on the ``drake`` Python wheel itself —
the on-disk URDF is sufficient and Drake-loading tests are
gated behind :func:`pytest.importorskip` in the unit test layer.
"""

from __future__ import annotations

from pathlib import Path

from .._subject_anthropometrics import SubjectAnthropometrics
from ._urdf_io import read_urdf_subject, write_urdf_subject


class DrakeAdapter:
    """Round-trip a :class:`SubjectAnthropometrics` through a Drake URDF."""

    engine_name: str = "drake"

    def export(
        self, anthropometrics: SubjectAnthropometrics, output_path: Path
    ) -> None:
        """Write *anthropometrics* as a Drake-compatible URDF at *output_path*."""
        write_urdf_subject(anthropometrics, Path(output_path))

    def import_back(self, input_path: Path) -> SubjectAnthropometrics:
        """Re-load a previously exported subject from *input_path*."""
        return read_urdf_subject(Path(input_path))
