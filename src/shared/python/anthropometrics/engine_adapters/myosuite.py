"""MyoSuite :class:`EngineAdapter` — emits BOTH URDF and MJCF.

MyoSuite (https://sites.google.com/view/myosuite) is built atop
MuJoCo and consumes both URDF (via converters) and native MJCF
``.xml`` documents. The :class:`MyoSuiteAdapter` therefore writes
*both* a URDF sibling (``output_path.with_suffix(".urdf")``) and
an MJCF sibling (``output_path.with_suffix(".xml")``) so downstream
pipelines can pick whichever the active MyoSuite environment
expects.

:meth:`import_back` accepts either a ``.urdf`` or ``.xml``
(MJCF) path — the dispatch is driven solely by the file
extension — and recovers the canonical
:class:`SubjectAnthropometrics`.
"""

from __future__ import annotations

from pathlib import Path

from .._subject_anthropometrics import SubjectAnthropometrics
from ._mjcf_io import read_mjcf_subject, write_mjcf_subject
from ._urdf_io import read_urdf_subject, write_urdf_subject


class MyoSuiteAdapter:
    """Round-trip a :class:`SubjectAnthropometrics` via URDF + MJCF."""

    engine_name: str = "myosuite"

    def export(
        self, anthropometrics: SubjectAnthropometrics, output_path: Path
    ) -> None:
        """Write both a ``.urdf`` and a ``.xml`` (MJCF) under *output_path*'s stem.

        ``output_path`` may carry any suffix (or none) — the
        method derives both sibling files from its stem so that
        downstream code can pass either format to
        :meth:`import_back` directly.
        """
        output_path = Path(output_path)
        urdf_path = output_path.with_suffix(".urdf")
        mjcf_path = output_path.with_suffix(".xml")
        write_urdf_subject(anthropometrics, urdf_path)
        write_mjcf_subject(anthropometrics, mjcf_path)

    def import_back(self, input_path: Path) -> SubjectAnthropometrics:
        """Re-load a subject from either the ``.urdf`` or the ``.xml`` sibling."""
        input_path = Path(input_path)
        suffix = input_path.suffix.lower()
        if suffix == ".urdf":
            return read_urdf_subject(input_path)
        if suffix in (".xml", ".mjcf"):
            return read_mjcf_subject(input_path)
        raise ValueError(
            "MyoSuiteAdapter.import_back requires a .urdf, .xml, or .mjcf "
            f"path; got {input_path.name!r}"
        )
