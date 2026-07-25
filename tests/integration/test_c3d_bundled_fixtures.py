"""Regression: every bundled C3D fixture must load, and junk must be rejected.

Issue #8082 -- the C3D Viewer crashed with ``IndexError: list index out of
range`` on the repository's own ``tests/data/motion_pipeline/golden/sample.c3d``
because ``POINT:UNITS`` is present but empty, and then again with a bogus
"positions exceed 10m" ``ValueError`` because the biomechanical range check
compared millimetre coordinates against a metre threshold. 16 of the 18 bundled
``.c3d`` files were unloadable.

Issue #8073 -- feeding a non-C3D file to the viewer must produce a clean,
user-facing message rather than a raw ``OSError`` traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "motion_pipeline" / "golden"


def _bundled_c3d_files() -> list[Path]:
    """Return every ``.c3d`` file committed to the repository."""
    found = sorted((REPO_ROOT / "tests").rglob("*.c3d"))
    found += sorted((REPO_ROOT / "data").rglob("*.c3d"))
    return found


def _reader_class() -> type:
    ezc3d = pytest.importorskip("ezc3d")  # noqa: F841 - availability probe
    from sidekick.lab.bio.c3d_reader import C3DDataReader

    return C3DDataReader


@pytest.mark.parametrize(
    "fixture",
    _bundled_c3d_files(),
    ids=lambda p: p.name,
)
def test_bundled_c3d_fixture_loads(fixture: Path) -> None:
    """Every shipped fixture parses metadata and marker frames without raising."""
    reader = _reader_class()(fixture)

    metadata = reader.get_metadata()
    frame = reader.points_dataframe(include_time=False)

    assert metadata.units, "metadata must report a POINT unit"
    assert not frame.empty, f"{fixture.name} produced no marker rows"


def test_golden_sample_reports_default_units_when_parameter_empty() -> None:
    """``sample.c3d`` ships an empty ``POINT:UNITS``; it must default, not crash."""
    from sidekick.lab.bio._c3d_io import DEFAULT_POINT_UNITS

    metadata = _reader_class()(GOLDEN_DIR / "sample.c3d").get_metadata()

    assert metadata.units == DEFAULT_POINT_UNITS


def test_non_c3d_file_raises_actionable_value_error(tmp_path: Path) -> None:
    """A CSV fed to the C3D reader yields a user-facing ValueError, not OSError."""
    pytest.importorskip("ezc3d")
    from sidekick.lab.bio.c3d_reader import C3DDataReader

    bogus = tmp_path / "not_really.c3d"
    bogus.write_text("frame,marker,x,y,z\n0,M1,1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        C3DDataReader(bogus).get_metadata()

    message = str(excinfo.value)
    assert "not_really.c3d" in message
    assert "not a readable C3D file" in message


def test_bundled_invalid_csv_fixture_is_rejected_cleanly() -> None:
    """The golden CSV is not a C3D file; rejection must stay a ValueError."""
    pytest.importorskip("ezc3d")
    from sidekick.lab.bio.c3d_reader import C3DDataReader

    with pytest.raises(ValueError, match="not a readable C3D file"):
        C3DDataReader(GOLDEN_DIR / "sample.csv").get_metadata()
