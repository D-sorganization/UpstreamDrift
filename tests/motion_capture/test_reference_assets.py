"""Reference assets keep source timing, missing data and coordinate provenance."""

from pathlib import Path

import pytest

from src.motion_capture.reference import ReferenceMotion, ReferenceSource
from src.motion_capture.reference.storage import ReferenceLibrary

pytestmark = pytest.mark.unit


def motion(**changes: object) -> ReferenceMotion:
    return ReferenceMotion.model_validate(
        {
            "title": "Expert driver",
            "source": ReferenceSource(path="expert.c3d", sha256="a" * 64, format="c3d"),
            "source_units": "mm",
            "source_axes": ("+X", "+Y", "+Z"),
            "source_names": ("hip", "hand"),
            "joint_names": ("pelvis", "wrist"),
            "edges": ((0, 1),),
            "time_s": (0.0, 0.01, 0.025),
            "points_m": (
                ((0, 0, 0), (1, 0, 0)),
                ((0, 0, 0), None),
                ((0, 0, 0), (2, 0, 0)),
            ),
        }
        | changes
    )


def test_library_round_trip_preserves_masks_clock_and_metadata(tmp_path: Path) -> None:
    library = ReferenceLibrary(tmp_path)
    original = motion()
    library.save(original)
    restored = library.load(original.id)
    assert restored == original
    assert restored.points_m[1][1] is None
    assert restored.points_m[0][0] == (0, 0, 0)  # Origin is valid, not missing.
    assert restored.time_s[-1] == 0.025
    updated = restored.changed(
        title="Lesson 2", notes="Compare wrist at impact", archived=True
    )
    library.save(updated)
    assert library.list() == []
    assert library.list(archived=True) == [updated]


@pytest.mark.parametrize(
    "changes",
    [
        {"time_s": (0, 0, 0.02)},
        {"time_s": (0, float("inf"), 2)},
        {"joint_names": ("wrist", "wrist")},
        {"source_names": ("hip",)},
        {"edges": ((0, 2),)},
        {"source_axes": ("+X", "+X", "+Z")},
        {"points_m": (((1, 2, float("nan")), None),) * 3},
        {"schema_version": "reference-asset/2.0.0"},
    ],
)
def test_rejects_ambiguous_or_invalid_motion(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        motion(**changes)


def test_library_rejects_path_escape_and_mismatched_identity(tmp_path: Path) -> None:
    library = ReferenceLibrary(tmp_path)
    with pytest.raises(ValueError):
        library.load("../outside")
    first, second = motion(), motion()
    library.save(first)
    library.save(second)
    (tmp_path / f"{first.id}.json").write_bytes(
        (tmp_path / f"{second.id}.json").read_bytes()
    )
    with pytest.raises(ValueError, match="identity"):
        library.load(first.id)


def test_catalog_scan_keeps_good_assets_when_another_document_is_corrupt(
    tmp_path: Path,
) -> None:
    library = ReferenceLibrary(tmp_path)
    asset = motion()
    library.save(asset)
    (tmp_path / "bad.json").write_text("broken")
    result = library.scan()
    assert result.assets == (asset,)
    assert len(result.problems) == 1 and "bad.json" in result.problems[0]
