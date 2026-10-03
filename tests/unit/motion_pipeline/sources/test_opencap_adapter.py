"""Tests for the OpenCap session adapter.

Fixtures follow the layout and marker names real OpenCap sessions use
(see ``opencap_fixtures``); #11402 replaced the earlier invented labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.contracts import CanonicalObservations
from src.shared.python.motion_pipeline.scaling.marker_maps import (
    OPENCAP_LAI_UHLRICH_2022,
    get_marker_set,
)
from src.shared.python.motion_pipeline.sources import (
    OpenCapSessionAdapter,
    detect_format,
    list_formats,
)
from src.shared.python.motion_pipeline.sources.opencap_adapter import (
    normalize_opencap_marker_name,
)
from tests.unit.motion_pipeline.sources.opencap_fixtures import (
    OPENCAP_AUGMENTED_MARKERS,
    OPENCAP_DETECTOR_KEYPOINTS,
    write_opencap_session,
    write_trc,
)


def test_real_opencap_session_imports_to_canonical_observations(
    tmp_path: Path,
) -> None:
    session = write_opencap_session(tmp_path)

    observations = OpenCapSessionAdapter().load_checked(session)

    assert isinstance(observations, CanonicalObservations)
    assert observations.num_frames == 3
    assert observations.marker_set_name == "OpenCap-LaiUhlrich2022"
    asis = observations.frames[0].markers["r.ASIS_study"]
    # Detector keypoints come first in the fixture, so r.ASIS_study is
    # column len(OPENCAP_DETECTOR_KEYPOINTS); units are already metres.
    assert asis.x == pytest.approx(float(len(OPENCAP_DETECTOR_KEYPOINTS)))
    assert observations.source_provenance["format"] == "opencap_session"


def test_every_real_augmented_marker_is_recognized(tmp_path: Path) -> None:
    session = write_opencap_session(tmp_path)

    observations = OpenCapSessionAdapter().load_checked(session)

    assert set(OPENCAP_LAI_UHLRICH_2022.markers) == set(OPENCAP_AUGMENTED_MARKERS)
    assert observations.metadata["augmented_markers"] == list(OPENCAP_AUGMENTED_MARKERS)
    assert observations.metadata["detector_keypoints"] == list(
        OPENCAP_DETECTOR_KEYPOINTS
    )


def test_real_names_pass_through_unchanged() -> None:
    for name in OPENCAP_AUGMENTED_MARKERS + OPENCAP_DETECTOR_KEYPOINTS:
        assert normalize_opencap_marker_name(name) == name


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("R_ASIS", "r.ASIS_study"),
        ("l.psis", "L.PSIS_study"),
        ("r.ASIS", "r.ASIS_study"),
        ("R_Shoulder", "r_shoulder_study"),
        ("L.Acromium", "L_shoulder_study"),
        ("RHJC", "RHJC_study"),
    ],
)
def test_legacy_labels_map_onto_the_opencap_vocabulary(
    label: str, expected: str
) -> None:
    assert normalize_opencap_marker_name(label) == expected


def test_detector_shoulder_is_not_merged_into_the_augmented_shoulder() -> None:
    # OpenCap writes both RShoulder (detector) and r_shoulder_study (augmented)
    # into the same TRC; aliasing one onto the other would drop a column.
    assert normalize_opencap_marker_name("RShoulder") == "RShoulder"


def test_colliding_labels_are_rejected(tmp_path: Path) -> None:
    trc = write_trc(tmp_path / "MarkerData" / "swing1.trc", ("R_ASIS", "r.ASIS_study"))

    with pytest.raises(ValueError, match=r"r\.ASIS_study"):
        OpenCapSessionAdapter().load(trc.parent.parent)


def test_opencap_marker_set_is_registered() -> None:
    assert get_marker_set("opencap") is OPENCAP_LAI_UHLRICH_2022
    assert get_marker_set("LaiUhlrich2022") is OPENCAP_LAI_UHLRICH_2022


def test_opencap_metadata_reports_augmented_marker_session(tmp_path: Path) -> None:
    session = write_opencap_session(tmp_path)

    metadata = OpenCapSessionAdapter().metadata(session)

    assert metadata.format_name == "opencap_session"
    assert metadata.frame_count == 3
    assert metadata.fps == pytest.approx(60.0)
    assert metadata.marker_set_name == "OpenCap-LaiUhlrich2022"


def test_legacy_json_metadata_is_still_read(tmp_path: Path) -> None:
    session = write_opencap_session(tmp_path, with_metadata=False)
    (session / "sessionMetadata.json").write_text(
        json.dumps({"sessionName": "demo-session", "massKg": 72.5}),
        encoding="utf-8",
    )

    observations = OpenCapSessionAdapter().load_checked(session)

    assert observations.subject is not None
    assert observations.subject["massKg"] == 72.5


def test_opencap_adapter_is_registered(tmp_path: Path) -> None:
    session = write_opencap_session(tmp_path)

    assert "opencap_session" in list_formats()
    assert detect_format(session) is OpenCapSessionAdapter
