"""Tests for the OpenCap session adapter."""

from __future__ import annotations

import json
from pathlib import Path

from src.shared.python.motion_pipeline.contracts import CanonicalObservations
from src.shared.python.motion_pipeline.sources import (
    OpenCapSessionAdapter,
    detect_format,
    list_formats,
)


def _write_opencap_session(tmp_path: Path) -> Path:
    session = tmp_path / "opencap_session"
    outputs = session / "OpenSimData" / "MarkerData"
    outputs.mkdir(parents=True)
    (session / "sessionMetadata.json").write_text(
        json.dumps({"sessionName": "demo-session", "massKg": 72.5}),
        encoding="utf-8",
    )
    (outputs / "augmented_markers.trc").write_text(
        "\n".join(
            [
                "PathFileType\t4\t(X/Y/Z)\taugmented_markers.trc",
                (
                    "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                    "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
                ),
                "100.00\t100.00\t2\t3\tmm\t100.00\t1\t2",
                "Frame#\tTime\tR_ASIS\t\t\tL_ASIS\t\t\tR_Shoulder\t\t\t",
                "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3",
                "1\t0.00\t100\t200\t300\t110\t210\t310\t120\t220\t320",
                "2\t0.01\t101\t201\t301\t111\t211\t311\t121\t221\t321",
            ]
        ),
        encoding="utf-8",
    )
    return session


def test_opencap_session_imports_to_canonical_observations(tmp_path: Path) -> None:
    session = _write_opencap_session(tmp_path)

    observations = OpenCapSessionAdapter().load_checked(session)

    assert isinstance(observations, CanonicalObservations)
    assert observations.num_frames == 2
    assert observations.marker_names == ["R.ASIS", "L.ASIS", "R.Acromium"]
    first_marker = observations.frames[0].markers["R.ASIS"]
    assert first_marker.x == 0.1
    assert observations.source_provenance["format"] == "opencap_session"
    assert observations.subject is not None
    assert observations.subject["massKg"] == 72.5


def test_opencap_metadata_reports_augmented_marker_session(tmp_path: Path) -> None:
    session = _write_opencap_session(tmp_path)

    metadata = OpenCapSessionAdapter().metadata(session)

    assert metadata.format_name == "opencap_session"
    assert metadata.frame_count == 2
    assert metadata.fps == 100.0
    assert metadata.marker_set_name == "OpenCap-OpenSim"


def test_opencap_adapter_is_registered(tmp_path: Path) -> None:
    session = _write_opencap_session(tmp_path)

    assert "opencap_session" in list_formats()
    assert detect_format(session) is OpenCapSessionAdapter
