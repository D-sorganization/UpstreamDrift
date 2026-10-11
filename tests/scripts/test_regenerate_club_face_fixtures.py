"""OSV-10 (#11759): the club-face fixture regeneration script's pure parts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import regenerate_club_face_fixtures as regen
from src.shared.python.motion_matching.club_face_target import HEAD_TRIAD_LABELS

pytestmark = pytest.mark.unit
FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/club_face"


def test_pipeline_argv_keeps_the_original_run_flags_and_the_weight() -> None:
    argv = regen.pipeline_argv("iron", Path("runs/iron"), 3.0)
    assert argv[argv.index("--capture") + 1] == "iron"
    assert "--zmp-filter" in argv and "--static-seeds" in argv
    assert argv[argv.index("--face-weight") + 1] == "3.0"
    assert argv[argv.index("--spec") + 1].endswith("full_body_spec_anthro_iron7.json")
    # The OSV-4 address toe-out calibration, held through the swing (#12222).
    assert argv[argv.index("--foot-progression") + 1] == "capture"
    assert "--zmp-filter" not in regen.pipeline_argv("driver", Path("d"), 0.0)
    with pytest.raises(ValueError, match="unknown capture"):
        regen.pipeline_argv("owner", Path("o"), 3.0)


def test_driver_fixture_run_holds_the_measured_club_cluster_fixed() -> None:
    # The driver's grip rotation is fitted on the fixed cluster; a free
    # cluster re-absorbs it and moves the capture's own face (#12221). The
    # 7-iron keeps its free cluster and grip until its refit is adopted.
    argv = regen.pipeline_argv("driver", Path("r"), 3.0)
    assert "--fix-club-markers" in argv and "--static-seeds" in argv
    assert "--fix-club-markers" not in regen.pipeline_argv("iron", Path("r"), 3.0)


def test_previous_fixture_is_measured_on_its_own_committed_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After a document change the old joint angles only describe the old
    # swing on the old scaled document, committed beside the fixture (#12221).
    import numpy as np

    npz = tmp_path / "swing_q_iron7.npz"
    np.savez_compressed(npz, q=np.zeros((3, 2), dtype=np.float32))
    old_spec = tmp_path / "spec_iron7.json"
    old_spec.write_text("{}", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_measure(run_dir, q, dt_s, model_spec=None):  # noqa: ANN001, ANN202
        seen.update(run_dir=run_dir, shape=q.shape, model_spec=model_spec)
        return {"faces": {}}

    monkeypatch.setattr(regen, "measure_swing", fake_measure)
    record = regen._previous_fixture_record(npz, old_spec, tmp_path / "run")  # noqa: SLF001
    assert seen == {
        "run_dir": tmp_path / "run",
        "shape": (3, 2),
        "model_spec": old_spec,
    }
    assert record["fixture_sha256"] == regen._sha256(npz)  # noqa: SLF001
    assert record["model_spec_sha256"] == regen._sha256(old_spec)  # noqa: SLF001
    assert (
        regen._previous_fixture_record(tmp_path / "none.npz", old_spec, tmp_path)
        is None
    )  # noqa: SLF001


def test_triad_offsets_must_sit_on_the_face_frame() -> None:
    labels = ("Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3")
    receipt = {
        "ik": {
            "attachments_m": {
                k: {"body": "Clubhead", "offset_m": [0.0, i, 0.0]}
                for i, k in enumerate(labels)
            }
        }
    }
    assert regen._triad_offsets(receipt)[labels[2]] == [0.0, 2, 0.0]  # noqa: SLF001
    receipt["ik"]["attachments_m"][labels[0]]["body"] = "Club"
    with pytest.raises(ValueError, match="not attached"):
        regen._triad_offsets(receipt)  # noqa: SLF001


def test_committed_provenance_matches_the_committed_fixtures() -> None:
    import hashlib

    doc = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    assert doc["generator"] == "scripts/regenerate_club_face_fixtures.py"
    for club, record in doc["clubs"].items():
        data = (FIXTURES / f"swing_q_{club}.npz").read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["fixture_sha256"]
        assert record["face_weight"] > 0.0
        assert regen.CAPTURE_RUNS[record["capture"]][0] == club
        assert tuple(record["capture_triad_offsets_m"]) == HEAD_TRIAD_LABELS
