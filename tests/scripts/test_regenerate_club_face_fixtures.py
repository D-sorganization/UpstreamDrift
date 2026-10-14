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


@pytest.mark.parametrize("capture", ["driver", "iron"])
def test_fixture_runs_hold_the_measured_club_cluster_fixed(capture: str) -> None:
    # Each grip rotation is fitted on the fixed cluster; a free cluster
    # re-absorbs it and moves the capture's own face (#12221). The 7-iron
    # refit is adopted with the 0.125 ms export hold (#12279).
    argv = regen.pipeline_argv(capture, Path("r"), 3.0)
    assert "--fix-club-markers" in argv and "--static-seeds" in argv


def test_club_face_export_requests_the_opt_in_eighth_millisecond_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Owner ruling on #12279 (2026-10-11): the club-face fixtures export with
    # the 0.125 ms effort hold; the global same-input default stays 1 ms.
    import inspect

    from scripts import same_input_bundle

    run_dir = tmp_path / "iron"
    run_dir.mkdir()
    (run_dir / "receipt.json").write_text("{}", encoding="utf-8")
    calls: list[tuple] = []

    def fake_export(run, out, duration_s, effort_hold_ms=1.0):  # noqa: ANN001, ANN202
        calls.append((run, out, duration_s, effort_hold_ms))
        return {}

    monkeypatch.setattr(same_input_bundle, "export", fake_export)
    run, bundle = regen.run_and_export("iron", tmp_path, 3.0, reuse=True)
    assert run == run_dir and bundle == tmp_path / "bundles" / "iron.npz"
    assert calls == [(run_dir, bundle, None, 0.125)]
    assert regen.CLUB_FACE_EFFORT_HOLD_MS == 0.125
    default = inspect.signature(same_input_bundle.export).parameters
    assert default["effort_hold_ms"].default == 1.0


def test_a_reused_bundle_with_another_hold_is_re_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --reuse-run must not adopt a bundle exported on the 1 ms contract.
    import numpy as np

    from scripts import same_input_bundle

    (tmp_path / "iron").mkdir()
    (tmp_path / "iron" / "receipt.json").write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundles" / "iron.npz"
    bundle.parent.mkdir()
    calls: list[float] = []
    monkeypatch.setattr(
        same_input_bundle,
        "export",
        lambda run, out, d, effort_hold_ms=1.0: calls.append(effort_hold_ms) or {},
    )
    for hold_s, expected in ((1e-3, [0.125]), (1.25e-4, [])):
        manifest = {"policy": {"effort_hold_s": hold_s}}
        np.savez(bundle, manifest=np.array(json.dumps(manifest)))
        calls.clear()
        regen.run_and_export("iron", tmp_path, 3.0, reuse=True)
        assert calls == expected
    np.savez(bundle, manifest=np.array(json.dumps({"policy": {}})))
    calls.clear()
    regen.run_and_export("iron", tmp_path, 3.0, reuse=True)
    assert calls == [0.125]  # a legacy bundle holds 1 ms


def test_tracked_swing_is_written_beside_the_fixture_on_the_bundle_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The #12279 foot acceptance compares the fixture with the run's tracked
    # swing, so the regeneration commits that q_track (float32) as well.
    import numpy as np

    run = tmp_path / "run"
    run.mkdir()
    q = np.arange(12.0).reshape(3, 4)
    np.savez(run / "dynamics_record.npz", track_time_s=[0.5, 0.6, 0.7], q_track=q)
    monkeypatch.setattr(regen, "FIXTURES", tmp_path)
    path = regen._write_tracked_swing("iron7", run)  # noqa: SLF001
    assert path == tmp_path / "tracked_q_iron7.npz"
    with np.load(path) as data:
        np.testing.assert_allclose(data["time_s"], [0.0, 0.1, 0.2], atol=1e-12)
        assert data["q"].dtype == np.float32
        np.testing.assert_array_equal(data["q"], q)


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
