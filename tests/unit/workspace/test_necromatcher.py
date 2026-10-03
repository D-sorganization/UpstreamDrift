"""Durable Necromatcher library contracts (#11233)."""

from pathlib import Path
import json
import pytest
from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.workspace.necromatcher import NecromatcherLibrary

pytestmark = pytest.mark.unit


@pytest.fixture
def library(tmp_path):
    value = NecromatcherLibrary.create(tmp_path / "library")
    value.add_player("hogan", "Ben Hogan")
    value.add_swing("practice", "hogan", "Practice Swing")
    return value


def test_player_and_swing_survive_restart(library):
    fresh = NecromatcherLibrary(library.root)
    assert fresh.players()[0].display_name == "Ben Hogan"
    assert fresh.swings("hogan")[0].session_id == "practice"
    with pytest.raises(StateError, match="already exists"):
        fresh.add_player("hogan", "Different Identity")


def test_model_version_copies_and_checks_bytes(library, tmp_path):
    source = tmp_path / "golfer.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    model = library.add_model(
        "hogan-v1", "practice", source, engine="mujoco", dofs=("hip", "knee")
    )
    source.unlink()
    fresh = NecromatcherLibrary(library.root)
    assert fresh.load_asset("hogan-v1") == model
    Path(model.path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        fresh.load_asset("hogan-v1")


def test_version_overwrite_rejected(library, tmp_path):
    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip",))
    with pytest.raises(StateError, match="already exists"):
        library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip",))


def test_profile_binds_model_order_units_and_physical_clock(library, tmp_path):
    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip", "knee"))
    profile = {
        "schema_version": "necromatcher/torque-profile/1",
        "model_id": "v1",
        "dofs": ["hip", "knee"],
        "units": "N*m",
        "timebase": "physical_seconds",
        "provenance": {"kind": "authored", "description": "Test authored controls"},
        "segments": [
            {
                "start_s": 0.0,
                "end_s": 1.0,
                "coefficients": [[1, 3], [2, 4]],
                "is_bernstein": True,
            }
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    library.add_profile("controls-v1", "practice", path)
    torque = NecromatcherLibrary(library.root).load_torque("controls-v1", "v1")
    assert torque.evaluate(0.5).tolist() == [2, 3]
    for field, invalid in [
        ("dofs", ["knee", "hip"]),
        ("units", "pixels"),
        ("timebase", "presentation_seconds"),
    ]:
        bad = dict(profile, **{field: invalid})
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            library.add_profile("invalid", "practice", path)
    assert {x.dataset_id for x in library.assets("practice")} == {"v1", "controls-v1"}


def test_cross_swing_profile_rejected(library, tmp_path):
    library.add_player("tiger", "Tiger Woods")
    library.add_swing("tiger-drive", "tiger", "Drive")
    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip",))
    profile = {
        "schema_version": "necromatcher/torque-profile/1",
        "model_id": "v1",
        "dofs": ["hip"],
        "units": "N*m",
        "timebase": "physical_seconds",
        "provenance": {"kind": "authored", "description": "Controls"},
        "segments": [
            {"start_s": 0, "end_s": 1, "coefficients": [[1]], "is_bernstein": True}
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="session"):
        library.add_profile("wrong-player", "tiger-drive", path)


def test_capture_import_rejects_unbound_observations(library, tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "historical-capture/1.0.0",
                "subject_id": "hogan",
                "qualification": "image_observations_only",
                "physical_time_verified": False,
                "observations_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    (capture / "observations.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        library.add_capture("capture-v1", "practice", capture)
    assert library.assets("practice") == []


def test_profile_cannot_bind_another_model_revision(library, tmp_path):
    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip",))
    library.add_model("v2", "practice", source, engine="mujoco", dofs=("hip",))
    profile = {
        "schema_version": "necromatcher/torque-profile/1",
        "model_id": "v1",
        "dofs": ["hip"],
        "units": "N*m",
        "timebase": "physical_seconds",
        "provenance": {"kind": "authored", "description": "Controls"},
        "segments": [
            {"start_s": 0, "end_s": 1, "coefficients": [[1]], "is_bernstein": True}
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    library.add_profile("controls", "practice", path)
    with pytest.raises(ValueError, match="revision"):
        library.load_torque("controls", "v2")


def test_library_rejects_concurrent_writer(library):
    lock = library.root / ".necromatcher-write.lock"
    lock.write_text("other writer", encoding="utf-8")
    with pytest.raises(StateError, match="write in progress"):
        library.add_player("tiger", "Tiger Woods")
    assert lock.read_text(encoding="utf-8") == "other writer"


def test_capture_archive_preserves_frames_and_rational_pts(library, tmp_path):
    import numpy as np
    from zipfile import ZipFile

    av = pytest.importorskip("av")
    pytest.importorskip("cv2")
    from src.shared.python.shadow_tracker.historical_capture import (
        CaptureWindow,
        export_capture,
    )
    from src.shared.python.pose_estimation.interface import PoseEstimationResult

    video = tmp_path / "source.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class MissingEstimator:
        def estimate_from_image(self, image, timestamp_ms):
            return PoseEstimationResult({}, 0, 0)

    capture = tmp_path / "capture"
    receipt = export_capture(
        video,
        capture,
        CaptureWindow(start_s=0, end_s=0.3),
        subject_id="hogan",
        estimator=MissingEstimator(),
        detector_identity={"name": "synthetic-test-only"},
    )
    saved = library.add_capture("capture-v1", "practice", capture)
    assert saved.metadata["qualification"] == "image_observations_only"
    assert saved.metadata["frame_count"] == 3
    with ZipFile(saved.path) as archive:
        assert json.loads(archive.read("receipt.json")) == receipt
        assert (
            archive.read("observations.jsonl")
            == (capture / "observations.jsonl").read_bytes()
        )
        assert len([x for x in archive.namelist() if x.endswith(".png")]) == 3
    (capture / "observations.jsonl").write_text("changed", encoding="utf-8")
    assert NecromatcherLibrary(library.root).load_asset("capture-v1") == saved


def test_export_is_portable_and_refuses_corrupted_assets(library, tmp_path):
    from zipfile import ZipFile

    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    model = library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip",))
    destination = tmp_path / "swing.zip"
    library.export_swing("practice", destination)
    with ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["player"]["display_name"] == "Ben Hogan"
        asset = manifest["assets"][0]
        assert asset["path"] == "assets/v1.xml"
        assert asset["metadata"]["qualification"] == "unqualified_candidate"
        assert archive.read(asset["path"]) == source.read_bytes()
    with pytest.raises(FileExistsError):
        library.export_swing("practice", destination)
    Path(model.path).unlink()
    refused = tmp_path / "bad.zip"
    with pytest.raises(FileNotFoundError):
        library.export_swing("practice", refused)
    assert not refused.exists()


def test_copy_failure_does_not_publish_partial_asset(library, tmp_path, monkeypatch):
    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")

    def refuse(*args, **kwargs):
        raise StateError("Interrupted metadata save")

    monkeypatch.setattr(library._store, "register_dataset", refuse)
    with pytest.raises(StateError, match="Interrupted"):
        library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip",))
    assert not (library.root / "assets" / "v1.xml").exists()
    assert library.assets("practice") == []


def test_export_rejects_source_changed_during_copy(library, tmp_path, monkeypatch):
    from zipfile import ZipFile

    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    model = library.add_model("v1", "practice", source, engine="mujoco", dofs=("hip",))
    original = ZipFile.write

    def change_then_copy(archive, path, *args, **kwargs):
        Path(model.path).write_text("changed while exporting", encoding="utf-8")
        return original(archive, path, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "write", change_then_copy)
    output = tmp_path / "swing.zip"
    with pytest.raises(ValueError, match="hash mismatch"):
        library.export_swing("practice", output)
    assert not output.exists()


def test_malformed_profile_shape_rejected_without_version(library, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        library.add_profile("invalid", "practice", path)
    assert library.assets("practice") == []
