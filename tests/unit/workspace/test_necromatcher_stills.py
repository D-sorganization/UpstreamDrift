"""Selected source stills reuse canonical video composition without codecs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from tests.unit.workspace import test_necromatcher_video as video_fixtures
from tests.unit.workspace import test_necromatcher_video_shaft as shaft_fixtures

video_case = video_fixtures.video_case
shaft_case = shaft_fixtures.shaft_case
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("layers", [False, True])
def test_stills_match_video_pixels_and_diagnostics_without_codec(
    shaft_case, tmp_path, monkeypatch, layers
):
    import cv2
    from src.shared.python.workspace import necromatcher_video as module
    from src.shared.python.workspace import CaptionOverlayOptions, ShapeOverlayOptions

    library, evidence = shaft_case
    options = (
        {
            "shaft_evidence": evidence,
            "shape_overlay": ShapeOverlayOptions(0.35),
            "caption_overlay": CaptionOverlayOptions(),
        }
        if layers
        else {}
    )
    video = module.export_fit_video(
        library, "video-fit", tmp_path / "video", selected_frames=(0, 1), **options
    )
    original = module._render_layers
    calls = []

    def render(*args):
        calls.append(args[2])
        return original(*args)

    def forbidden(*args, **kwargs):
        pytest.fail("Stills must never invoke a video codec")

    monkeypatch.setattr(module, "_render_layers", render)
    monkeypatch.setattr(cv2, "VideoWriter", forbidden)
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    out = tmp_path / "stills"
    result = module.export_fit_stills(
        library, "video-fit", out, selected_frames=(1, 0), **options
    )
    assert calls == [1, 0]
    assert result["schema"] == "necromatcher/source-overlay-stills/1"
    assert "video" not in result and "source_frame_rate" not in result
    for row in result["frames"]:
        index = row["frame_index"]
        assert row == video["frames"][index]
        name = f"frame-{index:06d}.png"
        np.testing.assert_array_equal(
            cv2.imread(str(out / name)), cv2.imread(str(tmp_path / "video" / name))
        )
    assert result["frame_pts"] == [
        {"frame_index": 1, "numerator": 1, "denominator": 30},
        {"frame_index": 0, "numerator": 0, "denominator": 1},
    ]
    if layers:
        assert result["frames"][0]["shaft_overlay"]["status"] == "ambiguous"
        assert result["shaft_overlay"] == video["shaft_overlay"]
        assert result["shape_overlay"] == video["shape_overlay"]
        assert result["caption_overlay"] == video["caption_overlay"]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == result
    assert not (out / "overlay.mp4").exists()


@pytest.mark.parametrize("selected", [(), (0, 0), (True,), (-1,), (2,), (0.0,)])
def test_invalid_selection_rejected_before_render(
    video_case, tmp_path, monkeypatch, selected
):
    from src.shared.python.workspace import necromatcher_video as module

    monkeypatch.setattr(
        module, "_render_layers", lambda *a: pytest.fail("Invalid selection rendered")
    )
    with pytest.raises(ValueError, match="[Ss]elected"):
        module.export_fit_stills(
            video_case[0], "video-fit", tmp_path / "bad", selected_frames=selected
        )
    assert not (tmp_path / "bad").exists()


def test_selected_only_preserves_one_binding_source_bytes_and_output_hashes(
    video_case, tmp_path, monkeypatch
):
    from src.shared.python.workspace import necromatcher_video as module

    library, capture, _ = video_case
    before = Path(capture.path).read_bytes()
    original = module.load_native_fit_binding
    loads = []

    def load(*args):
        loads.append(args)
        return original(*args)

    monkeypatch.setattr(module, "load_native_fit_binding", load)
    out = tmp_path / "one"
    result = module.export_fit_stills(library, "video-fit", out, selected_frames=(1,))
    assert len(loads) == 1 and len(result["frames"]) == 1
    assert Path(capture.path).read_bytes() == before
    assert result["physical_time_qualified"] is False
    assert result["qualification"] == "monocular_research_hypothesis"
    for item in result["pngs"]:
        assert (
            hashlib.sha256((out / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    with pytest.raises(FileExistsError):
        module.export_fit_stills(library, "video-fit", out, selected_frames=(1,))


def test_nonuniform_pts_supported_without_frame_rate_inference(video_case, tmp_path):
    from src.shared.python.workspace import necromatcher_video as module

    library, capture, payload = video_case
    changed = tmp_path / "irregular.zip"
    with ZipFile(capture.path) as original, ZipFile(changed, "w") as target:
        rows = [
            json.loads(line)
            for line in original.read("observations.jsonl").splitlines()
        ]
        rows[1]["frame"]["pts_ticks"] = 7
        rows[1]["frame"]["timebase_denominator"] = 30000
        rows.append(deepcopy(rows[1]))
        rows[2]["frame"].update(frame_id="video-2", pts_ticks=1001)
        rows[2]["image"] = "frame-2.png"
        target.writestr("frame-2.png", original.read("frame-1.png"))
        for name in original.namelist():
            if name == "receipt.json":
                receipt = json.loads(original.read(name))
                receipt["frame_count"] = 3
                target.writestr(name, json.dumps(receipt))
                continue
            target.writestr(
                name,
                "\n".join(json.dumps(r) for r in rows)
                if name == "observations.jsonl"
                else original.read(name),
            )
    asset = library._save_asset(
        "irregular",
        "practice",
        changed,
        "image_capture",
        {k: v for k, v in capture.metadata.items() if k != "hash"},
    )
    payload.update(
        capture_id=asset.dataset_id,
        capture_hash=asset.metadata["hash"],
        frames=[r["frame"] for r in rows],
        frame_indices=[0, 1, 2],
        q=[*payload["q"], payload["q"][-1]],
    )
    source = tmp_path / "irregular-fit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    library.add_fit("irregular-fit", "practice", source)
    result = module.export_fit_stills(
        library, "irregular-fit", tmp_path / "irregular-stills", selected_frames=(2, 1)
    )
    assert result["frame_pts"] == [
        {"frame_index": 2, "numerator": 1001, "denominator": 30000},
        {"frame_index": 1, "numerator": 7, "denominator": 30000},
    ]
    with pytest.raises(ValueError, match="uniform"):
        module.export_fit_video(library, "irregular-fit", tmp_path / "cannot-video")


@pytest.mark.parametrize("option", ["shape_overlay", "caption_overlay"])
def test_options_require_typed_contract_before_native_loading(
    video_case, tmp_path, monkeypatch, option
):
    from src.shared.python.workspace import necromatcher_video as module

    monkeypatch.setattr(
        module,
        "load_native_fit_binding",
        lambda *a: pytest.fail("Invalid options compiled a model"),
    )
    with pytest.raises(TypeError, match="typed"):
        module.export_fit_stills(
            video_case[0],
            "video-fit",
            tmp_path / "bad-option",
            selected_frames=(0,),
            **{option: {}},
        )


def test_changed_frame_identity_fails_before_publication(
    video_case, tmp_path, monkeypatch
):
    from src.shared.python.workspace import necromatcher_video as module

    original = module.CaptureReview.frame

    def frame(self, index):
        result = original(self, index)
        result["frame"]["pts_ticks"] += 1
        return result

    monkeypatch.setattr(module.CaptureReview, "frame", frame)
    with pytest.raises(ValueError, match="frame identity mismatch"):
        module.export_fit_stills(
            video_case[0], "video-fit", tmp_path / "wrong-frame", selected_frames=(0,)
        )
    assert not (tmp_path / "wrong-frame").exists()


def test_failed_lossless_png_readback_cannot_publish(video_case, tmp_path, monkeypatch):
    import cv2
    from src.shared.python.workspace import necromatcher_video as module

    monkeypatch.setattr(cv2, "imread", lambda *a, **k: None)
    with pytest.raises(ValueError, match="readback"):
        module.export_fit_stills(
            video_case[0], "video-fit", tmp_path / "unreadable", selected_frames=(0,)
        )
    assert not (tmp_path / "unreadable").exists()


@pytest.mark.parametrize("tamper", ["fit", "capture", "model"])
def test_changed_authoritative_bytes_during_render_cannot_publish(
    video_case, tmp_path, monkeypatch, tamper
):
    from src.shared.python.workspace import necromatcher_video as module

    library, capture, _ = video_case
    original = module._render_layers

    def render(*args):
        result = original(*args)
        if tamper == "capture":
            path = Path(capture.path)
        else:
            asset_id = args[0].model_id if tamper == "model" else "video-fit"
            path = Path(library.load_asset(asset_id).path)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(module, "_render_layers", render)
    with pytest.raises((ValueError, OSError)):
        module.export_fit_stills(
            library, "video-fit", tmp_path / "changed", selected_frames=(0,)
        )
    assert not (tmp_path / "changed").exists()


def test_still_facade_is_sdk_free_in_clean_process():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.shared.python.workspace import export_fit_stills; import sys; assert 'mujoco' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
