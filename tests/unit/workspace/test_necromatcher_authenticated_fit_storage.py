"""Fit persistence closes fresh capture authentication before copying bytes."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import pytest

from hypothesis_fixture import imported_capture
from test_scope_fixtures import fixture_review_artifact
from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.workspace import necromatcher as owner
from src.shared.python.workspace import necromatcher_fit as fit
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity
from src.shared.python.workspace.necromatcher_source_scope import bind_source_fit_scope

pytestmark = pytest.mark.unit


@pytest.fixture
def scoped_storage(fit_case: Any, tmp_path: Path) -> tuple[Any, Path, dict[str, Any]]:
    library, source, payload = fit_case
    capture = imported_capture(library, tmp_path / "original-images")
    identity = capture_identity(library, capture.dataset_id)
    reference = fixture_review_artifact(tmp_path, identity, 0, 3)
    library.add_source_scope_review("review", "practice", Path(reference.path))
    scope = library.load_source_scope_review("review")
    config = asdict(ImageFitConfig())
    payload.update(
        capture_id=identity.capture_id,
        capture_hash=identity.capture_hash,
        frame_indices=[0, 1, 2],
        frames=[frame.to_dict() for frame in identity.frames],
        q=[[0.1], [0.15], [0.2]],
    )
    payload["provenance"].update(
        source_fit_scope=scope.to_record(),
        source_fit_scope_binding=fit.scope_binding_record(
            bind_source_fit_scope(identity, scope, library.root), (0, 2)
        ),
        request_options={"config": config},
    )
    payload["evidence"]["original_fit"] = {
        "frame_indices": [0, 2],
        "source_times": [0.0, 0.2],
        "config": config,
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    return library, source, payload


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_close_mutation_prevents_any_fit_persistence(
    scoped_storage: Any, monkeypatch: Any
) -> None:
    library, source, payload = scoped_storage
    capture = Path(library.load_asset(payload["capture_id"]).path)
    before = snapshot(library.root)
    reader = owner.read_kinematic_fit
    saved = []

    def mutate_after_validation(*args: Any) -> Any:
        result = reader(*args)
        stat = capture.stat()
        raw = bytearray(capture.read_bytes())
        raw[-10] ^= 1
        capture.write_bytes(raw)
        os.utime(capture, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return result

    monkeypatch.setattr(owner, "read_kinematic_fit", mutate_after_validation)
    monkeypatch.setattr(library, "_save_asset", lambda *a, **k: saved.append(a))
    with pytest.raises(ValueError, match="hash|changed"):
        library.add_fit("new-fit", "practice", source)
    assert saved == []
    after = snapshot(library.root)
    assert after.keys() == before.keys()
    assert {
        k: v for k, v in after.items() if k != str(capture.relative_to(library.root))
    } == {
        k: v for k, v in before.items() if k != str(capture.relative_to(library.root))
    }


def test_scoped_save_authenticates_once_and_closes_before_save(
    scoped_storage: Any, monkeypatch: Any
) -> None:
    import cv2

    library, source, payload = scoped_storage
    decode, save = cv2.imdecode, library._save_asset
    calls = []

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return decode(*args, **kwargs)

    def save_after_close(*args: Any, **kwargs: Any) -> Any:
        with library.authenticated_read():
            pass
        return save(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecode", counted)
    monkeypatch.setattr(library, "_save_asset", save_after_close)
    raw = source.read_bytes()
    asset = library.add_fit("new-fit", "practice", source)
    assert len(calls) == 3
    assert Path(asset.path).read_bytes() == raw
    assert asset.metadata["capture_id"] == payload["capture_id"]
    assert asset.metadata["physical_time_qualified"] is False


def test_legacy_save_retains_bytes_without_full_capture_decode(
    fit_case: Any, monkeypatch: Any
) -> None:
    import cv2

    library, source, _ = fit_case
    monkeypatch.setattr(
        cv2, "imdecode", lambda *_: pytest.fail("Legacy decoded capture")
    )
    raw = source.read_bytes()
    asset = library.add_fit("legacy", "practice", source)
    assert Path(asset.path).read_bytes() == raw
    assert (
        library.load_fit("legacy")["qualification"] == "monocular_research_hypothesis"
    )


@pytest.mark.parametrize("fault", ["receipt", "widen"])
def test_invalid_scope_still_rejects_before_save(
    scoped_storage: Any, fault: str
) -> None:
    library, source, payload = scoped_storage
    if fault == "receipt":
        receipt = library.load_asset("review")
        Path(receipt.path).write_bytes(b"{}")
    else:
        payload["provenance"]["source_fit_scope"]["end_exclusive_frame"] = 4
        source.write_text(json.dumps(payload), encoding="utf-8")
    before = snapshot(library.root)
    with pytest.raises(ValueError):
        library.add_fit("invalid", "practice", source)
    assert snapshot(library.root) == before
