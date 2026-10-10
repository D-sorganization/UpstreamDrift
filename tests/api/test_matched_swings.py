"""API tests for matched-swing results routes (MS-85, #10358)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.routes.matched_swings import get_matched_swings_service
from src.api.server import app
from src.api.services.matched_swings_service import MatchedSwingsService
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMetadata,
    CandidateProfile,
)
from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    Ledger,
    LedgerRow,
    SharedMetrics,
)

pytestmark = pytest.mark.unit


def _build_matched_swing_ledger(
    tmp_path: Path,
    *,
    with_target_markers: bool = False,
    n_frames: int = 2,
) -> tuple[Path, str]:
    """Build a minimal ledger with receipt, npz, gif, and parity artefacts.

    ``with_target_markers`` additionally writes ``target_markers_m`` (same
    shape as ``model_markers_m``) so :func:`export_video` can render real
    GIF/MP4 frames (desktop parity, #11987). The default omits it, which
    exercises ``export_video``'s marker-trajectory validation error path
    (``ValueError``, never an absolute path in the message).
    """
    evidence = tmp_path / "evidence" / "matched" / "driver_test"
    evidence.mkdir(parents=True)

    marker_names = ("pelvis", "thorax", "head")
    time_s = np.linspace(0.0, 0.1 * (n_frames - 1), n_frames)
    base = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.3, 1.2], [0.0, 0.5, 1.5]], dtype=np.float64
    )
    offsets = np.arange(n_frames, dtype=np.float64).reshape(-1, 1, 1) * 0.01
    markers = base[np.newaxis, :, :] + offsets

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine="pinocchio",
        model_name="test_model",
        marker_names=marker_names,
    )
    npz_path = evidence / "candidate.npz"
    arrays: dict[str, np.ndarray] = {
        "manifest_json": np.array(json.dumps(meta.to_dict())),
        "time_s": time_s,
        "q": np.zeros((n_frames, 1)),
        "model_markers_m": markers,
    }
    if with_target_markers:
        arrays["target_markers_m"] = markers + 0.001
    np.savez_compressed(npz_path, **arrays)

    receipt = {
        "schema_version": "matched-swing-fit/test-v1",
        "engine": "pinocchio",
        "candidate_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    receipt_path = evidence / "receipt.json"
    receipt_bytes = json.dumps(receipt).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()

    gif_path = evidence / "playback.gif"
    gif_path.write_bytes(
        b"GIF89a\x01\x00\x01\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    )

    parity = {
        "schema_version": "matched-swing-parity-report-v1",
        "candidate_sha256": receipt["candidate_sha256"],
        "status": "PARTIAL",
    }
    (evidence / "parity_vs_mujoco.json").write_text(
        json.dumps(parity), encoding="utf-8"
    )

    rel_receipt = receipt_path.relative_to(tmp_path).as_posix()
    row = LedgerRow(
        receipt_path=rel_receipt,
        sha256=receipt_sha,
        engine="pinocchio",
        lane="matched",
        capture="driver",
        candidate_sha=receipt["candidate_sha256"],
        horizon_s=0.85,
        metrics=SharedMetrics(whole_marker_rmse_m=0.022),
        acceptance={"status": "PASSED", "gates": [{"name": "G1", "passed": True}]},
        artefacts=ArtefactPaths(
            npz=npz_path.relative_to(tmp_path).as_posix(),
            gif=gif_path.relative_to(tmp_path).as_posix(),
        ),
    )
    ledger = Ledger(
        schema_version="1.0.0",
        generated_at="2026-09-21T00:00:00Z",
        total_receipts=1,
        rows=[row],
    )
    ledger_path = tmp_path / "reports" / "matched_swing_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(ledger.to_json(), encoding="utf-8")
    return ledger_path, receipt_sha


@pytest.fixture
def ledger_fixture(tmp_path: Path) -> tuple[Path, str]:
    """Minimal ledger whose candidate lacks target_markers_m (no video)."""
    return _build_matched_swing_ledger(tmp_path)


@pytest.fixture
def ledger_fixture_with_video(tmp_path: Path) -> tuple[Path, str]:
    """Ledger fixture whose candidate carries target_markers_m, 3 frames, so
    GET /video can render a real GIF/MP4 (#11987)."""
    return _build_matched_swing_ledger(tmp_path, with_target_markers=True, n_frames=3)


@pytest.fixture
def client(ledger_fixture: tuple[Path, str]) -> TestClient:
    ledger_path, _ = ledger_fixture
    service = MatchedSwingsService.from_ledger_file(
        ledger_path, repo_root=ledger_path.parent.parent
    )
    app.dependency_overrides[get_matched_swings_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_matched_swings_service, None)


@pytest.fixture
def video_client(ledger_fixture_with_video: tuple[Path, str]) -> TestClient:
    ledger_path, _ = ledger_fixture_with_video
    service = MatchedSwingsService.from_ledger_file(
        ledger_path, repo_root=ledger_path.parent.parent
    )
    app.dependency_overrides[get_matched_swings_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_matched_swings_service, None)


def test_list_matched_swings(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get("/api/matched-swings")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["runs"][0]["id"] == run_id
    assert data["runs"][0]["verdict"] == "PASSED"
    assert data["runs"][0]["capabilities"]["has_candidate_npz"] is True
    assert "receipt_path" not in json.dumps(data)
    assert data["runs"][0]["gates"] == [
        {"name": "G1", "status": "", "measured": None, "threshold": None, "unit": "m"}
    ]


def test_run_summary_gates_empty_without_acceptance_gates(tmp_path: Path) -> None:
    """A run whose acceptance block has no "gates" key reports an empty list
    (unavailable, not a fabricated zero/blank entry)."""
    ledger_path = tmp_path / "reports" / "matched_swing_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    row = LedgerRow(
        receipt_path="evidence/receipt.json",
        sha256="a" * 64,
        engine="mujoco",
        lane="matched",
    )
    ledger = Ledger(
        schema_version="1.0.0",
        generated_at="2026-09-21T00:00:00Z",
        total_receipts=1,
        rows=[row],
    )
    ledger_path.write_text(ledger.to_json(), encoding="utf-8")
    service = MatchedSwingsService.from_ledger_file(ledger_path, repo_root=tmp_path)

    summary = service.get_run_summary(row.sha256)

    assert summary.gates == []
    assert summary.to_dict()["gates"] == []


def test_run_summary_gate_values_are_finite_floats_or_none(tmp_path: Path) -> None:
    """Non-numeric, boolean and non-finite gate values are unavailable (None)."""
    ledger_path = tmp_path / "reports" / "matched_swing_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    row = LedgerRow(
        receipt_path="evidence/receipt.json",
        sha256="b" * 64,
        engine="mujoco",
        lane="matched",
        acceptance={
            "gates": [
                {"name": "G1", "status": "pass", "measured": 2, "threshold": 0.05},
                {"name": "G2", "measured": "n/a", "threshold": True},
                {"name": "G3", "measured": float("nan"), "threshold": None},
            ]
        },
    )
    ledger = Ledger(
        schema_version="1.0.0",
        generated_at="2026-09-21T00:00:00Z",
        total_receipts=1,
        rows=[row],
    )
    ledger_path.write_text(ledger.to_json(), encoding="utf-8")
    service = MatchedSwingsService.from_ledger_file(ledger_path, repo_root=tmp_path)

    gates = service.get_run_summary(row.sha256).gates

    assert [(g.name, g.status, g.measured, g.threshold) for g in gates] == [
        ("G1", "PASS", 2.0, 0.05),
        ("G2", "", None, None),
        ("G3", "", None, None),
    ]


@pytest.mark.parametrize(
    ("acceptance", "expected_note"),
    [
        # Gates recorded: the desktop label shows the qualification note.
        (
            {"gates": [], "qualification_note": "Independent uninterrupted replay"},
            "Independent uninterrupted replay",
        ),
        # No acceptance block: unavailable, not a fabricated blank string.
        (None, None),
        # No gates block: ``_populate_gates_info`` returns early with ``reason``.
        ({"qualification_note": "note without gates"}, None),
    ],
)
def test_run_summary_qualification_note_matches_desktop_label(
    tmp_path: Path,
    acceptance: dict[str, object] | None,
    expected_note: str | None,
) -> None:
    """qualification_note follows ``MatchedSwingBrowserWidget._populate_gates_info``
    (gui.py); ``reason`` is always passed through unchanged."""
    ledger_path = tmp_path / "reports" / "matched_swing_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    row = LedgerRow(
        receipt_path="evidence/receipt.json",
        sha256="c" * 64,
        engine="mujoco",
        lane="matched",
        acceptance=acceptance,
        reason="the reason",
    )
    ledger = Ledger(
        schema_version="1.0.0",
        generated_at="2026-09-21T00:00:00Z",
        total_receipts=1,
        rows=[row],
    )
    ledger_path.write_text(ledger.to_json(), encoding="utf-8")
    service = MatchedSwingsService.from_ledger_file(ledger_path, repo_root=tmp_path)

    summary = service.get_run_summary(row.sha256)

    assert summary.qualification_note == expected_note
    assert summary.to_dict()["qualification_note"] == expected_note
    assert summary.reason == "the reason"


def test_get_receipt(client: TestClient, ledger_fixture: tuple[Path, str]) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run_id
    assert body["receipt"]["engine"] == "pinocchio"
    assert body["candidate_sha256"]


def test_get_candidate_npz(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/candidate")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")


def test_get_candidate_preview(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(
        f"/api/matched-swings/{run_id}/candidate",
        params={"preview_frame": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["frame_count"] == 2
    assert len(body["joints"]) == 3
    assert body["joints"][0]["name"] == "pelvis"


def test_get_parity(client: TestClient, ledger_fixture: tuple[Path, str]) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/parity")
    assert response.status_code == 200
    assert response.json()["schema_version"] == "matched-swing-parity-report-v1"


def test_get_report_markdown(
    client: TestClient, ledger_fixture: tuple[Path, str], tmp_path: Path
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/report")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert (
        'filename="fit_report_pinocchio_driver.md"'
        in response.headers["content-disposition"]
    )
    body = response.text
    assert run_id in body
    assert "pinocchio" in body.lower()
    # Public responses never expose absolute filesystem paths (service contract).
    assert str(tmp_path) not in body
    assert tmp_path.as_posix() not in body


def test_get_report_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/matched-swings/not-a-real-id/report")
    assert response.status_code == 404


def test_get_report_missing_receipt_returns_error(
    client: TestClient, ledger_fixture: tuple[Path, str], tmp_path: Path
) -> None:
    _, run_id = ledger_fixture
    receipt_path = tmp_path / "evidence" / "matched" / "driver_test" / "receipt.json"
    receipt_path.unlink()

    response = client.get(f"/api/matched-swings/{run_id}/report")

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "report_unavailable"


def test_get_report_pdf(client: TestClient, ledger_fixture: tuple[Path, str]) -> None:
    _, run_id = ledger_fixture
    response = client.get(
        f"/api/matched-swings/{run_id}/report", params={"format": "pdf"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert (
        'filename="fit_report_pinocchio_driver.pdf"'
        in response.headers["content-disposition"]
    )


def test_get_report_no_format_param_is_still_markdown(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/report")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")


def test_get_report_pdf_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get(
        "/api/matched-swings/not-a-real-id/report", params={"format": "pdf"}
    )
    assert response.status_code == 404


def test_get_video_unavailable_returns_404_without_absolute_path(
    client: TestClient, ledger_fixture: tuple[Path, str], tmp_path: Path
) -> None:
    """The default fixture's candidate lacks target_markers_m, so export_video
    raises ValueError; the 404 body must never leak the temp export path."""
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/video")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["error"]["code"] == "video_unavailable"
    body_text = json.dumps(body)
    assert str(tmp_path) not in body_text
    assert tmp_path.as_posix() not in body_text


def test_get_video_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/matched-swings/not-a-real-id/video")
    assert response.status_code == 404


def test_get_video_gif(
    video_client: TestClient, ledger_fixture_with_video: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture_with_video
    response = video_client.get(f"/api/matched-swings/{run_id}/video")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/gif"
    assert response.content.startswith(b"GIF8")
    assert 'filename="pinocchio_driver.gif"' in response.headers["content-disposition"]


def test_get_video_mp4(
    video_client: TestClient, ledger_fixture_with_video: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture_with_video
    response = video_client.get(
        f"/api/matched-swings/{run_id}/video", params={"format": "mp4"}
    )
    if response.status_code != 200:
        pytest.skip(
            f"OpenCV mp4v VideoWriter unavailable on this host: {response.text}"
        )
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 0
    assert 'filename="pinocchio_driver.mp4"' in response.headers["content-disposition"]


def test_get_video_invalid_format_returns_422(
    video_client: TestClient, ledger_fixture_with_video: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture_with_video
    response = video_client.get(
        f"/api/matched-swings/{run_id}/video", params={"format": "avi"}
    )
    assert response.status_code == 422


def test_get_animation_gif(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/animation.gif")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/gif"


def test_get_animation_frame_info(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/animation/frames")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "matched-swing-animation/1"
    assert body["frame_count"] == 1
    assert body["durations_ms"] == [100]
    assert body["width"] == 1
    assert body["height"] == 1


def test_get_animation_frame_png(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/animation/frames/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_get_animation_frame_png_out_of_range_returns_404(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/animation/frames/7")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "animation_unavailable"


def test_get_animation_frame_info_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/matched-swings/not-a-real-id/animation/frames")
    assert response.status_code == 404


def test_get_animation_frame_png_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/matched-swings/not-a-real-id/animation/frames/0")
    assert response.status_code == 404


def test_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/matched-swings/not-a-real-id")
    assert response.status_code == 404


def test_remote_client_blocked(ledger_fixture: tuple[Path, str]) -> None:
    ledger_path, run_id = ledger_fixture
    service = MatchedSwingsService.from_ledger_file(
        ledger_path, repo_root=ledger_path.parent.parent
    )
    app.dependency_overrides[get_matched_swings_service] = lambda: service
    try:
        remote_client = TestClient(app, client=("203.0.113.1", 1234))
        response = remote_client.get("/api/matched-swings")
        assert response.status_code == 403
        assert (
            remote_client.get(
                f"/api/matched-swings/{run_id}/animation/frames"
            ).status_code
            == 403
        )
        assert (
            remote_client.get(
                f"/api/matched-swings/{run_id}/animation/frames/0"
            ).status_code
            == 403
        )
        assert (
            remote_client.get(f"/api/matched-swings/{run_id}/report").status_code == 403
        )
        assert (
            remote_client.get(f"/api/matched-swings/{run_id}/video").status_code == 403
        )
    finally:
        app.dependency_overrides.pop(get_matched_swings_service, None)
