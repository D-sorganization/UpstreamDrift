"""Web/API surface tests for comparable-candidate filtering and honest residuals (MMR-16, #11102).

Validates:
1. MatchedSwingsService supports comparable candidate ranking by ascending RMSE.
2. Honest rejections are preserved without masking failure verdicts.
3. Candidate preview frames return observed dots alongside fitted positions.
4. Residual summaries are accessible via public service APIs for web consumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from src.api.services.matched_swings_service import (
    MatchedSwingsService,
    RunSummary,
)
from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    LedgerRow,
    SharedMetrics,
)

pytestmark = pytest.mark.unit


def _make_row(
    receipt_path: str,
    engine: str,
    rmse: float,
    *,
    capture: str = "driver",
    lane: str = "matched",
    drive_mode: str = "torque_driven",
    status: str = "PASSED",
    candidate_sha: str = "abc12345",
    sha256: str = "f" * 64,
) -> LedgerRow:
    return LedgerRow(
        receipt_path=receipt_path,
        sha256=sha256,
        engine=engine,
        lane=lane,
        capture=capture,
        candidate_sha=candidate_sha,
        metrics=SharedMetrics(whole_marker_rmse_m=rmse),
        artefacts=ArtefactPaths(npz=f"evidence/{engine}_replay.npz"),
        acceptance={"status": status, "is_physically_accepted": (status == "PASSED")},
        reason=f"drive_mode:{drive_mode}",
    )


class TestMatchedSwingsServiceBestCandidate:
    def test_list_runs_ranked_filters_and_sorts_by_rmse(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            _make_row("r1.json", "mujoco", 0.024, capture="driver", sha256="1" * 64),
            _make_row("r2.json", "pinocchio", 0.011, capture="driver", sha256="2" * 64),
            _make_row("r3.json", "drake", 0.017, capture="driver", sha256="3" * 64),
            _make_row("r4.json", "simscape", 0.009, capture="iron", sha256="4" * 64),
        ]
        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", lambda: list(rows))

        # Ranked for driver capture
        ranked = service.list_runs(capture="driver", ranked=True)
        assert len(ranked) == 3
        assert ranked[0].engine == "pinocchio"
        assert ranked[0].metrics["whole_marker_rmse_m"] == 0.011
        assert ranked[1].engine == "drake"
        assert ranked[2].engine == "mujoco"

    def test_list_runs_ranked_honestly_preserves_rejection(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            _make_row("r1.json", "mujoco", 0.005, status="REJECTED", sha256="1" * 64),
            _make_row("r2.json", "drake", 0.015, status="PASSED", sha256="2" * 64),
        ]
        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", lambda: list(rows))

        ranked = service.list_runs(capture="driver", ranked=True)
        assert len(ranked) == 2
        assert ranked[0].engine == "mujoco"
        assert ranked[0].verdict == "REJECTED"
        assert ranked[1].engine == "drake"
        assert ranked[1].verdict == "PASSED"

    def test_list_runs_filters_by_profile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """profile reuses MatchedSwingBrowserModel.filter_rows's dynamic/
        kinematic rule (DRY — #GCV-5)."""
        rows = [
            _make_row(
                "r1.json",
                "mujoco",
                0.02,
                lane="matched",
                drive_mode="torque_driven",
                sha256="1" * 64,
            ),
            _make_row(
                "r2.json",
                "pinocchio",
                0.03,
                lane="tour_matching",
                drive_mode="kinematic_prescribed",
                sha256="2" * 64,
            ),
        ]
        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", lambda: list(rows))

        dynamic_only = service.list_runs(profile="dynamic")
        assert [r.engine for r in dynamic_only] == ["mujoco"]

        kinematic_only = service.list_runs(profile="kinematic")
        assert [r.engine for r in kinematic_only] == ["pinocchio"]

    def test_fastapi_route_profile_filter_combined_with_ranked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.routes.matched_swings import router, get_matched_swings_service

        rows = [
            _make_row(
                "r1.json",
                "mujoco",
                0.02,
                capture="driver",
                lane="matched",
                drive_mode="torque_driven",
                sha256="1" * 64,
            ),
            _make_row(
                "r2.json",
                "pinocchio",
                0.01,
                capture="driver",
                lane="tour_matching",
                drive_mode="kinematic_prescribed",
                sha256="2" * 64,
            ),
            _make_row(
                "r3.json",
                "drake",
                0.005,
                capture="driver",
                lane="matched",
                drive_mode="torque_driven",
                sha256="3" * 64,
            ),
        ]
        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", lambda: list(rows))

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_matched_swings_service] = lambda: service
        client = TestClient(app)

        res = client.get("/matched-swings?profile=dynamic&ranked=true")
        assert res.status_code == 200
        data = res.json()
        assert [r["engine"] for r in data["runs"]] == ["drake", "mujoco"]

    def test_fastapi_route_profile_filter_rejects_unknown_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.routes.matched_swings import router, get_matched_swings_service

        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", list)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_matched_swings_service] = lambda: service
        client = TestClient(app)

        res = client.get("/matched-swings?profile=bogus")
        assert res.status_code == 422

    def test_candidate_preview_frame_includes_observed_dots_and_residuals(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Create a candidate package with both model_markers_m and target_markers_m
        npz_path = tmp_path / "candidate.npz"
        times = np.array([0.0, 0.1])
        q = np.zeros((2, 4))
        model_markers = np.zeros((2, 3, 3)) + 0.010
        target_markers = np.zeros((2, 3, 3))
        valid_mask = np.ones((2, 3), dtype=bool)

        np.savez(
            npz_path,
            manifest_json=json.dumps({"schema_version": "candidate/1"}),
            time_s=times,
            q=q,
            model_markers_m=model_markers,
            target_markers_m=target_markers,
            marker_validity=valid_mask,
        )

        row = _make_row("r1.json", "drake", 0.010, sha256="a" * 64)
        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", lambda: [row])
        monkeypatch.setattr(
            service, "resolve_artifact_path", lambda run_id, art: npz_path
        )

        preview = service.candidate_preview_frame(row.sha256, frame_index=0)
        assert "joints" in preview
        assert "target_markers" in preview
        assert "residual_vectors" in preview
        assert preview["rms_error_m"] == pytest.approx(
            float(np.sqrt(np.mean(0.010**2 * 3)))
        )
        assert len(preview["target_markers"]) == 3

    def test_candidate_preview_residual_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        npz_path = tmp_path / "candidate.npz"
        times = np.array([0.0, 0.1, 0.2])
        q = np.zeros((3, 4))
        model_markers = np.zeros((3, 3, 3)) + 0.008
        target_markers = np.zeros((3, 3, 3))
        valid_mask = np.ones((3, 3), dtype=bool)

        np.savez(
            npz_path,
            manifest_json=json.dumps({"schema_version": "candidate/1"}),
            time_s=times,
            q=q,
            model_markers_m=model_markers,
            target_markers_m=target_markers,
            marker_validity=valid_mask,
        )

        row = _make_row("r1.json", "pinocchio", 0.008, sha256="b" * 64)
        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", lambda: [row])
        monkeypatch.setattr(
            service, "resolve_artifact_path", lambda run_id, art: npz_path
        )

        res = service.candidate_preview_residual_summary(row.sha256)
        assert res["id"] == row.sha256
        assert res["mean_rms_m"] > 0.0
        assert "worst_frame_idx" in res
        assert "worst_phase" in res

    def test_fastapi_route_ranked_and_residuals(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.routes.matched_swings import router, get_matched_swings_service

        npz_path = tmp_path / "candidate.npz"
        times = np.array([0.0, 0.1])
        q = np.zeros((2, 4))
        model_markers = np.zeros((2, 2, 3)) + 0.012
        target_markers = np.zeros((2, 2, 3))
        valid_mask = np.ones((2, 2), dtype=bool)

        np.savez(
            npz_path,
            manifest_json=json.dumps({"schema_version": "candidate/1"}),
            time_s=times,
            q=q,
            model_markers_m=model_markers,
            target_markers_m=target_markers,
            marker_validity=valid_mask,
        )

        rows = [
            _make_row("r1.json", "mujoco", 0.024, capture="driver", sha256="1" * 64),
            _make_row("r2.json", "pinocchio", 0.011, capture="driver", sha256="2" * 64),
        ]
        service = MatchedSwingsService.from_ledger_file(
            tmp_path / "ledger.json", tmp_path
        )
        monkeypatch.setattr(service, "_load_rows", lambda: list(rows))
        monkeypatch.setattr(
            service, "resolve_artifact_path", lambda run_id, art: npz_path
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_matched_swings_service] = lambda: service

        client = TestClient(app)

        # 1. Test GET /matched-swings with ranked=true
        res = client.get("/matched-swings?ranked=true&capture=driver")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 2
        assert data["runs"][0]["engine"] == "pinocchio"
        assert data["runs"][1]["engine"] == "mujoco"

        # 2. Test GET /matched-swings/{id}/residuals
        res_resid = client.get(f"/matched-swings/{rows[0].sha256}/residuals")
        assert res_resid.status_code == 200
        resid_data = res_resid.json()
        assert "mean_rms_m" in resid_data

        # 3. Test GET /matched-swings/{id}/candidate with preview_frame
        res_cand = client.get(
            f"/matched-swings/{rows[0].sha256}/candidate?preview_frame=0"
        )
        assert res_cand.status_code == 200
        cand_data = res_cand.json()
        assert "joints" in cand_data
        assert "target_markers" in cand_data
        assert "residual_vectors" in cand_data
