"""API tests for the web Spine Viewer routes (SPN-15c, #12815)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.task_manager import TaskManager
from tests.tools.spine_viewer.conftest import SOURCE, write_study

pytestmark = pytest.mark.unit

ROOT_ENV = "UPSTREAMDRIFT_SPINE_STUDY_ROOT"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "true")
    studies = tmp_path / "studies"
    studies.mkdir()
    monkeypatch.setenv(ROOT_ENV, str(studies))
    return studies


@pytest.fixture
def client(root: Path) -> TestClient:
    if getattr(app.state, "task_manager", None) is None:
        app.state.task_manager = TaskManager()
    return TestClient(app)


def test_study_returns_verified_view_model(client: TestClient, root: Path) -> None:
    write_study(root / "a")
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "a/s1_receipt.json"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["scientific_status"] == "unqualified"
    assert body["distribution_source"] == SOURCE
    assert "UNQUALIFIED" in body["banner"]
    assert body["heatmaps"]["compression"]["values"][2] == [300.0, 20.0]


def test_qualified_study_is_reported_qualified(client: TestClient, root: Path) -> None:
    write_study(root / "q", status="pass", scientific_status="qualified")
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "q/s1_receipt.json"}
    )
    assert resp.json()["scientific_status"] == "qualified"


def test_tampered_artifact_is_rejected(client: TestClient, root: Path) -> None:
    write_study(root / "t")
    npz = root / "t" / "s1_loads.npz"
    npz.write_bytes(npz.read_bytes() + b"x")
    assert hashlib.sha256(npz.read_bytes()).hexdigest()
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "t/s1_receipt.json"}
    )
    assert resp.status_code == 422
    assert "sha256" in resp.json()["detail"]


@pytest.mark.parametrize(
    "bad", ["../outside.json", "/etc/passwd", "a/../../x.json", "", "a\\..\\b.json"]
)
def test_path_traversal_is_rejected(client: TestClient, root: Path, bad: str) -> None:
    resp = client.get("/api/tools/spine-viewer/study", params={"receipt": bad})
    assert resp.status_code in (400, 404, 422)


def test_symlink_escape_is_rejected(
    client: TestClient, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    write_study(outside)
    (root / "link").symlink_to(outside, target_is_directory=True)
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "link/s1_receipt.json"}
    )
    assert resp.status_code == 400


def test_missing_receipt_is_404(client: TestClient, root: Path) -> None:
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "nope/s1_receipt.json"}
    )
    assert resp.status_code == 404


def test_receipts_listing_is_relative_and_sorted(
    client: TestClient, root: Path
) -> None:
    write_study(root / "b")
    write_study(root / "a")
    resp = client.get("/api/tools/spine-viewer/receipts")
    assert resp.status_code == 200
    assert resp.json()["receipts"] == ["a/s1_receipt.json", "b/s1_receipt.json"]


def test_assets_listing_includes_manifest_entries_and_fixture(
    client: TestClient,
) -> None:
    resp = client.get("/api/tools/spine-viewer/assets")
    assert resp.status_code == 200
    body = resp.json()
    assert "assets" in body
    assets = {a["asset_id"]: a for a in body["assets"]}

    # Fixture is always available and qualified
    assert "fixture" in assets
    assert assets["fixture"]["status"] == "qualified"
    assert assets["fixture"]["is_fixture"] is True

    # Manifest assets
    assert "bruno2015_fullbody_os4_v2" in assets
    assert assets["bruno2015_fullbody_os4_v2"]["status"] in ("missing", "qualified")
    assert assets["bruno2015_fullbody_os4_v2"]["is_fixture"] is False

    assert "senteler2016_lumbspine_3dof" in assets
    assert assets["senteler2016_lumbspine_3dof"]["status"] == "unpinned"

    assert "senteler2016_lumbspine_30dof" in assets
    assert assets["senteler2016_lumbspine_30dof"]["status"] == "unpinned"

    # Verify no licensed file contents or forbidden keys leaked
    for item in body["assets"]:
        assert "mesh" not in item
        assert "osim" not in item
        assert "coordinates" not in item


def test_post_study_bad_request_returns_422(client: TestClient, tmp_path: Path) -> None:
    # Empty body
    resp = client.post("/api/tools/spine-viewer/studies", json={})
    assert resp.status_code == 422

    # Invalid study_id
    resp = client.post(
        "/api/tools/spine-viewer/studies",
        json={"study_id": "../escape", "model_id": "fixture"},
    )
    assert resp.status_code == 422

    # Path traversal in output_dir
    resp = client.post(
        "/api/tools/spine-viewer/studies",
        json={
            "study_id": "s1",
            "model_id": "fixture",
            "output_dir": "../../escaped",
        },
    )
    assert resp.status_code == 422

    # Missing input files
    resp = client.post(
        "/api/tools/spine-viewer/studies",
        json={
            "study_id": "s1",
            "model_id": "fixture",
            "motion_path": str(tmp_path / "absent.sto"),
            "golf_model_path": str(tmp_path / "absent.osim"),
            "model_path": str(tmp_path / "absent_model.osim"),
        },
    )
    assert resp.status_code == 422


def test_study_job_not_found_returns_404(client: TestClient) -> None:
    resp = client.get(
        "/api/tools/spine-viewer/studies/00000000-0000-0000-0000-000000000000"
    )
    assert resp.status_code == 404


def test_study_root_created_on_first_write(
    client: TestClient, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure study_root does not exist initially
    fresh_root = tmp_path / "fresh_studies"
    monkeypatch.setenv(ROOT_ENV, str(fresh_root))
    assert not fresh_root.exists()

    # Create dummy input files
    motion = tmp_path / "m.sto"
    motion.write_text("motion")
    golf_model = tmp_path / "g.osim"
    golf_model.write_text("model")
    model = tmp_path / "x.osim"
    model.write_text("model")

    from src.api.routes import spine_viewer

    def fake_run(req: object) -> None:
        pass

    monkeypatch.setattr(spine_viewer, "run_spine_study", fake_run)

    resp = client.post(
        "/api/tools/spine-viewer/studies",
        json={
            "study_id": "first_write",
            "model_id": "fixture",
            "motion_path": str(motion),
            "golf_model_path": str(golf_model),
            "model_path": str(model),
        },
    )
    assert resp.status_code == 202
    assert fresh_root.exists()


def test_post_study_job_lifecycle_with_patched_runner(
    client: TestClient, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    from src.api.routes import spine_viewer

    motion = tmp_path / "m.sto"
    motion.write_text("motion")
    golf_model = tmp_path / "g.osim"
    golf_model.write_text("model")
    model = tmp_path / "x.osim"
    model.write_text("model")

    def fake_run(req: Any) -> None:
        out_dir = req.output_dir
        write_study(out_dir)

    monkeypatch.setattr(spine_viewer, "run_spine_study", fake_run)

    resp = client.post(
        "/api/tools/spine-viewer/studies",
        json={
            "study_id": "job_test",
            "model_id": "fixture",
            "motion_path": str(motion),
            "golf_model_path": str(golf_model),
            "model_path": str(model),
        },
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert job_id

    # Poll status until completed
    deadline = time.monotonic() + 10.0
    status_body = {}
    while time.monotonic() < deadline:
        st_resp = client.get(f"/api/tools/spine-viewer/studies/{job_id}")
        assert st_resp.status_code == 200
        status_body = st_resp.json()
        if status_body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert status_body["status"] == "completed"
    receipt_path = status_body["receipt_path"]
    assert receipt_path is not None

    # Load receipt via existing /study route
    view_resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": receipt_path}
    )
    assert view_resp.status_code == 200
    assert view_resp.json()["scientific_status"] == "unqualified"


def test_study_job_failure_reported(
    client: TestClient, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    from src.api.routes import spine_viewer

    motion = tmp_path / "m.sto"
    motion.write_text("motion")
    golf_model = tmp_path / "g.osim"
    golf_model.write_text("model")
    model = tmp_path / "x.osim"
    model.write_text("model")

    def failing_run(req: object) -> None:
        raise ValueError("simulated solver divergence")

    monkeypatch.setattr(spine_viewer, "run_spine_study", failing_run)

    resp = client.post(
        "/api/tools/spine-viewer/studies",
        json={
            "study_id": "fail_test",
            "model_id": "fixture",
            "motion_path": str(motion),
            "golf_model_path": str(golf_model),
            "model_path": str(model),
        },
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + 10.0
    status_body = {}
    while time.monotonic() < deadline:
        st_resp = client.get(f"/api/tools/spine-viewer/studies/{job_id}")
        assert st_resp.status_code == 200
        status_body = st_resp.json()
        if status_body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert status_body["status"] == "failed"
    assert "ValueError: simulated solver divergence" in status_body["error"]


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_opensim_fixture_study_end_to_end(
    client: TestClient, root: Path, tmp_path: Path
) -> None:
    import time
    import numpy as np
    from src.engines.physics_engines.opensim.python.spine.fixture_model import (
        FixtureSpec,
        write_fixture_osim,
    )
    from src.shared.python.spine.coupling import LinearCouplingLaw
    from src.shared.python.spine.levels import SpinalLevel, levels_between
    from tests.helpers.opensim_support import require_opensim

    osim = require_opensim()
    lumbar = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
    phi = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")
    mats = {lv: np.eye(3) * 0.1 for lv in lumbar}
    law = LinearCouplingLaw("invented", "1", mats, phi, "invented test law")
    model_path = write_fixture_osim(
        FixtureSpec(), tmp_path / "fixture.osim", coupling=law, muscles=True
    )
    # Retune tendon slack length
    m = osim.Model(str(model_path))
    s = m.initSystem()
    muscles = m.getMuscles()
    for i in range(muscles.getSize()):
        mus = muscles.get(i)
        mus.set_tendon_slack_length(mus.getLength(s) - mus.get_optimal_fiber_length())
    m.printToXML(str(model_path))

    # Write 5-sample synthetic motion
    motion_path = tmp_path / "swing.sto"
    t = [0.0, 0.1, 0.2, 0.3, 0.4]
    cols = {
        "lumbar_extension": [0.05, 0.1, 0.15, 0.1, 0.05],
        "lumbar_bending": [0.1, 0.07, 0.04, 0.01, -0.02],
        "lumbar_rotation": [0.1, 0.18, 0.26, 0.34, 0.42],
    }
    lines = [
        "synthetic",
        "version=1",
        f"nRows={len(t)}",
        f"nColumns={len(cols) + 1}",
        "inDegrees=no",
        "endheader",
        "time\t" + "\t".join(cols),
    ]
    for i, ti in enumerate(t):
        lines.append("\t".join([f"{ti:.6f}"] + [f"{cols[k][i]:.6g}" for k in cols]))
    motion_path.write_text("\n".join(lines) + "\n")

    repo_root = Path(__file__).resolve().parents[2]
    golf_model_path = (
        repo_root / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
    )

    payload = {
        "study_id": "e2e_fixture",
        "model_id": "fixture",
        "pelvis": "fixture",
        "motion_path": str(motion_path),
        "golf_model_path": str(golf_model_path),
        "model_path": str(model_path),
        "events": {"mid": 0.2, "late": 0.3},
        "max_frames": 5,
        "residual_force_fraction_bw": 1e3,
        "residual_moment_fraction": 1e3,
        "reserve_limit_nm": 1e6,
        "reserve_max_share": 1e3,
        "torso_alignment": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "torso_alignment_source": "test fixture: identity by construction",
    }

    resp = client.post("/api/tools/spine-viewer/studies", json=payload)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + 30.0
    status_body = {}
    while time.monotonic() < deadline:
        st_resp = client.get(f"/api/tools/spine-viewer/studies/{job_id}")
        assert st_resp.status_code == 200
        status_body = st_resp.json()
        if status_body["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert status_body["status"] == "completed", status_body.get("error")
    receipt_path = status_body["receipt_path"]
    assert receipt_path is not None

    view_resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": receipt_path}
    )
    assert view_resp.status_code == 200, view_resp.text
    assert view_resp.json()["study_id"] == "e2e_fixture"
