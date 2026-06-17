"""Static assertions for the Docker hardening in issues #7159 and #7161.

These guard the build-chain contract so a future edit cannot silently
re-introduce the swallowed-failure / missing-audit defects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ── #7159 D2 — image build audits the resolved environment ─────────────────


def test_dockerfile_runs_in_image_pip_audit() -> None:
    dockerfile = _read("Dockerfile")
    # pip-audit runs inside the image build, reusing the shared waiver policy.
    assert "pip_audit" in dockerfile or "pip-audit" in dockerfile
    assert "check_pip_audit_waivers.py" in dockerfile
    assert "pip_audit_waivers.json" in dockerfile
    # Escape hatch for air-gapped builds, defaulting to enforced.
    assert "ARG SKIP_AUDIT=false" in dockerfile


def test_dockerfile_audit_reuses_shared_waiver_file() -> None:
    """DRY: the image must not ship its own waiver copy — it copies the canonical
    scripts/config/pip_audit_waivers.json."""
    dockerfile = _read("Dockerfile")
    assert "scripts/config/pip_audit_waivers.json" in dockerfile


# ── #7161 D2 — heavy-test optional installs are attributable ────────────────


def test_heavy_test_records_optional_dep_status() -> None:
    heavy = _read("Dockerfile.heavy_test")
    # Optional installs write a marker file rather than only `|| echo`.
    assert ".optional_deps_status" in heavy
    # And the entrypoint surfaces that marker.
    assert "heavy_test_entrypoint.sh" in heavy


def test_heavy_test_entrypoint_surfaces_marker() -> None:
    entry = _read("docker/heavy_test_entrypoint.sh")
    assert ".optional_deps_status" in entry
    assert "missing" in entry


@pytest.mark.parametrize(
    "relpath",
    [
        "Dockerfile.heavy_test",
        "src/engines/physics_engines/drake/Dockerfile",
        "src/engines/physics_engines/mujoco/docker/Dockerfile",
        "src/engines/physics_engines/pinocchio/Dockerfile",
    ],
)
def test_trivy_high_policy_dockerfiles_install_minimally_and_drop_root(
    relpath: str,
) -> None:
    """Dockerfiles scanned by full-main Trivy must satisfy high-risk policy."""
    dockerfile = _read(relpath)
    if "apt-get install -y" in dockerfile:
        assert "apt-get install -y --no-install-recommends" in dockerfile
    assert "\nUSER " in dockerfile


# ── #7161 D3 — compose health gating + python healthcheck ──────────────────


def test_compose_frontend_waits_for_backend_health() -> None:
    compose = _read("docker-compose.yml")
    assert "condition: service_healthy" in compose


def test_healthchecks_do_not_depend_on_curl() -> None:
    dockerfile = _read("Dockerfile")
    compose = _read("docker-compose.yml")
    # The image healthcheck uses a python one-liner, not curl.
    assert "HEALTHCHECK" in dockerfile
    assert "urllib.request" in dockerfile
    # Compose healthcheck likewise avoids curl.
    assert "urllib.request" in compose
    assert '"curl"' not in compose


def test_dockerfile_sets_mujoco_gl_default() -> None:
    dockerfile = _read("Dockerfile")
    assert 'MUJOCO_GL="osmesa"' in dockerfile or "MUJOCO_GL=osmesa" in dockerfile


# ── #7161 D3 — modular profile validated before the expensive layer ────────


def test_modular_validates_profile_before_lockfile_install() -> None:
    modular = _read("Dockerfile.modular")
    validate_idx = modular.find("--dry-run")
    lockfile_idx = modular.find("pip install -r /tmp/requirements.lock")
    assert validate_idx != -1, "expected a --dry-run profile validation step"
    assert lockfile_idx != -1
    # Validation must come before the expensive lockfile install.
    assert validate_idx < lockfile_idx
