"""Policy tests for database migration release and CI coverage."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
RELEASE_RUNBOOK = REPO_ROOT / "docs" / "operations" / "release-runbook.md"


def test_ci_standard_runs_alembic_drift_check() -> None:
    """CI Standard must reject model changes without paired migrations."""
    workflow = CI_STANDARD.read_text(encoding="utf-8")

    assert "name: Alembic Drift Check" in workflow
    assert "python3 scripts/check_alembic_drift.py" in workflow


def test_ci_standard_runs_round_trip_against_postgresql() -> None:
    """CI Standard must exercise migration downgrade/upgrade on PostgreSQL."""
    workflow = CI_STANDARD.read_text(encoding="utf-8")

    assert "alembic-postgres-round-trip:" in workflow
    assert "postgres:" in workflow
    assert "ALEMBIC_ROUND_TRIP_DATABASE_URL" in workflow
    assert "tests/integration/test_alembic_round_trip.py" in workflow


def test_release_runbook_requires_alembic_upgrade_before_server_rollout() -> None:
    """Release operators must run schema migrations before starting servers."""
    runbook = RELEASE_RUNBOOK.read_text(encoding="utf-8")

    assert "python3 scripts/db_migrate.py upgrade head" in runbook
    assert "before starting the new server version" in runbook
