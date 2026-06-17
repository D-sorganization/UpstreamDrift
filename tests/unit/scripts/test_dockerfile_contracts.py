from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import check_dockerfile_contracts as guard

pytestmark = pytest.mark.unit


def _write_minimal_tree(root: Path, *, modular_pip: str = "26.1.2") -> None:
    (root / "Dockerfile").write_text(
        """
RUN pip install --upgrade pip==26.1.2
ARG SKIP_AUDIT=false
RUN pip install pip-audit==2.10.0
COPY scripts/ci/check_pip_audit_waivers.py /tmp/check_pip_audit_waivers.py
COPY scripts/config/pip_audit_waivers.json /tmp/pip_audit_waivers.json
ENV PATH="/opt/venv/bin:$PATH" \\
    PYTHONPATH="/workspace" \\
    MUJOCO_GL=osmesa
HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=5)"
""",
        encoding="utf-8",
    )
    (root / "Dockerfile.modular").write_text(
        f"""
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN python scripts/docker/install_features.py --profile standard --dry-run
COPY launch_golf_suite.py ./
RUN python scripts/docker/install_features.py --profile "$PROFILE"
RUN pip install --upgrade pip=={modular_pip}
""",
        encoding="utf-8",
    )
    (root / "Dockerfile.heavy_test").write_text(
        """
RUN pip install PyQt6
RUN pip install drake
RUN pip install opensim
RUN pip install myosuite
""",
        encoding="utf-8",
    )
    (root / "docker-compose.yml").write_text(
        """
services:
  frontend:
    depends_on:
      backend:
        condition: service_healthy
""",
        encoding="utf-8",
    )


def test_docker_contracts_pass_for_current_shape(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)

    assert guard.docker_contract_failures(tmp_path) == []


def test_docker_contracts_reject_pip_pin_skew(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, modular_pip="25.3")

    assert any(
        "pip pins diverge" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def test_docker_contracts_reject_masked_heavy_install(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.heavy_test").write_text(
        "RUN pip install drake || echo skip\n",
        encoding="utf-8",
    )

    assert any(
        "masks drake install failure" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def test_docker_contracts_reject_modular_profile_dry_run_before_registry_copy(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.modular").write_text(
        """
RUN python scripts/docker/install_features.py --profile standard --dry-run
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN pip install --upgrade pip==26.1.2
""",
        encoding="utf-8",
    )

    assert any(
        "copy feature_registry before profile dry-run" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def test_docker_contracts_reject_modular_profile_dry_run_before_engine_core_copy(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.modular").write_text(
        """
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
RUN python scripts/docker/install_features.py --profile standard --dry-run
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN pip install --upgrade pip==26.1.2
""",
        encoding="utf-8",
    )

    assert any(
        "copy engine_core before profile dry-run" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def test_docker_contracts_reject_modular_feature_install_before_launcher_copy(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.modular").write_text(
        """
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN python scripts/docker/install_features.py --profile standard --dry-run
RUN python scripts/docker/install_features.py --profile "$PROFILE"
COPY launch_golf_suite.py ./
RUN pip install --upgrade pip==26.1.2
""",
        encoding="utf-8",
    )

    assert any(
        "copy launch_golf_suite.py before feature install" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )
