"""TDD checks for root container policy documentation (issue #6097)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]


class TestContainerStrategyAdr:
    """ADR-0021 must record the root container policy."""

    def _read(self) -> str:
        path = _REPO_ROOT / "docs" / "adr" / "0021-container-strategy.md"
        assert path.exists(), "ADR-0021 must exist for issue #6097"
        return path.read_text(encoding="utf-8")

    def test_exists(self) -> None:
        assert (_REPO_ROOT / "docs" / "adr" / "0021-container-strategy.md").exists()

    def test_is_accepted(self) -> None:
        assert "Status: Accepted" in self._read()

    def test_cross_links_issue(self) -> None:
        assert "#6097" in self._read()

    def test_names_all_root_dockerfiles(self) -> None:
        content = self._read()
        for name in ("Dockerfile", "Dockerfile.heavy_test", "Dockerfile.modular"):
            assert name in content


class TestDockerReadme:
    """docker/README.md must explain the three root Dockerfile roles."""

    def _read(self) -> str:
        path = _REPO_ROOT / "docker" / "README.md"
        assert path.exists(), "docker/README.md must exist for issue #6097"
        return path.read_text(encoding="utf-8")

    def test_exists(self) -> None:
        assert (_REPO_ROOT / "docker" / "README.md").exists()

    def test_names_all_root_dockerfiles(self) -> None:
        content = self._read()
        for name in ("Dockerfile", "Dockerfile.heavy_test", "Dockerfile.modular"):
            assert name in content

    def test_describes_canonical_release_path(self) -> None:
        content = self._read().lower()
        assert "canonical" in content and "release" in content

    def test_describes_heavy_test_parity_path(self) -> None:
        content = self._read().lower()
        assert "heavy_test" in content and "parity" in content

    def test_describes_modular_profile_path(self) -> None:
        content = self._read().lower()
        assert "modular" in content and "profile" in content


class TestDockerfileHeaders:
    """Each root Dockerfile must point readers at the policy docs."""

    @pytest.mark.parametrize(
        "relative_path",
        ("Dockerfile", "Dockerfile.heavy_test", "Dockerfile.modular"),
    )
    def test_header_references_policy_docs(self, relative_path: str) -> None:
        content = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "docker/README.md" in content
        assert "0021-container-strategy" in content
