"""Regression tests for the production Docker entrypoint (issue #7129).

Docker exec-form ``CMD``/``ENTRYPOINT`` arrays do not perform shell parameter
expansion, so passing ``"${FORWARDED_ALLOW_IPS:-127.0.0.1}"`` as an argument
handed uvicorn the literal string instead of the environment value. These tests
pin the fix: a ``/bin/sh`` entrypoint wrapper expands the variable (with the
documented localhost default) before exec'ing uvicorn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "Dockerfile"
_ENTRYPOINT = _REPO_ROOT / "docker" / "entrypoint.sh"


def _dockerfile_text() -> str:
    assert _DOCKERFILE.exists(), "Root Dockerfile must exist"
    return _DOCKERFILE.read_text(encoding="utf-8")


def _entrypoint_text() -> str:
    assert _ENTRYPOINT.exists(), "docker/entrypoint.sh must exist for issue #7129"
    return _ENTRYPOINT.read_text(encoding="utf-8")


class TestDockerfileDoesNotMisuseExpansion:
    def test_no_literal_expansion_in_exec_array(self) -> None:
        """The unexpanded ``${FORWARDED_ALLOW_IPS...}`` token must not appear in
        an exec-form CMD/ENTRYPOINT array, where Docker would pass it verbatim.
        """
        text = _dockerfile_text()
        assert "${FORWARDED_ALLOW_IPS:-127.0.0.1}" not in text

    def test_entrypoint_is_wired_in(self) -> None:
        text = _dockerfile_text()
        assert "/usr/local/bin/entrypoint.sh" in text
        assert 'ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]' in text


class TestEntrypointExpandsForwardedAllowIps:
    def test_runs_under_posix_shell(self) -> None:
        text = _entrypoint_text()
        assert text.startswith("#!/bin/sh")

    def test_expands_with_localhost_default(self) -> None:
        text = _entrypoint_text()
        # POSIX default-expansion of the documented localhost default.
        assert 'FORWARDED_ALLOW_IPS="${FORWARDED_ALLOW_IPS:-127.0.0.1}"' in text

    def test_passes_expanded_value_to_uvicorn(self) -> None:
        text = _entrypoint_text()
        assert "--forwarded-allow-ips" in text
        assert '"${FORWARDED_ALLOW_IPS}"' in text

    def test_execs_uvicorn_for_signal_handling(self) -> None:
        text = _entrypoint_text()
        # exec replaces the shell so uvicorn receives SIGTERM as PID 1.
        assert "exec python3 -m uvicorn src.api.server:app" in text
