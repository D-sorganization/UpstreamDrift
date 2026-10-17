"""Issue #12901: the launcher-owned API child must not run under auto-reload.

``uvicorn.run(..., reload=True)`` imports the whole application in the parent,
again in the reloader and a third time in the worker, which made the Sidekick
API take ~90 s to become ready. The launcher contract now pins ``API_RELOAD=0``.
"""

from __future__ import annotations

import pytest

from src.api.auth.launcher_capability import LauncherCapability
from src.api.config import API_RELOAD_ENV, api_reload_enabled
from src.launchers.sidekick_runtime import configure_sidekick_runtime


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
def test_api_reload_explicit_off(raw: str) -> None:
    assert (
        api_reload_enabled({API_RELOAD_ENV: raw, "ENVIRONMENT": "development"}) is False
    )


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_api_reload_explicit_on_even_in_production(raw: str) -> None:
    assert (
        api_reload_enabled({API_RELOAD_ENV: raw, "ENVIRONMENT": "production"}) is True
    )


@pytest.mark.unit
def test_api_reload_defaults_to_development_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.config.environment import get_environment

    try:
        monkeypatch.setenv("ENVIRONMENT", "development")
        get_environment.cache_clear()
        assert api_reload_enabled({}) is True
        monkeypatch.setenv("ENVIRONMENT", "production")
        get_environment.cache_clear()
        assert api_reload_enabled({}) is False
    finally:
        monkeypatch.undo()
        get_environment.cache_clear()


@pytest.mark.unit
def test_api_reload_rejects_garbage_and_none() -> None:
    with pytest.raises(ValueError):
        api_reload_enabled({API_RELOAD_ENV: "maybe"})
    with pytest.raises(ValueError):
        api_reload_enabled(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_sidekick_runtime_contract_disables_reload() -> None:
    environ: dict[str, str] = {}
    configure_sidekick_runtime(
        environ,
        port_selector=lambda preferred: 8123,
        capability_factory=LauncherCapability.generate,
    )
    assert environ[API_RELOAD_ENV] == "0"
    assert api_reload_enabled(environ) is False
