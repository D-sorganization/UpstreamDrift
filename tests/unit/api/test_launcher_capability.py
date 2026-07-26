"""TDD coverage for the local launcher/API capability contract.

UpstreamDrift issue #8102: the classic launcher and its API child must share
an ephemeral proof without logging or persisting it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.auth.launcher_capability import (
    INSTANCE_ID_ENV,
    LAUNCHER_TOKEN_ENV,
    LauncherCapability,
    install_launcher_capability,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_generated_capability_is_nonempty_and_redacted() -> None:
    """Generated capability material must never appear in repr diagnostics."""
    capability = LauncherCapability.generate()

    assert len(capability.token) >= 32
    assert len(capability.instance_id) >= 16
    assert capability.token not in repr(capability)
    assert capability.instance_id in repr(capability)


def test_capability_exports_one_ephemeral_contract() -> None:
    """The launcher exports exactly the values consumed by its API child."""
    environ: dict[str, str] = {}
    capability = LauncherCapability(token="secret-token", instance_id="instance-1")

    capability.export(environ)

    assert environ == {
        LAUNCHER_TOKEN_ENV: "secret-token",
        INSTANCE_ID_ENV: "instance-1",
    }


def test_capability_rejects_blank_contract_values() -> None:
    """DbC: blank proof or identity values are programmer errors."""
    with pytest.raises(ValueError, match="token"):
        LauncherCapability(token="", instance_id="instance-1")
    with pytest.raises(ValueError, match="instance_id"):
        LauncherCapability(token="secret-token", instance_id="")


def test_install_launcher_capability_copies_environment_to_app_state() -> None:
    """The API app receives the same capability exported by the launcher."""
    app = SimpleNamespace(state=SimpleNamespace())
    environ = {
        LAUNCHER_TOKEN_ENV: "secret-token",
        INSTANCE_ID_ENV: "instance-1",
    }

    install_launcher_capability(app, environ)

    assert app.state.launcher_csrf_token == "secret-token"
    assert app.state.sidekick_instance_id == "instance-1"


def test_install_launcher_capability_fails_closed_when_missing() -> None:
    """A standalone server without launcher proof must not invent shared proof."""
    app = SimpleNamespace(state=SimpleNamespace())

    install_launcher_capability(app, {})

    assert app.state.launcher_csrf_token == ""
    assert app.state.sidekick_instance_id == ""
