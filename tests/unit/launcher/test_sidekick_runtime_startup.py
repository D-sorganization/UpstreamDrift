"""TDD coverage for isolated, resilient Sidekick runtime startup.

UpstreamDrift issue #8102: the PyQt6 launcher must not attach Sidekick to an
unrelated process that happens to own the historical port 8000.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.api.auth.launcher_capability import (
    INSTANCE_ID_ENV,
    LAUNCHER_TOKEN_ENV,
    LauncherCapability,
)
from src.launchers import sidekick_runtime as runtime_module
from src.launchers.sidekick_runtime import (
    API_PORT_ENV,
    CANONICAL_API_PORT_ENV,
    DEFAULT_API_PORT,
    SidekickRuntimeConfig,
    configure_sidekick_runtime,
    select_loopback_port,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _test_capability() -> LauncherCapability:
    return LauncherCapability(token="secret-token", instance_id="instance-1")


def test_select_loopback_port_uses_default_when_available() -> None:
    """The historical port remains stable when it is genuinely free."""
    assert select_loopback_port(
        DEFAULT_API_PORT, is_port_available=lambda _port: True
    ) == (DEFAULT_API_PORT)


def test_select_loopback_port_avoids_occupied_default() -> None:
    """An occupied port 8000 must produce a distinct free loopback port."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    occupied_port = listener.getsockname()[1]
    try:
        selected = select_loopback_port(occupied_port)
    finally:
        listener.close()

    assert selected != occupied_port
    with socket.socket() as verifier:
        verifier.bind(("127.0.0.1", selected))


def test_configure_sidekick_runtime_exports_consistent_port_and_capability() -> None:
    """The API server and Tools chat client must inherit one runtime contract."""
    environ: dict[str, str] = {}

    config = configure_sidekick_runtime(
        environ,
        port_selector=lambda _preferred: 8123,
        capability_factory=_test_capability,
    )

    assert config.port == 8123
    assert config.instance_id == "instance-1"
    assert "secret-token" not in repr(config)
    assert environ[API_PORT_ENV] == "8123"
    assert environ[CANONICAL_API_PORT_ENV] == "8123"
    assert environ[LAUNCHER_TOKEN_ENV] == "secret-token"
    assert environ[INSTANCE_ID_ENV] == "instance-1"


def test_configure_sidekick_runtime_respects_one_explicit_port() -> None:
    """An explicit canonical port is copied to the legacy server setting."""
    environ = {CANONICAL_API_PORT_ENV: "9123"}

    config = configure_sidekick_runtime(
        environ,
        port_selector=lambda _preferred: pytest.fail("selector should not run"),
        capability_factory=_test_capability,
    )

    assert config.port == 9123
    assert environ[API_PORT_ENV] == "9123"


def test_configure_sidekick_runtime_rejects_conflicting_explicit_ports() -> None:
    """DbC: ambiguous API/client port configuration must fail visibly."""
    environ = {
        API_PORT_ENV: "8123",
        CANONICAL_API_PORT_ENV: "9123",
    }

    with pytest.raises(ValueError, match="conflict"):
        configure_sidekick_runtime(environ)


@pytest.mark.parametrize("value", ["0", "65536", "not-a-port"])
def test_configure_sidekick_runtime_rejects_invalid_explicit_port(
    value: str,
) -> None:
    """DbC: invalid explicit ports are not silently replaced."""
    with pytest.raises(ValueError, match="port"):
        configure_sidekick_runtime({API_PORT_ENV: value})


def test_each_dynamic_restart_reselects_port_and_preserves_capability() -> None:
    """A bind race is retried with one identity and a freshly checked port."""
    capability = _test_capability()
    config = SidekickRuntimeConfig(port=8123, capability=capability)
    environ: dict[str, str] = {}
    preferred_ports: list[int] = []
    selected_ports = iter((8124, 8125))

    def select_port(preferred_port: int) -> int:
        preferred_ports.append(preferred_port)
        return next(selected_ports)

    first_restart = runtime_module.reselect_sidekick_runtime_port(
        config,
        environ,
        port_selector=select_port,
    )
    second_restart = runtime_module.reselect_sidekick_runtime_port(
        first_restart,
        environ,
        port_selector=select_port,
    )

    assert preferred_ports == [8123, 8124]
    assert second_restart.port == 8125
    assert first_restart.capability is capability
    assert second_restart.capability is capability
    assert second_restart.instance_id == "instance-1"
    assert environ[API_PORT_ENV] == "8125"
    assert environ[CANONICAL_API_PORT_ENV] == "8125"
    assert environ[LAUNCHER_TOKEN_ENV] == "secret-token"
    assert environ[INSTANCE_ID_ENV] == "instance-1"


def test_restart_preserves_explicit_port_without_dynamic_reselection() -> None:
    """A user-selected port is an explicit contract, not a fallback candidate."""
    environ = {CANONICAL_API_PORT_ENV: "9123"}
    config = configure_sidekick_runtime(
        environ,
        port_selector=lambda _preferred: pytest.fail("selector should not run"),
        capability_factory=_test_capability,
    )

    restarted = runtime_module.reselect_sidekick_runtime_port(
        config,
        environ,
        port_selector=lambda _preferred: pytest.fail("selector should not run"),
    )

    assert restarted is config
    assert environ[API_PORT_ENV] == "9123"
    assert environ[CANONICAL_API_PORT_ENV] == "9123"


def test_runtime_port_reselection_validates_contract_inputs() -> None:
    """DbC: malformed runtime contracts fail before mutating the environment."""
    capability = _test_capability()
    with pytest.raises(TypeError, match="port_is_explicit"):
        SidekickRuntimeConfig(
            port=8123,
            capability=capability,
            port_is_explicit=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="SidekickRuntimeConfig"):
        runtime_module.reselect_sidekick_runtime_port(  # type: ignore[arg-type]
            object(),
            {},
        )
    with pytest.raises(ValueError, match="environ"):
        runtime_module.reselect_sidekick_runtime_port(
            SidekickRuntimeConfig(port=8123, capability=capability),
            None,  # type: ignore[arg-type]
        )


def test_launcher_reselects_runtime_contract_before_relaunch() -> None:
    """The replacement child inherits the refreshed port and original identity."""
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

    original = SidekickRuntimeConfig(port=8123, capability=_test_capability())
    refreshed = SidekickRuntimeConfig(port=8124, capability=original.capability)
    replacement_process = object()
    launcher = SimpleNamespace(
        _sidekick_runtime_config=original,
        _launch_sidekick_background_api=MagicMock(return_value=replacement_process),
    )

    with patch(
        "src.launchers.upstream_drift_launcher.reselect_sidekick_runtime_port",
        return_value=refreshed,
    ) as reselect:
        result = UpstreamDriftLauncher._restart_sidekick_background_api(launcher)

    assert result is replacement_process
    assert launcher._sidekick_runtime_config is refreshed
    reselect.assert_called_once_with(original, os.environ)
    launcher._launch_sidekick_background_api.assert_called_once_with()
