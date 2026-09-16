"""Unit tests for configurable GSPro installation discovery and support receipts (GS-08, #10197).

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.shared.python.golf_simulator.discovery import (
    SimulatorEndpoint,
    SupportReceipt,
    discover_simulator_installation,
)

pytestmark = pytest.mark.unit


def test_simulator_endpoint_valid_loopback() -> None:
    endpoint = SimulatorEndpoint(host="127.0.0.1", port=921)
    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 921
    assert endpoint.url == "127.0.0.1:921"


def test_simulator_endpoint_rejects_non_loopback() -> None:
    with pytest.raises(
        ValueError, match="Vendor simulator socket must be restricted to loopback"
    ):
        SimulatorEndpoint(host="0.0.0.0", port=921)

    with pytest.raises(
        ValueError, match="Vendor simulator socket must be restricted to loopback"
    ):
        SimulatorEndpoint(host="192.168.1.50", port=921)


def test_simulator_endpoint_rejects_invalid_ports() -> None:
    with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
        SimulatorEndpoint(host="127.0.0.1", port=0)

    with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
        SimulatorEndpoint(host="127.0.0.1", port=70000)

    with pytest.raises(TypeError, match="Port cannot be a boolean"):
        SimulatorEndpoint(host="127.0.0.1", port=True)  # type: ignore[arg-type]


def test_discover_default_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear any ambient GSPRO_* env vars
    for key in (
        "GSPRO_INSTALL_DIR",
        "GSPRO_API_PORT",
        "GSPRO_HOST",
        "GSPRO_PROFILE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)

    config = discover_simulator_installation()
    assert config.endpoint.host == "127.0.0.1"
    assert config.endpoint.port == 921
    assert config.profile_version == "v1"
    assert config.discovery_method == "defaults"
    assert config.install_dir is None


def test_discover_custom_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom_install = tmp_path / "GSPro"
    custom_install.mkdir()
    monkeypatch.setenv("GSPRO_INSTALL_DIR", str(custom_install))
    monkeypatch.setenv("GSPRO_API_PORT", "922")
    monkeypatch.setenv("GSPRO_HOST", "localhost")
    monkeypatch.setenv("GSPRO_PROFILE_PATH", "v2")

    config = discover_simulator_installation()
    assert config.endpoint.host == "localhost"
    assert config.endpoint.port == 922
    assert config.profile_version == "v2"
    assert config.discovery_method == "environment"
    assert config.install_dir == custom_install


def test_support_receipt_contains_no_secrets() -> None:
    config = discover_simulator_installation()
    receipt = config.build_support_receipt(
        app_version="2.1.0",
        connector_version="1.0.0",
        optional_packages={"pywin32": False, "psutil": True},
    )
    assert isinstance(receipt, SupportReceipt)
    assert receipt.app_version == "2.1.0"
    assert receipt.connector_version == "1.0.0"
    assert receipt.profile_version == "v1"
    assert receipt.endpoint == "127.0.0.1:921"
    assert receipt.discovery_method == "defaults"
    assert receipt.optional_packages == {"pywin32": False, "psutil": True}

    as_dict = receipt.to_dict()
    # Check that common secret keys are absent
    secret_terms = ["token", "secret", "password", "key", "credential", "auth"]
    for term in secret_terms:
        assert term not in as_dict


def test_discover_from_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Clear env vars
    for key in (
        "GSPRO_INSTALL_DIR",
        "GSPRO_API_PORT",
        "GSPRO_HOST",
        "GSPRO_PROFILE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg_file = tmp_path / "gspro_config.json"
    cfg_file.write_text(
        '{"host": "localhost", "port": 925, "profile_version": "v2", "install_dir": "/opt/gspro"}'
    )

    config = discover_simulator_installation(config_path=cfg_file)
    assert config.endpoint.host == "localhost"
    assert config.endpoint.port == 925
    assert config.profile_version == "v2"
    assert config.discovery_method == "config_file"
    assert config.install_dir == Path("/opt/gspro")
