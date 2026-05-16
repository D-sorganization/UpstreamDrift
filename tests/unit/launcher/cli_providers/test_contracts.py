"""Tests for the CLI provider Pydantic contracts."""

from __future__ import annotations

import pytest

from src.shared.python.ai.cli_providers.contracts import (
    CliProviderConfig,
    CliProviderDescriptor,
)


def test_descriptor_roundtrip() -> None:
    desc = CliProviderDescriptor(
        id="claude-cli",
        name="Claude CLI",
        executable_path="/usr/local/bin/claude",
        transport="stdio",
        required_env=("ANTHROPIC_API_KEY",),
        working_dir_aware=True,
    )
    data = desc.model_dump()
    rebuilt = CliProviderDescriptor.model_validate(data)
    assert rebuilt == desc


def test_descriptor_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        CliProviderDescriptor(id="", name="x")


def test_descriptor_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CliProviderDescriptor(id="x", name="")


def test_descriptor_is_frozen() -> None:
    desc = CliProviderDescriptor(id="x", name="X")
    with pytest.raises(ValueError):
        desc.id = "y"  # type: ignore[misc]


def test_config_effective_working_dir_when_aware() -> None:
    desc = CliProviderDescriptor(id="x", name="X", working_dir_aware=True)
    cfg = CliProviderConfig(descriptor=desc, working_dir="/tmp/proj")
    assert cfg.effective_working_dir() == "/tmp/proj"


def test_config_effective_working_dir_when_not_aware() -> None:
    desc = CliProviderDescriptor(
        id="cline", name="Cline", working_dir_aware=False, transport="socket"
    )
    cfg = CliProviderConfig(descriptor=desc, working_dir="/tmp/proj")
    assert cfg.effective_working_dir() is None


def test_config_default_working_dir_is_none() -> None:
    desc = CliProviderDescriptor(id="x", name="X")
    cfg = CliProviderConfig(descriptor=desc)
    assert cfg.effective_working_dir() is None
