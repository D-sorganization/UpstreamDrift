from __future__ import annotations

from pathlib import Path

import yaml


def test_frontend_compose_bootstraps_dependencies_before_dev_server() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    frontend = compose["services"]["frontend"]
    command = frontend["command"]

    assert "npm ci" in command
    assert command.index("npm ci") < command.index("npm run dev")
    assert any(volume.endswith(":/app/node_modules") for volume in frontend["volumes"])
