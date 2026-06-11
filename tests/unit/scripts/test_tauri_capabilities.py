from __future__ import annotations

import json
from pathlib import Path

from scripts.ci import check_tauri_capabilities as guard


def _write_tauri_tree(root: Path, permissions: list[str]) -> None:
    tauri_root = root / "ui" / "src-tauri"
    (tauri_root / "src").mkdir(parents=True)
    (tauri_root / "capabilities").mkdir()
    (tauri_root / "permissions").mkdir()
    (tauri_root / "src" / "lib.rs").write_text(
        """
tauri::Builder::default()
  .invoke_handler(tauri::generate_handler![
    start_backend,
    stop_backend,
    backend_status,
    get_diagnostics,
  ]);
""",
        encoding="utf-8",
    )
    (tauri_root / "capabilities" / "main.json").write_text(
        json.dumps(
            {
                "identifier": "main-capability",
                "windows": ["main"],
                "permissions": permissions,
            }
        ),
        encoding="utf-8",
    )
    (tauri_root / "permissions" / "backend-ipc.toml").write_text(
        """
[[permission]]
identifier = "allow-start-backend"
commands.allow = ["start_backend"]

[[permission]]
identifier = "allow-stop-backend"
commands.allow = ["stop_backend"]

[[permission]]
identifier = "allow-backend-status"
commands.allow = ["backend_status"]

[[permission]]
identifier = "allow-get-diagnostics"
commands.allow = ["get_diagnostics"]
""",
        encoding="utf-8",
    )


def test_tauri_capabilities_accept_exact_backend_ipc_surface(tmp_path: Path) -> None:
    _write_tauri_tree(
        tmp_path,
        [
            "core:default",
            "log:default",
            "allow-start-backend",
            "allow-stop-backend",
            "allow-backend-status",
            "allow-get-diagnostics",
        ],
    )

    assert guard.tauri_capability_failures(tmp_path) == []


def test_tauri_capabilities_reject_missing_command_permission(tmp_path: Path) -> None:
    _write_tauri_tree(
        tmp_path,
        [
            "core:default",
            "log:default",
            "allow-start-backend",
            "allow-stop-backend",
            "allow-backend-status",
        ],
    )

    assert "missing Tauri permission allow-get-diagnostics" in (
        guard.tauri_capability_failures(tmp_path)
    )


def test_tauri_capabilities_reject_missing_permission_definition(
    tmp_path: Path,
) -> None:
    _write_tauri_tree(
        tmp_path,
        [
            "core:default",
            "log:default",
            "allow-start-backend",
            "allow-stop-backend",
            "allow-backend-status",
            "allow-get-diagnostics",
        ],
    )
    (tmp_path / "ui" / "src-tauri" / "permissions" / "backend-ipc.toml").unlink()

    assert "missing Tauri permission definition allow-start-backend" in (
        guard.tauri_capability_failures(tmp_path)
    )
