"""Regression tests for #12964: launcher logo route arbitrary file read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import launcher as launcher_routes
from src.api.utils.path_validation import resolve_file_within

PREFIXES = ("/launcher", "/api/launcher", "/api/v1/launcher")
_MOUNTS = ("", "/api", "/api/v1")
SECRET = "TOP-SECRET-CONTENT-12964"


@pytest.fixture
def logo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assets = tmp_path / "logos"
    assets.mkdir()
    (assets / "good.svg").write_text("<svg/>", encoding="utf-8")
    (assets / "notes.txt").write_text("not an image", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text(SECRET, encoding="utf-8")
    monkeypatch.setattr(launcher_routes, "ASSETS_DIR", assets)

    async def _no_manifest():
        class _M:
            tiles: list = []

        return _M()

    monkeypatch.setattr(launcher_routes, "_get_manifest_async", _no_manifest)
    app = FastAPI()
    for mount in _MOUNTS:
        app.include_router(launcher_routes.router, prefix=mount)
    return TestClient(app), assets, secret


def _assert_blocked(response) -> None:
    assert response.status_code in (400, 404)
    assert SECRET not in response.text
    assert "[fonts]" not in response.text.lower()


@pytest.mark.unit
@pytest.mark.parametrize("prefix", PREFIXES)
def test_legit_logo_still_served(logo_env, prefix: str) -> None:
    client, _, _ = logo_env
    response = client.get(f"{prefix}/logos/good.svg")
    assert response.status_code == 200
    assert response.text == "<svg/>"


@pytest.mark.unit
@pytest.mark.parametrize("prefix", PREFIXES)
def test_absolute_windows_path_blocked(logo_env, prefix: str) -> None:
    client, _, _ = logo_env
    _assert_blocked(client.get(f"{prefix}/logos/C:/Windows/win.ini"))


@pytest.mark.unit
@pytest.mark.parametrize("prefix", PREFIXES)
def test_absolute_posix_path_blocked(logo_env, prefix: str) -> None:
    client, _, secret = logo_env
    posix = secret.resolve().as_posix().lstrip("/")
    # "//abs/path" makes the {filename:path} capture start with "/".
    _assert_blocked(client.get(f"{prefix}/logos//{posix}"))
    _assert_blocked(client.get(f"{prefix}/logos//etc/passwd"))


@pytest.mark.unit
@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize(
    "payload",
    ["../secret.txt", "%2e%2e/secret.txt", "..%2fsecret.txt", "a/../../secret.txt"],
)
def test_dotdot_blocked(logo_env, prefix: str, payload: str) -> None:
    client, _, _ = logo_env
    _assert_blocked(client.get(f"{prefix}/logos/{payload}"))


@pytest.mark.unit
@pytest.mark.parametrize("prefix", PREFIXES)
def test_backslash_traversal_blocked(logo_env, prefix: str) -> None:
    client, _, _ = logo_env
    _assert_blocked(client.get(f"{prefix}/logos/..%5Csecret.txt"))
    _assert_blocked(client.get(f"{prefix}/logos/..\secret.txt"))


@pytest.mark.unit
@pytest.mark.parametrize("prefix", PREFIXES)
def test_symlink_escape_blocked(logo_env, prefix: str) -> None:
    client, assets, secret = logo_env
    link = assets / "escape.svg"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")
    _assert_blocked(client.get(f"{prefix}/logos/escape.svg"))


@pytest.mark.unit
@pytest.mark.parametrize("prefix", PREFIXES)
def test_disallowed_extension_blocked(logo_env, prefix: str) -> None:
    client, _, _ = logo_env
    assert client.get(f"{prefix}/logos/notes.txt").status_code == 404


@pytest.mark.unit
def test_resolve_file_within_contract(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    assert (
        resolve_file_within(tmp_path, "a.png", (".png",))
        == (tmp_path / "a.png").resolve()
    )
    assert (
        resolve_file_within(tmp_path, "a.PNG", (".png",)) is None
        or sys.platform == "win32"
    )
    assert resolve_file_within(tmp_path, "sub") is None  # directory
    assert resolve_file_within(tmp_path, "") is None
    assert resolve_file_within(tmp_path, "a\x00.png") is None
    assert resolve_file_within(tmp_path, str(tmp_path / "a.png")) is None
    assert resolve_file_within(tmp_path, "a.png", (".svg",)) is None
