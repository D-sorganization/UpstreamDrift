"""Regression: actionlint bootstrap must not trip Semgrep dynamic-urllib (merge_group)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/ci/run_actionlint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_actionlint_under_test", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_download_script_uses_curl_with_pinned_url() -> None:
    module = _load_module()
    with patch.object(module.subprocess, "run") as run:
        run.return_value = type("R", (), {"stdout": b"#!/bin/bash\n"})()
        body = module._fetch_download_script()
    assert body == b"#!/bin/bash\n"
    run.assert_called_once()
    cmd = run.call_args[0][0]
    assert cmd[0] == "curl"
    assert module.DOWNLOAD_SCRIPT_URL in cmd


def test_module_does_not_import_urllib() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "urllib" not in source
