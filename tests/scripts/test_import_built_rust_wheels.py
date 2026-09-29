from __future__ import annotations

from types import ModuleType
from typing import cast

import pytest

from scripts.ci.import_built_rust_wheels import smoke_module

pytestmark = pytest.mark.unit


class MissingFileMocapIo:
    @staticmethod
    def parse_trc(_path: object) -> None:
        raise FileNotFoundError("TRC file not found")


class BrokenMocapIo:
    @staticmethod
    def parse_trc(_path: object) -> None:
        raise RuntimeError("boom")


def test_mocap_io_smoke_accepts_missing_file_contract() -> None:
    smoke_module("upstream_mocap_io", cast(ModuleType, MissingFileMocapIo))


def test_mocap_io_smoke_rejects_unexpected_errors() -> None:
    with pytest.raises(RuntimeError, match="smoke failed unexpectedly"):
        smoke_module("upstream_mocap_io", cast(ModuleType, BrokenMocapIo))
